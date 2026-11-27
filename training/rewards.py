"""Reward shaping from privileged ground truth (training only, never visible to the agent).

Mirrors the evaluation categories: exit (largest), progression (keys, doors, unlocks), exploration
(unique areas), combat & survival (kills, damage, early death), plus a small time cost.
"""

from __future__ import annotations

import collections
import os
import math

import numpy as np

AREA = 128.0  # map units per exploration cell


def priv(env) -> dict:
    """privileged_state() plus the combat counters the shaper and the teacher use (training only)."""
    import vizdoom as vd

    GV = vd.GameVariable
    g = env._game
    ps = env.privileged_state()
    ps["hitcount"] = g.get_game_variable(GV.HITCOUNT)
    ps["dmg_dealt"] = g.get_game_variable(GV.DAMAGECOUNT)
    ps["ammo_sum"] = sum(g.get_game_variable(v) for v in (GV.AMMO2, GV.AMMO3, GV.AMMO5, GV.AMMO6))
    ps["weapons"] = [int(g.get_game_variable(getattr(GV, f"WEAPON{i}"))) for i in range(10)]
    return ps


CELL = 64.0
KEY_THINGS = {5: "B", 6: "Y", 13: "R"}
HELD_IDX = {"B": (0, 3), "Y": (1, 4), "R": (2, 5)}  # card, skull in privileged_state()["keys"]
PRIV_DIM = 12


class Navigator:
    """Ground-truth route knowledge for *training signals only* (reward shaping and the critic).
    Works out what the player has to reach next (the exit if it is reachable with the keys held,
    otherwise the nearest useful keycard) and the true walking distance to it from every cell."""

    def __init__(self, layout):
        self.g = layout.g
        self.h, self.w = layout.h, layout.w
        self.keys = {(r, c): KEY_THINGS[t] for t, r, c, _ in layout.things if t in KEY_THINGS}
        self.exit = next(((r, c) for r in range(self.h) for c in range(self.w) if self.g[r][c] == "E"), None)
        self.floor = sum(1 for row in self.g for ch in row if ch != "#")
        from mapgen import MONSTERS

        kinds = {m[0] for m in MONSTERS}
        self.n_monsters = sum(1 for t, _, _, _ in layout.things if t in kinds)
        # the 128-unit areas that contain floor (what the exploration score counts)
        self.area = {(int(math.floor((c * CELL + CELL / 2) / AREA)), int(math.floor((-(r + 1) * CELL + CELL / 2) / AREA)))
                     for r in range(self.h) for c in range(self.w) if self.g[r][c] != "#"}
        self._fields = {}

    @staticmethod
    def cell(x, y):
        return int(math.floor(-y / CELL)), int(math.floor(x / CELL))

    @staticmethod
    def center(rc):
        return rc[1] * CELL + CELL / 2, -(rc[0] + 1) * CELL + CELL / 2

    @staticmethod
    def held(ps):
        k = np.asarray(ps["keys"])
        return frozenset(col for col, idx in HELD_IDX.items() if k[idx[0]] > 0 or k[idx[1]] > 0)

    def field(self, goal, held):
        """Walking distance (cells) from every cell to `goal`; locked doors need their key."""
        key = (goal, held)
        f = self._fields.get(key)
        if f is None:
            f = np.full((self.h, self.w), np.inf, np.float32)
            f[goal] = 0.0
            q = collections.deque([goal])
            while q:
                r, c = q.popleft()
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    n = (r + dr, c + dc)
                    if 0 <= n[0] < self.h and 0 <= n[1] < self.w and not np.isfinite(f[n]):
                        ch = self.g[n[0]][n[1]]
                        if ch == "#" or (ch in HELD_IDX and ch not in held):
                            continue
                        f[n] = f[r, c] + 1.0
                        q.append(n)
            self._fields[key] = f
        return f

    def exit_distance(self, cell, held):
        """Walking distance (cells) to the exit, or None while it cannot be reached."""
        if self.exit is None or not (0 <= cell[0] < self.h and 0 <= cell[1] < self.w):
            return None
        d = float(self.field(self.exit, held)[cell])
        return d if np.isfinite(d) else None

    def objective(self, cell, held, leave=True):
        """('exit' | 'key' | None, goal cell, distance field). While the exit is reachable but it is
        not yet time to leave (`leave` False) there is no objective: the level's other goals come first."""
        inside = 0 <= cell[0] < self.h and 0 <= cell[1] < self.w
        if not inside:
            return None, None, None
        if self.exit is not None:
            f = self.field(self.exit, held)
            if np.isfinite(f[cell]):
                return ("exit", self.exit, f) if leave else (None, None, None)
        best = None
        for kc, col in self.keys.items():
            if col in held:
                continue
            f = self.field(kc, held)
            if np.isfinite(f[cell]) and (best is None or f[cell] < best[2][cell]):
                best = ("key", kc, f)
        return best if best else (None, None, None)


class RewardShaper:
    # These are the values the best policy so far was trained with (3.84M RL steps; 271-300 on three
    # fresh sets of 112 levels). Three later rounds of stronger shaping (leave the exit for last, 3x
    # aim, 5x movement penalties, 2x step cost) each made the same test worse: on one fresh set the
    # result went from 300 to 237, gauntlet exits from 44% to 19%, aim from 71% to 66% and gun swaps
    # from 1.4 to 10.3 a game. So the base values are back, and the newer terms stay only at a size
    # that cannot dominate them.
    W = dict(explore=0.08, kill=1.0, item=0.1, damage=0.02, key=4.0, door=0.5, unlock=4.0,
             exit=12.0, exit_health=3.0, exit_time=6.0, death=4.5, step=0.002, use_spam=0.003,
             heal=0.01,
             # Standing and turning on the spot (step log of the promoted policy, 36 games): 70% of its
             # left-right reversals are ones the network is over 75% sure of, 54% come back-to-back, and
             # it stands on 48% of steps, over 75% sure on 97% of them. So this is learned, not sampling
             # noise, and the old rewards explain it: re-walking a visited cell cost up to 0.032 while
             # standing cost 0.002 a step, so in explored space not moving was the cheapest thing to do,
             # and reversing a turn cost almost nothing. Now standing costs more than walking old ground,
             # and a reversal costs as much as it did in the one run where reversals fell (28 -> 17).
             # Exit memory, measured on 72 games: every time the pad came into view it was stored (20
             # units off, typically) and the exit was then reached in 100% of those games; in the games
             # where it never came into view the exit was reached in 4%. So the losses are levels where
             # the exit is never found, and 43% of all cell-to-cell moves were onto ground already
             # walked. New ground pays double and old ground costs three times more, kept just under
             # the cost of standing still (about 5 steps per cell).
             hit=0.05, miss=0.02, dmg_dealt=0.003, weapon=0.5, revisit=0.006,
             # blocked must cost more than idle: at 0.002 against 0.006, walking into a wall was the cheap
             # way out of the standing penalty (blocked steps went from 4% to 8% and it was seen spending a
             # whole game pressed against a wall)
             turn_flip=0.02, move_flip=0.02, idle=0.004, idle_busy=0.004, blocked=0.008, use_waste=0.01,
             swap=0.01,  # per press of "next weapon": nothing else discouraged cycling guns for no reason
             # Freezes (60 key/gauntlet games, 4-5 with a motionless stretch of 100-665 steps): in every one
             # the policy stood still holding FIRE on 98-100% of the steps and pressed nothing else, at a
             # target 140-360 units away, twice with bare fists, once for 665 steps at 13 health. Punching
             # uses no ammo, so nothing counted those as misses. Firing that has landed nothing for
             # 6 steps in a row now costs every step until something is hit or the trigger is released.
             futile_fire=0.02,
             # Going nowhere, whatever the keys: under 64 units of net movement over the last 20 steps.
             # A tight circle is not standing, not pushing a wall, has no reversal and stays inside one
             # cell, so no other term sees it (measured: 4% of steps walking in circles, 4% standing,
             # 2% shuffling). Half price with an enemy in view, where holding ground can be right.
             loiter=0.008,
             progress=0.06)
    # one-run experiments without a code fork: REWARD_OVERRIDES="dmg_dealt=0.01,damage=0.035"
    for _kv in filter(None, os.environ.get("REWARD_OVERRIDES", "").split(",")):
        _k, _v = _kv.split("=")
        assert _k in W, _k
        W[_k] = float(_v)

    def reset(self, ps: dict, layout=None) -> None:
        self.nav = Navigator(layout) if layout is not None else None
        self._obj, self._d, self._field = None, None, None
        self._leave = False
        self.kills0 = ps["kills"]
        self._ps = ps
        self.cells = {self._cell(ps)}
        self.visits = {self._cell(ps): 1}
        self.cur_cell = self._cell(ps)
        self.hit, self.dealt = ps.get("hitcount", 0.0), ps.get("dmg_dealt", 0.0)
        self.ammo, self.nweap = ps.get("ammo_sum", 0.0), sum(ps.get("weapons", []))
        self.kills, self.items, self.damage = ps["kills"], ps["items"], ps["damage_taken"]
        self.keys = np.asarray(ps["keys"]).copy()
        dl = ps["door_lock"]
        self.door_ids = np.flatnonzero(dl >= 0) if dl is not None else np.zeros(0, int)
        self.door_lock = dl
        self.opened = set()
        self.stats = dict(kills=0, keys=0, doors=0, unlocks=0, cells=1, exit=0, death=0, damage=0.0, items=0, shots=0, hits=0,
                          revisits=0, progress=0.0, flips=0, idle=0, healed=0.0, blocked=0, futile=0, loiter=0,
                          use_waste=0)
        self.last_use = 0
        self.fire_run = 0
        self.trail = collections.deque([(ps["x"], ps["y"])], maxlen=21)
        self.hit_seen = ps.get("hitcount", 0.0)
        self.since_use = 1000
        self.px, self.py = ps["x"], ps["y"]
        self.health = ps["health"]
        self.last_turn = self.last_move = 0

    @staticmethod
    def _cell(ps):
        return (math.floor(ps["x"] / AREA), math.floor(ps["y"] / AREA))

    def __call__(self, ps: dict, action: np.ndarray) -> float:
        W, st = self.W, self.stats
        r = -W["step"]
        c = self._cell(ps)
        if c not in self.cells:
            self.cells.add(c)
            st["cells"] += 1
            r += W["explore"]
        if c != self.cur_cell:  # walked into another cell: old ground costs, more each time it is re-walked
            n = self.visits.get(c, 0)
            if n:
                r -= W["revisit"] * min(n, 4)
                st["revisits"] += 1
            self.visits[c] = n + 1
            self.cur_cell = c
        if "hitcount" in ps:  # aim: a shot that lands pays, a shot into the wall costs
            fired = ps["ammo_sum"] < self.ammo
            if fired:
                st["shots"] += 1
                if ps["hitcount"] > self.hit:
                    st["hits"] += 1
                    r += W["hit"]
                else:
                    r -= W["miss"]
            r += W["dmg_dealt"] * min(max(ps["dmg_dealt"] - self.dealt, 0.0), 300.0)  # better guns hit harder
            nw = sum(ps["weapons"])
            if nw > self.nweap:
                r += W["weapon"] * (nw - self.nweap)
            self.hit, self.dealt, self.ammo, self.nweap = ps["hitcount"], ps["dmg_dealt"], ps["ammo_sum"], nw
        gained = ps["health"] - self.health
        if gained > 0 and self.health < 60:
            r += W["heal"] * min(gained, 50.0)
            st["healed"] += gained
        self.health = ps["health"]
        turn, move = int(action[4]) - int(action[5]), int(action[0]) - int(action[1])
        if turn and self.last_turn and turn != self.last_turn:
            r -= W["turn_flip"]
            st["flips"] += 1
        if move and self.last_move and move != self.last_move:
            r -= W["move_flip"]
            st["flips"] += 1
        self.last_turn, self.last_move = turn or self.last_turn, move or self.last_move
        if "hitcount" in ps:
            landed = ps["hitcount"] > self.hit_seen
            self.hit_seen = ps["hitcount"]
            self.fire_run = 0 if (landed or not action[6]) else self.fire_run + 1
            if self.fire_run > 6:
                r -= W["futile_fire"]
                st["futile"] += 1
        self.trail.append((ps["x"], ps["y"]))
        if len(self.trail) == 21 and math.hypot(ps["x"] - self.trail[0][0], ps["y"] - self.trail[0][1]) < 64.0:
            r -= W["loiter"] * (0.5 if ps.get("enemy_visible") else 1.0)
            st["loiter"] += 1
        pressing = bool(action[0] or action[1] or action[2] or action[3])
        if not pressing:  # keep moving: more so in a fight (dodge) and at a door (go through it)
            r -= W["idle_busy"] if (ps.get("enemy_visible") or ps.get("door_ahead")) else W["idle"]
            st["idle"] += 1
        elif math.hypot(ps["x"] - self.px, ps["y"] - self.py) < 3.0:
            r -= W["blocked"]  # pushing into a wall, a door or a monster and going nowhere
            st["blocked"] += 1
        self.px, self.py = ps["x"], ps["y"]
        if action[7] and (not ps.get("door_ahead", True) or self.since_use < 8):
            r -= W["use_waste"]  # USE with no door in front, or again while the door is still moving
            st["use_waste"] += 1
        self.since_use = 0 if action[7] else self.since_use + 1
        if action[8]:
            r -= W["swap"]
        self._ps = ps
        if self.nav is not None and not ps["finished"]:
            cell = self.nav.cell(ps["x"], ps["y"])
            held = self.nav.held(ps)
            kind, goal, field = self.nav.objective(cell, held, self._time_to_leave(ps, cell, held))
            obj = (kind, goal, held)
            d = float(field[cell]) if kind else None
            if kind and obj == self._obj and self._d is not None:
                gain = max(-2.0, min(2.0, self._d - d))
                r += W["progress"] * gain
                st["progress"] += gain
            self._obj, self._d, self._field = obj, d, field
        dk = ps["kills"] - self.kills
        if dk > 0:
            r += W["kill"] * dk
            st["kills"] += dk
        di = ps["items"] - self.items
        if di > 0:
            r += W["item"] * di
            st["items"] += di
        dd = ps["damage_taken"] - self.damage
        if dd > 0:
            r -= W["damage"] * dd
            st["damage"] += dd
        self.kills, self.items, self.damage = ps["kills"], ps["items"], ps["damage_taken"]
        keys = np.asarray(ps["keys"])
        nk = int(((keys > 0) & (self.keys == 0)).sum())
        if nk:
            r += W["key"] * nk
            st["keys"] += nk
        self.keys = keys.copy()
        if len(self.door_ids) and ps["sector_ceiling"] is not None:
            gap = ps["sector_ceiling"][self.door_ids] - ps["sector_floor"][self.door_ids]
            for sid in self.door_ids[gap > 40]:
                if sid not in self.opened:
                    self.opened.add(int(sid))
                    if self.door_lock[sid] > 0:
                        r += W["unlock"]
                        st["unlocks"] += 1
                    else:
                        r += W["door"]
                        st["doors"] += 1
        # discourage hammering USE every step (it bounces doors back)
        if action[7] and self.last_use:
            r -= W["use_spam"]
        self.last_use = int(action[7])
        if ps["finished"]:
            frac = ps["tic"] / max(ps["episode_ticks"], 1)
            if ps["exit_reached"]:
                r += W["exit"] + W["exit_health"] * min(ps["health"], 100) / 100 + W["exit_time"] * (1 - frac)
                st["exit"] = 1
            elif ps["dead"]:
                r -= W["death"] * (1.5 - frac)
                st["death"] = 1
        return r

    def _time_to_leave(self, ps, cell, held) -> bool:
        """Head for the exit only when the level is done (all enemies dead and nearly all of it walked),
        or when staying is the bigger risk: low health, over half the time gone, or just enough time
        left for the way back. Once true it stays true."""
        if not self._leave:
            left = (ps["episode_ticks"] - ps["tic"]) / 4.0  # decision steps left
            d = self.nav.exit_distance(cell, held)
            done = ps["kills"] - self.kills0 >= self.nav.n_monsters and len(self.cells) >= 0.85 * len(self.nav.area)
            # Waiting for the level to be "finished" before leaving cost exits on the long levels
            # (112-level test 258 -> 203 over the 4M steps it was in force), and the exit is the
            # largest score item. The exit is the objective as soon as it can be reached.
            self._leave = True
            if os.environ.get("FINISH_FIRST") == "1":  # experiment: clear the level first, unless staying is the bigger risk
                self._leave = bool(ps["kills"] - self.kills0 >= self.nav.n_monsters or ps["health"] < 40
                                   or ps["tic"] > 0.40 * ps["episode_ticks"])
        return self._leave

    # ------------------------------------------------------------------ training-only side information
    def priv_features(self) -> np.ndarray:
        """What the critic may know and the policy may not: the objective, the true distance to it and
        which way the route goes (in the player's own frame)."""
        f = np.zeros(PRIV_DIM, np.float32)
        ps = self._ps
        f[10] = min(ps["tic"] / max(ps["episode_ticks"], 1), 1.0)
        f[11] = min(max(ps["health"], 0.0), 200.0) / 100.0
        if self.nav is None:
            return f
        f[0] = 1.0
        cell = self.nav.cell(ps["x"], ps["y"])
        kind, goal, field = self.nav.objective(cell, self.nav.held(ps), self._leave)
        f[1], f[2], f[3] = kind == "exit", kind == "key", kind is None
        f[9] = min(len(self.cells) / max(self.nav.floor / 4.0, 1.0), 2.0)
        if kind is None:
            return f
        f[4] = min(float(field[cell]) / 40.0, 2.0)
        cur = cell
        for _ in range(3):  # three cells down the shortest route
            nb = [(cur[0] + dr, cur[1] + dc) for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1))]
            nb = [n for n in nb if 0 <= n[0] < self.nav.h and 0 <= n[1] < self.nav.w and field[n] < field[cur]]
            if not nb:
                break
            cur = min(nb, key=lambda n: field[n])
        th = math.radians(ps["angle"])
        for k, tgt in ((5, self.nav.center(cur)), (7, self.nav.center(goal))):
            dx, dy = tgt[0] - ps["x"], tgt[1] - ps["y"]
            n = math.hypot(dx, dy)
            if n > 1.0:
                f[k] = (math.cos(th) * dx + math.sin(th) * dy) / n       # ahead
                f[k + 1] = (-math.sin(th) * dx + math.cos(th) * dy) / n  # to the left
        return f

    def exit_targets(self, obs: dict) -> np.ndarray:
        """Supervision for the exit detector: [pad in view, bearing / 45 deg, range / 1024]."""
        out = np.zeros(3, np.float32)
        if self.nav is None or self.nav.exit is None:
            return out
        ps = self._ps
        ex, ey = self.nav.center(self.nav.exit)
        dx, dy = ex - ps["x"], ey - ps["y"]
        dist = math.hypot(dx, dy)
        bearing = (math.atan2(dy, dx) - math.radians(ps["angle"]) + math.pi) % (2 * math.pi) - math.pi
        if dist < 24 or dist > 1100 or abs(bearing) > math.radians(43):
            return out
        rd = np.asarray(obs["rays"]["distance"])
        i = int(round(bearing / (2 * math.pi / len(rd)))) % len(rd)
        if rd[i] * 1024.0 < dist - 40.0:  # something solid is in the way
            return out
        out[:] = (1.0, bearing / math.radians(45), dist / 1024.0)
        return out

    def color_targets(self, obs: dict) -> np.ndarray:
        """Supervision for the colour reader: [colour of the locked door ahead, colour of the keycard
        in view], each 0 none, 1 blue, 2 yellow, 3 red."""
        out = np.zeros(2, np.float32)
        if self.nav is None:
            return out
        ps = self._ps
        th = math.radians(ps["angle"])
        code = {"B": 1.0, "Y": 2.0, "R": 3.0}
        hit, rd = np.asarray(obs["rays"]["hit_type"]), np.asarray(obs["rays"]["distance"])
        for i in (0, 1, 63, 2, 62, 3, 61):
            if hit[i] == 3 and rd[i] < 0.6:
                a = th + i * 2 * math.pi / len(rd)
                for extra in (8.0, 32.0):
                    r, c = self.nav.cell(ps["x"] + (rd[i] * 1024.0 + extra) * math.cos(a), ps["y"] + (rd[i] * 1024.0 + extra) * math.sin(a))
                    if 0 <= r < self.nav.h and 0 <= c < self.nav.w and self.nav.g[r][c] in code:
                        out[0] = code[self.nav.g[r][c]]
                        break
                break
        e = obs["visible_entities"]
        m = np.asarray(e["mask"]).astype(bool) & (np.asarray(e["entity_type"]) == 9)
        f = np.asarray(e["features"], np.float64)
        m &= f[:, 2] > 0.73
        if m.any():
            j = int(np.flatnonzero(m)[np.argmin(f[m, 0])])
            d = f[j, 0] * 1024.0
            fwd, left = d * f[j, 2], d * f[j, 1]
            kx, ky = ps["x"] + math.cos(th) * fwd - math.sin(th) * left, ps["y"] + math.sin(th) * fwd + math.cos(th) * left
            best = min(self.nav.keys, key=lambda rc: math.hypot(self.nav.center(rc)[0] - kx, self.nav.center(rc)[1] - ky), default=None)
            if best is not None and math.hypot(self.nav.center(best)[0] - kx, self.nav.center(best)[1] - ky) < 80:
                out[1] = code[self.nav.keys[best]]
        return out


def aux_targets(obs: dict) -> np.ndarray:
    """Self-supervised targets: [enemy near the crosshair, door/locked door straight ahead]."""
    e = obs["visible_entities"]
    m = np.asarray(e["mask"]).astype(bool)
    t = np.asarray(e["entity_type"])
    f = np.asarray(e["features"])
    enemy = m & (t >= 1) & (t <= 3) & (np.abs(f[:, 1]) < 0.08) & (f[:, 2] > 0)
    hit = np.asarray(obs["rays"]["hit_type"])
    d = np.asarray(obs["rays"]["distance"])
    front = [62, 63, 0, 1, 2]
    door = np.any(((hit[front] == 2) | (hit[front] == 3)) & (d[front] < 0.25))
    return np.array([float(enemy.any()), float(door)], np.float32)
