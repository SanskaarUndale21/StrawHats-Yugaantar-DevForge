"""Reward shaping from privileged ground truth (training only, never visible to the agent).

Mirrors the evaluation categories: exit (largest), progression (keys, doors, unlocks), exploration
(unique areas), combat & survival (kills, damage, early death), plus a small time cost.
"""

from __future__ import annotations

import collections
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
    # The scorer pays for the exit most, but also for kills, new areas, keycards and doors, and the
    # episode ends at the exit; its time bonus is "small". So the time bonus here is small too and new
    # ground pays more than before: leaving with enemies alive and rooms unseen must not be the best deal.
    W = dict(explore=0.1, kill=1.5, item=0.1, damage=0.02, key=4.0, door=0.5, unlock=4.0,
             exit=12.0, exit_health=3.0, exit_time=1.0, death=4.5, step=0.004, use_spam=0.003,
             # survival (it died in a third to a half of gauntlets): damage and death cost more, and
             # health picked up while hurt pays per point regained
             heal=0.01,
             # aim and finishing the fight: the promoted policy hits with 74-77% of its shots and
             # kills 43% of the monsters (the teacher: 84% and 77%), so a hit, a miss and a kill each
             # weigh three times / 1.5 times what they did; every step costs twice as much (speed)
             hit=0.15, miss=0.06, dmg_dealt=0.003, weapon=0.5, revisit=0.008,
             # walking properly: reversing a turn or forward/back costs, and so does standing around
             # (measured on the copied policy: 12 left-right and 6.7 forward-back reversals per 100
             # steps, idle on 26-29% of steps)
             # (first values 0.004 / 0.004 / 0.001 were about 3% of an episode's return and changed
             # nothing; measured afterwards: 18% of all steps spent at a door, standing still on 70%
             # of them with 14 USE presses per 100 steps; 17 turn reversals per 100 steps while
             # walking; standing still on 46% of the steps with an enemy in view)
             # second measurement, 1M steps into these rewards: turn reversals at doors fell from 14.6 to
             # 3.2 per 100 steps and USE presses there halved, but it stands still on 78% of its steps
             # at a door (24% of all steps) and on 51% of its steps in a fight, so standing still in
             # those two situations now costs 2.5 times more
             turn_flip=0.02, move_flip=0.02, idle=0.006, idle_busy=0.02, blocked=0.004, use_waste=0.03,
             # paid per 64-unit cell of true walking distance gained toward the next objective (and
             # taken back when it is lost again, so walking in circles earns exactly nothing)
             progress=0.06)

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
                          revisits=0, progress=0.0, flips=0, idle=0, healed=0.0, blocked=0,
                          use_waste=0)
        self.last_use = 0
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
            self._leave = bool(done or ps["health"] < 35 or ps["tic"] > 0.40 * ps["episode_ticks"]
                               or (d is not None and left < d * 3.0 * 2.5))
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
