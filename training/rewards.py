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

    def objective(self, cell, held):
        """('exit' | 'key' | None, goal cell, distance field)."""
        inside = 0 <= cell[0] < self.h and 0 <= cell[1] < self.w
        if not inside:
            return None, None, None
        if self.exit is not None:
            f = self.field(self.exit, held)
            if np.isfinite(f[cell]):
                return "exit", self.exit, f
        best = None
        for kc, col in self.keys.items():
            if col in held:
                continue
            f = self.field(kc, held)
            if np.isfinite(f[cell]) and (best is None or f[cell] < best[2][cell]):
                best = ("key", kc, f)
        return best if best else (None, None, None)


class RewardShaper:
    W = dict(explore=0.04, kill=1.0, item=0.1, damage=0.01, key=4.0, door=0.5, unlock=4.0,
             exit=12.0, exit_health=3.0, exit_time=6.0, death=3.0, step=0.002, use_spam=0.003,
             hit=0.05, miss=0.02, dmg_dealt=0.003, weapon=0.5, revisit=0.004,
             # paid per 64-unit cell of true walking distance gained toward the next objective (and
             # taken back when it is lost again, so walking in circles earns exactly nothing)
             progress=0.06)

    def reset(self, ps: dict, layout=None) -> None:
        self.nav = Navigator(layout) if layout is not None else None
        self._obj, self._d, self._field = None, None, None
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
                          revisits=0, progress=0.0)
        self.last_use = 0

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
        self._ps = ps
        if self.nav is not None and not ps["finished"]:
            cell = self.nav.cell(ps["x"], ps["y"])
            held = self.nav.held(ps)
            kind, goal, field = self.nav.objective(cell, held)
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
        kind, goal, field = self.nav.objective(cell, self.nav.held(ps))
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
