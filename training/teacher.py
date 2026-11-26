"""Privileged scripted "pro gamer" teacher for DAgger distillation (TRAINING ONLY, never at evaluation).

The teacher reads ground truth the agent never gets (true pose from `privileged_state()`, the level
layout produced by mapgen), but it only *plans over cells it has already seen* through the same 64 rays
the agent sees, so its behaviour stays explainable from the agent's own observation history (a god-view
teacher that walks straight to an unseen keycard would teach the student behaviour it cannot explain).

Behaviour, in priority order every step:
  combat     nearest/most dangerous visible enemy: predictive aim (knows its own keyboard turn speed so it
             does not overshoot), fire when on target, circle-strafe, keep range (back off from melee,
             close in when too far), retreat while shooting when health is low, upgrade weapon
  damage     shot by something it cannot see: turn around and scan
  pickups    detour to visible weapons / ammo / armour / health when useful and close
  exit       exit pad seen and reachable (locks open with held keys)
  key        a seen locked door blocks progress and a matching keycard has been seen
  explore    information-gain frontier: nearest frontier weighted by how much unexplored floor it reveals
  movement   corridor-aware path following (clearance check for the player's 16-unit radius),
             walks into doors and presses USE once, stuck detector with short escapes

    python training/teacher.py --episodes 20 --stage 2      # teacher success on generated maps
"""

from __future__ import annotations

import math
import random
from collections import deque

import numpy as np

CELL = 64.0
LOCK_KEYS = {"B": (8, 11), "Y": (9, 12), "R": (10, 13)}  # vitals indices (card, skull)
KEY_THINGS = {5: "B", 6: "Y", 13: "R"}
ENEMY_TYPES = (1, 2, 3)
MELEE, HITSCAN, PROJ = 1, 2, 3
ITEM_HEALTH, ITEM_ARMOR, ITEM_AMMO, ITEM_WEAPON = 5, 6, 7, 8
RAY_ANG = np.arange(64) * (2 * math.pi / 64)
SAMPLES = np.arange(16.0, 1024.0 + 1, 32.0)
SLOW, FAST = 1.7578125, 3.515625


def buttons_to_choice(b) -> np.ndarray:
    b = np.asarray(b)
    move = 0 if b[0] and not b[1] else (2 if b[1] and not b[0] else 1)
    strafe = 0 if b[2] and not b[3] else (2 if b[3] and not b[2] else 1)
    turn = 0 if b[4] and not b[5] else (2 if b[5] and not b[4] else 1)
    return np.array([move, strafe, turn, int(b[6]), int(b[7]), int(b[8])], np.int64)


def wrap_deg(a):
    return (a + 180.0) % 360.0 - 180.0


class Teacher:
    def __init__(self, layout, seed: int = 0):
        self.L = layout
        self.h, self.w = layout.h, layout.w
        self.grid = layout.g
        self.wall = np.array([[ch == "#" for ch in row] for row in self.grid])
        self.keys = {(r, c): KEY_THINGS[t] for t, r, c, _ in layout.things if t in KEY_THINGS}
        self.weapon_cells = {(r, c) for t, r, c, _ in layout.things if t in (2001, 2002)}  # shotgun, chaingun
        self.exit = next(((r, c) for r in range(self.h) for c in range(self.w) if self.grid[r][c] == "E"), None)
        self.rng = random.Random(seed)
        self.reset()

    def reset(self):
        self.seen = np.zeros((self.h, self.w), bool)
        self.seen_front = np.zeros((self.h, self.w), bool)  # seen inside the 90 degree field of view
        self.taken = set()
        self.path = []
        self.goal = None
        self.replan = 0
        self.hist = deque(maxlen=10)
        self.escape = 0
        self.escape_act = None
        self.use_cool = 0
        self.prev_use = 0
        self.wpn_cool = 0
        self.strafe_dir = 2
        self.strafe_t = 0
        self.turn_held = 0      # consecutive tics a turn key has been held (engine turn profile)
        self.prev_damage = None
        self.scan = 0
        self.item_target = None
        self.item_t = 0
        self.owns_better = False
        self.prev_wpn = 0

    # ------------------------------------------------------------------ geometry
    def cell_of(self, x, y):
        return int(math.floor(-y / CELL)), int(math.floor(x / CELL))

    @staticmethod
    def center(rc):
        r, c = rc
        return c * CELL + CELL / 2, -(r * CELL + CELL / 2)

    def _in(self, r, c):
        return 0 <= r < self.h and 0 <= c < self.w

    def _observe(self, obs, x, y, th):
        d = np.asarray(obs["rays"]["distance"], np.float64) * 1024.0
        ang = th + RAY_ANG
        m = SAMPLES[None, :] <= d[:, None] + 8.0
        px = (x + np.cos(ang)[:, None] * SAMPLES[None, :])[m]
        py = (y + np.sin(ang)[:, None] * SAMPLES[None, :])[m]
        hx, hy = x + np.cos(ang) * (d + 8), y + np.sin(ang) * (d + 8)
        rr = np.floor(-np.concatenate([py, hy]) / CELL).astype(int)
        cc = np.floor(np.concatenate([px, hx]) / CELL).astype(int)
        ok = (rr >= 0) & (rr < self.h) & (cc >= 0) & (cc < self.w)
        self.seen[rr[ok], cc[ok]] = True
        # what the student can also recognise (pixels / visible_entities): front rays only, <= 900 units
        fr = np.r_[57:64, 0:8]
        mf = SAMPLES[None, :] <= np.minimum(d[fr], 900.0)[:, None]
        fx = (x + np.cos(ang[fr])[:, None] * SAMPLES[None, :])[mf]
        fy = (y + np.sin(ang[fr])[:, None] * SAMPLES[None, :])[mf]
        fr_, fc_ = np.floor(-fy / CELL).astype(int), np.floor(fx / CELL).astype(int)
        okf = (fr_ >= 0) & (fr_ < self.h) & (fc_ >= 0) & (fc_ < self.w)
        self.seen_front[fr_[okf], fc_[okf]] = True
        r0, c0 = self.cell_of(x, y)
        self.seen_front[r0, c0] = True
        self.seen[max(r0 - 1, 0):r0 + 2, max(c0 - 1, 0):c0 + 2] = True

    def _has_key(self, vit, col):
        a, b = LOCK_KEYS[col]
        return vit[a] > 0 or vit[b] > 0

    def _passable(self, r, c, vit):
        ch = self.grid[r][c]
        if ch == "#" or not self.seen[r, c]:
            return False
        if ch in LOCK_KEYS:
            return self._has_key(vit, ch)
        return True

    def _bfs_all(self, start, vit):
        prev = {start: None}
        dist = {start: 0}
        q = deque([start])
        while q:
            cur = q.popleft()
            r, c = cur
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                n = (r + dr, c + dc)
                if n not in prev and self._in(*n) and self._passable(*n, vit):
                    prev[n] = cur
                    dist[n] = dist[cur] + 1
                    q.append(n)
        return prev, dist

    @staticmethod
    def _path(prev, goal, start):
        path = []
        while goal != start:
            path.append(goal)
            goal = prev[goal]
        return path[::-1]

    def _frontier_gain(self):
        """Seen floor cells next to unseen *walkable* space, with how much unseen floor is around them."""
        unseen_floor = (~self.seen) & (~self.wall)
        out = {}
        for r, c in zip(*np.nonzero(self.seen & ~self.wall)):
            r0, r1, c0, c1 = max(r - 1, 0), r + 2, max(c - 1, 0), c + 2
            if unseen_floor[r0:r1, c0:c1].any():
                out[(int(r), int(c))] = int(unseen_floor[max(r - 3, 0):r + 4, max(c - 3, 0):c + 4].sum())
        return out

    def _plan(self, start, vit):
        prev, dist = self._bfs_all(start, vit)
        if self.exit is not None and self.exit in dist and self.seen_front[self.exit]:
            return self._path(prev, self.exit, start), "exit"
        need = {ch for r in range(self.h) for c in range(self.w)
                if (ch := self.grid[r][c]) in LOCK_KEYS and self.seen[r, c] and not self._has_key(vit, ch)}
        want = [rc for rc, col in self.keys.items() if rc in dist and rc not in self.taken and self.seen_front[rc]
                and (col in need or not self._has_key(vit, col))]
        if want:
            g = min(want, key=dist.get)
            return self._path(prev, g, start), "key"
        fr = {rc: gain for rc, gain in self._frontier_gain().items() if rc in dist and rc != start}
        if fr:
            g = min(fr, key=lambda rc: dist[rc] - 0.35 * fr[rc])
            return self._path(prev, g, start), "explore"
        return None, None

    def _clear(self, x0, y0, x1, y1, radius=18.0):
        """Straight-line walkability with the player's radius (three parallel lines)."""
        dx, dy = x1 - x0, y1 - y0
        n = math.hypot(dx, dy)
        if n < 1:
            return True
        ox, oy = -dy / n * radius, dx / n * radius
        steps = int(n / 12) + 1
        for sx, sy in ((0, 0), (ox, oy), (-ox, -oy)):
            for k in range(steps + 1):
                t = k / steps
                r, c = self.cell_of(x0 + sx + dx * t, y0 + sy + dy * t)
                if not self._in(r, c) or self.wall[r, c]:
                    return False
        return True

    # ------------------------------------------------------------------ low-level control
    def _turn_towards(self, b, err_deg, tol=2.0):
        """Turn so the heading error shrinks without overshooting: the engine turns 7 deg on the first
        held step, ~12 on the second and ~14 after, so only turn when the error exceeds half a step."""
        held = self.turn_held
        nxt = sum(SLOW if held + k + 1 <= 5 else FAST for k in range(4))
        if abs(err_deg) > max(tol, nxt * 0.5):
            if err_deg > 0:
                b[4] = 1
            else:
                b[5] = 1

    def _finish(self, b):
        self.turn_held = self.turn_held + 4 if (b[4] or b[5]) else 0
        self.prev_use = int(b[7])
        return b

    # ------------------------------------------------------------------ act
    def act(self, obs, ps) -> np.ndarray:
        x, y, th = ps["x"], ps["y"], math.radians(ps["angle"])
        vit = np.asarray(obs["vitals"])
        self._observe(obs, x, y, th)
        cur = self.cell_of(x, y)
        self.taken.add(cur)
        if cur in self.weapon_cells:
            self.owns_better = True
        b = np.zeros(9, np.int8)
        self.use_cool -= 1
        self.wpn_cool -= 1
        health = vit[0]

        dmg = ps["damage_taken"]
        hurt = self.prev_damage is not None and dmg > self.prev_damage
        self.prev_damage = dmg

        # ---- stuck escape
        self.hist.append((x, y))
        if self.escape > 0:
            self.escape -= 1
            return self._finish(self.escape_act.copy())

        ents = obs["visible_entities"]
        mask = np.asarray(ents["mask"]).astype(bool)
        et = np.asarray(ents["entity_type"])
        f = np.asarray(ents["features"])
        enemies = np.flatnonzero(mask & np.isin(et, ENEMY_TYPES) & (f[:, 0] < 2.0))

        if len(self.hist) == self.hist.maxlen and not len(enemies):
            x0, y0 = self.hist[0]
            if math.hypot(x - x0, y - y0) < 20 and self.goal is not None:
                self.escape = self.rng.randint(3, 6)
                e = np.zeros(9, np.int8)
                e[self.rng.choice([2, 3])] = 1
                e[1] = 1
                e[self.rng.choice([4, 5])] = self.rng.random() < 0.5
                self.escape_act = e
                self.hist.clear()
                self.path = []
                return self._finish(e.copy())

        # ---- weapon: upgrade from fist/pistol when shells/bullets for something better exist
        # weapon: never sit on the fist with bullets left; try to upgrade pistol -> shotgun when we have
        # shells (SELECT_NEXT cycles owned weapons; if it wraps to the fist, the first rule cycles on)
        if vit[2] <= 1 and vit[4:8].sum() > 0 and not self.prev_wpn:
            b[8] = 1  # on the fist with ammo left: cycle on immediately
        elif self.wpn_cool <= 0:
            if vit[3] <= 0 and vit[4:8].sum() > 0:
                b[8] = 1
                self.wpn_cool = 3
            elif vit[2] == 2 and self.owns_better and (vit[5] > 0 or vit[4] > 0):
                b[8] = 1
                self.wpn_cool = 40
        self.prev_wpn = int(b[8])

        # ---- combat
        if len(enemies) and (vit[3] > 0 or vit[2] <= 1):
            # threat: close enemies first, hitscan/projectile users weigh a bit more
            threat = [f[i, 0] * (0.8 if et[i] in (HITSCAN, PROJ) else 1.0) for i in enemies]
            i = enemies[int(np.argmin(threat))]
            dist = f[i, 0] * 1024.0
            err = math.degrees(math.atan2(f[i, 1], f[i, 2]))
            tol = max(2.5, math.degrees(math.atan2(18.0, max(dist, 1.0))))
            self._turn_towards(b, err, tol=tol)
            if abs(err) < tol + 3.0:
                b[6] = 1
            # movement: circle-strafe, keep range, retreat when hurt
            self.strafe_t -= 1
            if self.strafe_t <= 0:
                self.strafe_dir = 2 if self.strafe_dir == 3 else 3
                self.strafe_t = self.rng.randint(5, 12)
            rd = np.asarray(obs["rays"]["distance"])
            left_room, right_room = rd[14:19].min(), rd[46:51].min()
            here = left_room if self.strafe_dir == 2 else right_room
            there = right_room if self.strafe_dir == 2 else left_room
            if here < 0.06 and there > here:
                self.strafe_dir = 2 if self.strafe_dir == 3 else 3
                self.strafe_t = self.rng.randint(5, 12)
            if max(left_room, right_room) > 0.06:
                b[self.strafe_dir] = 1
            back_clear = np.asarray(obs["rays"]["distance"])[32] > 0.1
            outnumbered = len(enemies) >= 3
            if (et[i] == MELEE and dist < 220) or ((health < 50 or outnumbered) and back_clear):
                b[1] = 1
            elif (dist > 700 or (et[i] == HITSCAN and dist > 380)) and abs(err) < 30 and health >= 50:
                b[0] = 1  # close the distance: pistol/shotgun spread is useless at long range
            self.scan = 0
            return self._finish(b)

        # ---- shot by something unseen: turn around and look
        if hurt and self.scan <= 0:
            self.scan = 8
        if self.scan > 0:
            self.scan -= 1
            b[5] = 1
            b[self.strafe_dir] = 1
            return self._finish(b)

        # ---- pickups in view (gamer instinct): weapons always, ammo/armour/health when useful
        want_types = [ITEM_WEAPON, ITEM_ARMOR]
        if health < 90:
            want_types.append(ITEM_HEALTH)
        if vit[4:8].sum() < 120 or vit[3] < 15:
            want_types.append(ITEM_AMMO)
        items = np.flatnonzero(mask & np.isin(et, want_types) & (f[:, 0] < 0.6))
        if len(items):
            i = items[0]
            ix = x + f[i, 0] * 1024 * (math.cos(th) * f[i, 2] - math.sin(th) * f[i, 1])
            iy = y + f[i, 0] * 1024 * (math.sin(th) * f[i, 2] + math.cos(th) * f[i, 1])
            if self._clear(x, y, ix, iy, radius=12.0):
                err = math.degrees(math.atan2(f[i, 1], f[i, 2]))
                self._turn_towards(b, err, tol=5.0)
                if abs(err) < 35:
                    b[0] = 1
                self.goal = "item"
                return self._finish(b)

        # ---- navigation
        self.replan -= 1
        while self.path and self.path[0] == cur:
            self.path.pop(0)
        if not self.path or self.replan <= 0:
            self.path, self.goal = self._plan(cur, vit)
            self.path = self.path or []
            self.replan = 8
        if not self.path:
            b[0] = 1
            b[4] = 1
            return self._finish(b)
        nxt = self.path[0]
        door_next = self.grid[nxt[0]][nxt[1]] in "DBYR"
        tgt = nxt
        if not door_next:
            for k in range(min(5, len(self.path))):
                p = self.path[k]
                if self.grid[p[0]][p[1]] in "DBYR":
                    break
                cx, cy = self.center(p)
                if self._clear(x, y, cx, cy):
                    tgt = p
        tx, ty = self.center(tgt)
        err = wrap_deg(math.degrees(math.atan2(ty - y, tx - x) - th))
        self._turn_towards(b, err, tol=4.0)
        if abs(err) < 50:
            b[0] = 1
        elif abs(err) < 100:
            b[2 if err > 0 else 3] = 1  # strafe-turn round corners instead of stopping
        # doors: face the door and press USE once, then wait for it to open
        if door_next:
            hit = np.asarray(obs["rays"]["hit_type"])
            dist = np.asarray(obs["rays"]["distance"])
            dx, dy = tx - x, ty - y
            close = math.hypot(dx, dy) < 110
            closed_ahead = hit[0] in (2, 3) and dist[0] < 0.15
            if close and abs(err) < 25 and closed_ahead and self.use_cool <= 0 and not self.prev_use:
                b[7] = 1
                self.use_cool = 14
            if closed_ahead and dist[0] < 0.05:
                b[0] = 0  # don't grind into the door while it opens
        return self._finish(b)


def evaluate(episodes: int, stage: int, seed: int = 0):
    import sys
    import tempfile
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "training"))
    from doom_env import make_env
    from mapgen import spec_for_stage, write_level
    from vec_env import _hires_timer

    _hires_timer()
    rng = random.Random(seed)
    res = []
    for ep in range(episodes):
        spec = spec_for_stage(stage, rng)
        wad = str(Path(tempfile.gettempdir()) / "teacher_eval.wad")
        L = write_level(wad, spec, rng.randrange(1 << 30))
        env = make_env(wad, episode_ticks=4200, render_buffers=False)
        obs, _ = env.reset(seed=ep)
        T = Teacher(L, seed=ep)
        done = False
        while not done:
            a = T.act(obs, env.privileged_state())
            obs, r, te, tr, info = env.step(a)
            done = te or tr
        ps = env.privileged_state()
        res.append(dict(tags=spec.tags[0], has_exit=spec.exit, locks=spec.locks, exit=info["exit_reached"], dead=ps["dead"],
                        kills=ps["kills"], keys=int(np.asarray(ps["keys"]).sum()), tic=ps["tic"]))
        env.close()
        print(res[-1], flush=True)
    ex = [r for r in res if r["has_exit"]]
    print(f"stage {stage}: exit rate {np.mean([r['exit'] for r in ex]):.2f} (n={len(ex)}), "
          f"death {np.mean([r['dead'] for r in res]):.2f}, kills {np.mean([r['kills'] for r in res]):.2f}")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--stage", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    evaluate(a.episodes, a.stage, a.seed)
