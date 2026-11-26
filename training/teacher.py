"""Privileged scripted teacher for DAgger distillation (TRAINING ONLY, never used at evaluation).

The teacher reads ground truth the agent never gets (true pose from `privileged_state()`, the level
layout produced by mapgen), but it only *plans over cells it has already seen* through the same 64 rays
the agent sees. Its behaviour is therefore reproducible from the agent's own observation history,
which is what makes it imitable (a god-view teacher that walks straight to an unseen keycard would
teach the student behaviour it cannot explain).

Priorities each step:
  1. combat  - nearest visible enemy within range: turn onto it and fire; back off from melee demons
  2. exit    - exit pad seen and reachable (locks open with held keys) -> path to it
  3. key     - a locked door blocks progress and a matching keycard has been seen -> fetch it
  4. explore - path to the nearest frontier (seen floor next to unseen space)
  doors are opened with a single USE press when the path runs into a closed one;
  a stuck detector injects short random escapes.

Outputs both MultiBinary buttons and the factorised head choices used by DoomNet.

    python training/teacher.py --episodes 20 --stage 2      # measure teacher success on generated maps
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
RAY_ANG = np.arange(64) * (2 * math.pi / 64)
SAMPLES = np.arange(16.0, 1024.0 + 1, 32.0)


def buttons_to_choice(b) -> np.ndarray:
    b = np.asarray(b)
    move = 0 if b[0] and not b[1] else (2 if b[1] and not b[0] else 1)
    strafe = 0 if b[2] and not b[3] else (2 if b[3] and not b[2] else 1)
    turn = 0 if b[4] and not b[5] else (2 if b[5] and not b[4] else 1)
    return np.array([move, strafe, turn, int(b[6]), int(b[7]), int(b[8])], np.int64)


class Teacher:
    def __init__(self, layout, seed: int = 0):
        self.L = layout
        self.h, self.w = layout.h, layout.w
        self.grid = layout.g
        self.keys = {}  # (r, c) -> colour letter
        for t, r, c, _ in layout.things:
            if t in KEY_THINGS:
                self.keys[(r, c)] = KEY_THINGS[t]
        self.health = {(r, c) for t, r, c, _ in layout.things if t in (2011, 2012, 2014)}
        self.exit = next(((r, c) for r in range(self.h) for c in range(self.w) if self.grid[r][c] == "E"), None)
        self.rng = random.Random(seed)
        self.reset()

    def reset(self):
        self.seen = np.zeros((self.h, self.w), bool)
        self.taken = set()
        self.path = []
        self.goal = None
        self.replan = 0
        self.hist = deque(maxlen=12)
        self.escape = 0
        self.escape_act = None
        self.use_cool = 0
        self.prev_use = 0
        self.strafe_dir = 2
        self.strafe_t = 0

    # ------------------------------------------------------------------ geometry helpers
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
        r0, c0 = self.cell_of(x, y)
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if self._in(r0 + dr, c0 + dc):
                    self.seen[r0 + dr, c0 + dc] = True

    def _has_key(self, vit, col):
        a, b = LOCK_KEYS[col]
        return vit[a] > 0 or vit[b] > 0

    def _passable(self, r, c, vit, known_only=True):
        ch = self.grid[r][c]
        if ch == "#":
            return False
        if known_only and not self.seen[r, c]:
            return False
        if ch in LOCK_KEYS:
            return self._has_key(vit, ch)
        return True

    def _bfs(self, start, vit, targets):
        """Shortest known path from start to any target cell. Returns list of cells (excl. start)."""
        prev = {start: None}
        q = deque([start])
        hit = None
        while q:
            cur = q.popleft()
            if cur in targets:
                hit = cur
                break
            r, c = cur
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                n = (r + dr, c + dc)
                if n not in prev and self._in(*n) and self._passable(*n, vit):
                    prev[n] = cur
                    q.append(n)
        if hit is None:
            return None
        path = []
        while hit != start:
            path.append(hit)
            hit = prev[hit]
        return path[::-1]

    def _frontier(self):
        out = set()
        for r in range(1, self.h - 1):
            for c in range(1, self.w - 1):
                if self.seen[r, c] and self.grid[r][c] != "#":
                    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        if not self.seen[r + dr, c + dc]:
                            out.add((r, c))
                            break
        return out

    def _plan(self, start, vit):
        if self.exit is not None and self.seen[self.exit]:
            p = self._bfs(start, vit, {self.exit})
            if p is not None:
                return p, "exit"
        # keys that open a seen locked door we cannot pass
        need = {ch for r in range(self.h) for c in range(self.w)
                if (ch := self.grid[r][c]) in LOCK_KEYS and self.seen[r, c] and not self._has_key(vit, ch)}
        want = {rc for rc, col in self.keys.items() if self.seen[rc] and rc not in self.taken
                and (col in need or not self._has_key(vit, col))}
        if want:
            p = self._bfs(start, vit, want)
            if p is not None:
                return p, "key"
        if vit[0] < 40:
            hp = {rc for rc in self.health if self.seen[rc] and rc not in self.taken}
            if hp:
                p = self._bfs(start, vit, hp)
                if p is not None and len(p) < 25:
                    return p, "health"
        fr = self._frontier() - {start}
        if fr:
            p = self._bfs(start, vit, fr)
            if p is not None:
                return p, "explore"
        return None, None

    def _line_clear(self, x0, y0, x1, y1):
        n = int(max(abs(x1 - x0), abs(y1 - y0)) / 16) + 1
        for k in range(n + 1):
            t = k / n
            r, c = self.cell_of(x0 + (x1 - x0) * t, y0 + (y1 - y0) * t)
            if not self._in(r, c) or self.grid[r][c] == "#":
                return False
        return True

    # ------------------------------------------------------------------ act
    def act(self, obs, ps) -> np.ndarray:
        x, y, th = ps["x"], ps["y"], math.radians(ps["angle"])
        vit = np.asarray(obs["vitals"])
        self._observe(obs, x, y, th)
        cur = self.cell_of(x, y)
        if cur in self.keys or cur in self.health:
            self.taken.add(cur)
        b = np.zeros(9, np.int8)

        # ---- stuck escape
        self.hist.append((x, y))
        if self.escape > 0:
            self.escape -= 1
            return self.escape_act.copy()
        if len(self.hist) == self.hist.maxlen:
            (x0, y0) = self.hist[0]
            if math.hypot(x - x0, y - y0) < 24 and self.goal is not None:
                self.escape = self.rng.randint(3, 7)
                e = np.zeros(9, np.int8)
                e[self.rng.choice([2, 3])] = 1
                e[self.rng.choice([4, 5])] = self.rng.random() < 0.6
                e[1] = self.rng.random() < 0.5
                self.escape_act = e
                self.hist.clear()
                self.path = []
                return e.copy()

        # ---- combat
        ents = obs["visible_entities"]
        mask = np.asarray(ents["mask"]).astype(bool)
        et = np.asarray(ents["entity_type"])
        f = np.asarray(ents["features"])
        enemies = np.flatnonzero(mask & np.isin(et, ENEMY_TYPES) & (f[:, 0] < 1.4))
        ammo = vit[3]
        if len(enemies) and (ammo > 0 or vit[2] <= 1):
            i = enemies[0]  # nearest first
            a = math.atan2(f[i, 1], f[i, 2])
            deg = math.degrees(a)
            if deg > 4:
                b[4] = 1
            elif deg < -4:
                b[5] = 1
            if abs(deg) < 7:
                b[6] = 1
            if et[i] == 1 and f[i, 0] < 0.2:  # melee enemy in our face: back off
                b[1] = 1
            # circle-strafe to dodge hitscan/projectiles, flipping direction every few steps
            self.strafe_t -= 1
            if self.strafe_t <= 0:
                self.strafe_dir = 2 if self.strafe_dir == 3 else 3
                self.strafe_t = self.rng.randint(4, 10)
            side = np.asarray(obs["rays"]["distance"])[16 if self.strafe_dir == 2 else 48]
            if side < 0.06:  # wall on that side: flip
                self.strafe_dir = 2 if self.strafe_dir == 3 else 3
            b[self.strafe_dir] = 1
            return b
        if ammo <= 0 and vit[4:8].sum() > 0 and self.rng.random() < 0.1:
            b[8] = 1

        # ---- navigation
        self.replan -= 1
        if not self.path or self.replan <= 0 or (self.path and self.path[0] == cur):
            self.path, self.goal = self._plan(cur, vit)
            self.path = self.path or []
            self.replan = 6
        while self.path and self.path[0] == cur:
            self.path.pop(0)
        if not self.path:
            b[0] = 1
            b[4] = self.rng.random() < 0.3
            return b
        # lookahead: farthest path cell (<= 4 ahead) with a clear straight line
        tgt = self.path[0]
        for k in range(min(4, len(self.path))):
            cx, cy = self.center(self.path[k])
            if self.grid[self.path[k][0]][self.path[k][1]] in "DBYR" and k > 0:
                break
            if self._line_clear(x, y, cx, cy):
                tgt = self.path[k]
        tx, ty = self.center(tgt)
        err = math.degrees(math.atan2(ty - y, tx - x) - th)
        err = (err + 180) % 360 - 180
        if err > 6:
            b[4] = 1
        elif err < -6:
            b[5] = 1
        if abs(err) < 40:
            b[0] = 1
        # doors: closed door right in front on our path -> one USE press, then wait
        hit = np.asarray(obs["rays"]["hit_type"])
        dist = np.asarray(obs["rays"]["distance"])
        nxt = self.path[0]
        door_next = self.grid[nxt[0]][nxt[1]] in "DBYR"
        self.use_cool -= 1
        if door_next and hit[0] in (2, 3) and dist[0] < 0.12 and self.use_cool <= 0 and not self.prev_use:
            b[7] = 1
            self.use_cool = 12
        self.prev_use = int(b[7])
        return b


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
