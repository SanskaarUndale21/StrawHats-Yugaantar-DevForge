"""Ego-centric spatial memory: dead-reckoning odometry + a multi-scale allocentric occupancy map.

The observation has no global coordinates, so the agent keeps its own private frame:
  * heading is integrated from its own TURN actions (ViZDoom's keyboard turn profile: the first five
    held tics turn slowly, then full speed; measured in training/README.md),
  * position is integrated from the ego-frame velocity in `vitals` (trapezoid rule over the 4 tics),
  * while touching a wall or a monster the step is re-estimated by scan matching against the previous
    scan (the engine keeps reporting velocity while the player is stuck, so dead reckoning alone
    drifts by whole rooms).

Every step the 64 rays paint free space / walls / doors / locked doors into a toroidal grid at two
resolutions (64 and 256 map units per cell); keycards and pickups seen in `visible_entities` are
projected into the grid; the cells the agent stands on accumulate visit counts. The policy reads an
ego-centric, heading-aligned crop of both scales (a 16-channel image), per-ray "already visited"
fractions, and pointers to the nearest remembered locked door / closed door / keycard. Everything is
plain numpy and runs in well under a millisecond, identically in training workers and at evaluation.
"""

from __future__ import annotations

import math

import numpy as np

NUM_RAYS = 64
RAY_RANGE = 1024.0
SLOW_TURN, FAST_TURN, SLOW_TICS = 1.7578125, 3.515625, 5  # degrees per tic
FRAME_SKIP = 4
VEL_A, VEL_B = 1.5197, 2.8941  # exact for constant thrust under Doom friction 0.90625 (4 tics)
DEAD_TICS = 7  # the engine ignores input for the first 7 tics of an episode (measured)

# contact correction: while touching a wall or a monster the engine still reports velocity, so the
# step is re-estimated by sliding this scan onto the previous one (point-to-line fit)
CONTACT = 28.0   # map units; the player radius is 16
FIT_PRIOR = 1.0  # weight of the odometry guess; it decides the direction the walls leave free

FINE, COARSE = 64.0, 256.0
G_FINE, G_COARSE = 128, 32  # both span 8192 map units, wrapped (toroidal)
CROP = 21
HALF = CROP // 2

# map channels
C_VISIT, C_FREE, C_WALL, C_DOOR, C_LOCK, C_KEY, C_ITEM, C_FRONT = range(8)
N_MAP_CH = 8
RGB_H, RGB_W = 30, 40  # low-res colour view (exit pads and keycards are only recognisable by colour)

RAY_WALL, RAY_DOOR, RAY_LOCK, RAY_ENEMY, RAY_ITEM = 1, 2, 3, 4, 5
ENT_KEY = 9
ENT_ITEMS = (5, 6, 7, 8)  # health, armor, ammo, weapon

RAY_CH = 9
VEC_DIM = 49
AMMO_SCALE = np.array([200.0, 50.0, 50.0, 300.0], np.float32)

_RAY_ANG = np.arange(NUM_RAYS) * (2 * math.pi / NUM_RAYS)
_SAMPLES = np.arange(32.0, RAY_RANGE + 1, 32.0)  # free-space samples along a ray
_NOV_SAMPLES = np.linspace(0.15, 0.95, 8)          # novelty samples (fraction of hit distance)
_ii, _jj = np.meshgrid(np.arange(CROP), np.arange(CROP), indexing="ij")
_CROP_U = (HALF - _ii).astype(np.float64)  # forward (row 0 = far ahead)
_CROP_V = (HALF - _jj).astype(np.float64)  # left    (col 0 = far left)


class SpatialMemory:
    def __init__(self):
        self.reset()

    def reset(self) -> None:
        self.fine = np.zeros((N_MAP_CH, G_FINE, G_FINE), np.float32)
        self.coarse = np.zeros((N_MAP_CH, G_COARSE, G_COARSE), np.float32)
        self.x = self.y = 0.0
        self.th = 0.0  # radians, CCW
        self.turnheld = 0
        self.tic = 0
        self.pscan = None
        self.ppos = (0.0, 0.0)
        self.pcontact = False
        self.prev_vel = np.zeros(2)
        self.prev_action = np.zeros(9, np.float32)
        self.step = 0
        self.since_new = 0
        self.n_cells = 0
        self.locks: dict[tuple[int, int], tuple[float, float]] = {}
        self.doors: dict[tuple[int, int], tuple[float, float]] = {}
        self.keys: dict[tuple[int, int], tuple[float, float]] = {}

    # ------------------------------------------------------------------ odometry
    def _integrate(self, action: np.ndarray, vel_fr: np.ndarray) -> None:
        left, right = bool(action[4]), bool(action[5])
        d = float(left) - float(right)
        dth = 0.0
        for _ in range(FRAME_SKIP):
            self.tic += 1
            if left or right:
                self.turnheld += 1  # the held counter runs even while input is dead
                if self.tic > DEAD_TICS:
                    dth += d * (SLOW_TURN if self.turnheld <= SLOW_TICS else FAST_TURN)
            else:
                self.turnheld = 0
        th0 = self.th
        self.th = (self.th + math.radians(dth)) % (2 * math.pi)
        # ego velocity (forward, right) -> world, at the heading where it was measured
        w0 = _rot(self.prev_vel[0], -self.prev_vel[1], th0)
        w1 = _rot(vel_fr[0], -vel_fr[1], self.th)
        # per-step displacement ~ a*v_prev + b*v_now (least-squares fit to engine ground truth)
        self.x += VEL_A * w0[0] + VEL_B * w1[0]
        self.y += VEL_A * w0[1] + VEL_B * w1[1]
        self.prev_vel = np.array(vel_fr, np.float64)

    # ------------------------------------------------------------------ scan matching
    def _refit(self, d_odo: np.ndarray, q: np.ndarray) -> np.ndarray:
        """Displacement since the previous scan. q: this step's wall hits relative to the agent
        (world-aligned axes); the previous scan is kept as wall segments around the previous pose.
        Point-to-line least squares: walls fix the motion across them, the odometry guess keeps the
        motion along them (so a plain corridor cannot fake "I did not move")."""
        A, AB, N, L2 = self.pscan
        if len(A) < 4 or len(q) < 8:
            return d_odo
        d = d_odo.copy()
        for tol in (40.0, 20.0, 12.0):
            pts = q + d
            rel = pts[:, None, :] - A[None]
            t = np.clip((rel * AB[None]).sum(2) / L2[None], 0.0, 1.0)
            diff = rel - t[..., None] * AB[None]
            dist = np.hypot(diff[..., 0], diff[..., 1])
            j = np.argmin(dist, 1)
            ok = dist[np.arange(len(q)), j] < tol
            if ok.sum() < 8:
                return d_odo
            n = N[j[ok]]
            r = (n * (A[j[ok]] - q[ok])).sum(1)  # want n . (d + q - A) = 0
            H = n.T @ n + FIT_PRIOR * np.eye(2)
            d = np.linalg.solve(H, n.T @ r + FIT_PRIOR * d_odo)
        if np.hypot(*(d - d_odo)) > 1.5 * np.hypot(*d_odo) + 12.0:
            return d_odo
        return d

    @staticmethod
    def _segments(ex, ey, static, dunits):
        """Join neighbouring wall hits that lie on the same wall (no depth jump) into segments."""
        nx, ny = np.roll(ex, -1), np.roll(ey, -1)
        gap = np.hypot(nx - ex, ny - ey)
        pair = static & np.roll(static, -1) & (gap > 1e-3) & (gap < 0.5 * np.minimum(dunits, np.roll(dunits, -1)) + 24.0)
        A = np.stack([ex[pair], ey[pair]], 1)
        AB = np.stack([(nx - ex)[pair], (ny - ey)[pair]], 1)
        L2 = (AB**2).sum(1)
        N = np.stack([-AB[:, 1], AB[:, 0]], 1) / np.sqrt(L2)[:, None] if len(A) else AB
        return A, AB, N, L2

    # ------------------------------------------------------------------ map writes
    def _paint(self, ch: int, wx: np.ndarray, wy: np.ndarray, value: float = 1.0) -> None:
        fx = np.floor(wx / FINE).astype(np.int64) % G_FINE
        fy = np.floor(wy / FINE).astype(np.int64) % G_FINE
        self.fine[ch, fx, fy] = value
        cx = np.floor(wx / COARSE).astype(np.int64) % G_COARSE
        cy = np.floor(wy / COARSE).astype(np.int64) % G_COARSE
        self.coarse[ch, cx, cy] = np.maximum(self.coarse[ch, cx, cy], value)

    def _cell(self, wx: float, wy: float) -> tuple[int, int]:
        return int(math.floor(wx / FINE)) % G_FINE, int(math.floor(wy / FINE)) % G_FINE

    # ------------------------------------------------------------------ main entry
    def update(self, obs: dict, action: np.ndarray | None) -> dict:
        """Integrate one observation (after `action` was executed). Returns policy features."""
        vit = np.asarray(obs["vitals"], np.float32)
        if action is not None and self.step > 0:
            self._integrate(np.asarray(action), vit[14:16].astype(np.float64))
        else:
            self.prev_vel = vit[14:16].astype(np.float64)
        self.step += 1

        dist = np.asarray(obs["rays"]["distance"], np.float32)
        hit = np.asarray(obs["rays"]["hit_type"], np.int64)
        ang = self.th + _RAY_ANG
        ca, sa = np.cos(ang), np.sin(ang)
        dunits = dist.astype(np.float64) * RAY_RANGE

        static = ((hit == RAY_WALL) | (hit == RAY_DOOR) | (hit == RAY_LOCK)) & (dist < 0.97)
        ex, ey = ca * dunits, sa * dunits
        contact = bool((dunits[(hit >= RAY_WALL) & (hit <= RAY_ENEMY)] < CONTACT).any())
        if self.pscan is not None and (contact or self.pcontact):
            d = self._refit(np.array([self.x - self.ppos[0], self.y - self.ppos[1]]), np.stack([ex[static], ey[static]], 1))
            self.x, self.y = self.ppos[0] + d[0], self.ppos[1] + d[1]
        self.pscan = self._segments(ex, ey, static, dunits)
        self.ppos, self.pcontact = (self.x, self.y), contact

        # novelty along each ray, measured before this step's visit is recorded
        s = dunits[:, None] * _NOV_SAMPLES[None, :]
        nx = np.floor((self.x + ca[:, None] * s) / FINE).astype(np.int64) % G_FINE
        ny = np.floor((self.y + sa[:, None] * s) / FINE).astype(np.int64) % G_FINE
        visited_frac = (self.fine[C_VISIT, nx, ny] > 0).mean(axis=1).astype(np.float32)

        # free space along rays (stop a little short of the hit)
        m = _SAMPLES[None, :] < (dunits[:, None] - 24.0)
        sx = (self.x + ca[:, None] * _SAMPLES[None, :])[m]
        sy = (self.y + sa[:, None] * _SAMPLES[None, :])[m]
        self._paint(C_FREE, sx, sy)
        # a ray passing through a remembered door means it is open now: forget it (re-added below if still shut)
        fx = np.floor(sx / FINE).astype(np.int64) % G_FINE
        fy = np.floor(sy / FINE).astype(np.int64) % G_FINE
        was_door = (self.fine[C_DOOR, fx, fy] > 0) | (self.fine[C_LOCK, fx, fy] > 0)
        if was_door.any():
            self.fine[C_DOOR, fx[was_door], fy[was_door]] = 0.0
            self.fine[C_LOCK, fx[was_door], fy[was_door]] = 0.0
            for c in set(zip(fx[was_door].tolist(), fy[was_door].tolist())):
                self.doors.pop(c, None)
                self.locks.pop(c, None)
        # obstacles at the hit point
        hx, hy = self.x + ca * (dunits + 8.0), self.y + sa * (dunits + 8.0)
        for t, ch in ((RAY_WALL, C_WALL), (RAY_DOOR, C_DOOR), (RAY_LOCK, C_LOCK)):
            sel = hit == t
            if sel.any():
                self._paint(ch, hx[sel], hy[sel])
                if t in (RAY_DOOR, RAY_LOCK):
                    store = self.locks if t == RAY_LOCK else self.doors
                    for px, py in zip(hx[sel], hy[sel]):
                        store[self._cell(px, py)] = (px, py)

        # entities: keys and pickups projected into the map
        ents = obs["visible_entities"]
        mask = np.asarray(ents["mask"]).astype(bool)
        feats = np.asarray(ents["features"], np.float32)
        etype = np.asarray(ents["entity_type"], np.int64)
        if mask.any():
            f = feats[mask]
            t = etype[mask]
            d = f[:, 0].astype(np.float64) * RAY_RANGE
            lx, ly = d * f[:, 2], d * f[:, 1]  # forward, left
            c, s_ = math.cos(self.th), math.sin(self.th)
            wx, wy = self.x + c * lx - s_ * ly, self.y + s_ * lx + c * ly
            k = t == ENT_KEY
            if k.any():
                self._paint(C_KEY, wx[k], wy[k])
                for px, py in zip(wx[k], wy[k]):
                    self.keys[self._cell(px, py)] = (px, py)
            it = np.isin(t, ENT_ITEMS)
            if it.any():
                self._paint(C_ITEM, wx[it], wy[it])

        # visit bookkeeping (also clears pickups/keys we walk over)
        cx, cy = self._cell(self.x, self.y)
        if self.fine[C_VISIT, cx, cy] == 0:
            self.n_cells += 1
            self.since_new = 0
        else:
            self.since_new += 1
        self.fine[C_VISIT, cx, cy] += 1.0
        self.fine[C_ITEM, cx, cy] = 0.0
        self.fine[C_KEY, cx, cy] = 0.0
        self.keys.pop((cx, cy), None)
        gx, gy = int(math.floor(self.x / COARSE)) % G_COARSE, int(math.floor(self.y / COARSE)) % G_COARSE
        self.coarse[C_VISIT, gx, gy] += 1.0
        if any(vit[8:14] > 0):
            self.keys.clear()  # holding a key: stop chasing remembered keycards

        self._frontier(self.fine)
        self._frontier(self.coarse)
        feats_out = self._features(obs, vit, dist, hit, visited_frac, feats, etype, mask)
        rgb = obs.get("rgb")
        if rgb is None:
            feats_out["rgb"] = np.zeros((3, RGB_H, RGB_W), np.uint8)
        else:  # area-average down to (3, RGB_H, RGB_W) uint8, from any input resolution
            arr = np.asarray(rgb, np.float32)
            H, W = arr.shape[:2]
            fh, fw = max(H // RGB_H, 1), max(W // RGB_W, 1)
            arr = arr[: fh * RGB_H, : fw * RGB_W]
            small = arr.reshape(RGB_H, fh, RGB_W, fw, 3).mean((1, 3))
            feats_out["rgb"] = small.transpose(2, 0, 1).astype(np.uint8)
        if action is not None:
            self.prev_action = np.asarray(action, np.float32)
        return feats_out

    @staticmethod
    def _frontier(g: np.ndarray) -> None:
        """Frontier channel: known free cells that touch space never observed (plain map filtering,
        no path planning). Tells the policy where unexplored space begins."""
        known = (g[C_FREE] > 0) | (g[C_WALL] > 0) | (g[C_DOOR] > 0) | (g[C_LOCK] > 0)
        unk = ~known
        near = np.roll(unk, 1, 0) | np.roll(unk, -1, 0) | np.roll(unk, 1, 1) | np.roll(unk, -1, 1)
        g[C_FRONT] = ((g[C_FREE] > 0) & (g[C_WALL] == 0) & near).astype(np.float32)

    # ------------------------------------------------------------------ features
    def _crop(self, grid: np.ndarray, cell: float, g: int) -> np.ndarray:
        c, s = math.cos(self.th), math.sin(self.th)
        wx = self.x + (c * _CROP_U - s * _CROP_V) * cell
        wy = self.y + (s * _CROP_U + c * _CROP_V) * cell
        ix = np.floor(wx / cell).astype(np.int64) % g
        iy = np.floor(wy / cell).astype(np.int64) % g
        return grid[:, ix, iy]

    def _pointer(self, store: dict) -> list[float]:
        if not store:
            return [0.0, 0.0, 1.0, 0.0]
        pts = np.array(list(store.values()))
        dx, dy = pts[:, 0] - self.x, pts[:, 1] - self.y
        d = np.hypot(dx, dy)
        i = int(np.argmin(d))
        # express the remembered point in the current ego frame (unit vector + distance);
        # this is a memory read-out only, the policy decides what to do with it
        c, s = math.cos(self.th), math.sin(self.th)
        fwd, left = (c * dx[i] + s * dy[i]) / max(d[i], 1e-6), (-s * dx[i] + c * dy[i]) / max(d[i], 1e-6)
        return [left, fwd, min(d[i] / 2048.0, 1.0), 1.0]

    def _features(self, obs, vit, dist, hit, visited_frac, feats, etype, mask) -> dict:
        rays = np.zeros((RAY_CH, NUM_RAYS), np.float32)
        rays[0] = dist
        rays[1] = np.exp(-dist * 8.0)
        rays[2 + hit, np.arange(NUM_RAYS)] = 1.0
        rays[8] = visited_frac

        fine = self._crop(self.fine, FINE, G_FINE)
        coarse = self._crop(self.coarse, COARSE, G_COARSE)
        fine[C_VISIT] = np.minimum(np.log1p(fine[C_VISIT]) / 3.0, 1.0)
        coarse[C_VISIT] = np.minimum(np.log1p(coarse[C_VISIT]) / 4.0, 1.0)
        img = np.concatenate([fine, coarse], 0).astype(np.float32)

        weapon = np.zeros(8, np.float32)
        weapon[int(np.clip(vit[2], 0, 7))] = 1.0
        vec = np.concatenate([
            [vit[0] / 100.0, vit[1] / 100.0, min(vit[3], 200.0) / 50.0],
            weapon,
            np.minimum(vit[4:8] / AMMO_SCALE, 2.0),
            vit[8:14],
            np.clip(vit[14:17] / 16.0, -2, 2),
            self.prev_action,
            self._pointer(self.locks), self._pointer(self.doors), self._pointer(self.keys),
            [min(self.step / 1050.0, 1.0), min(self.since_new / 100.0, 1.0), min(self.n_cells / 300.0, 1.0),
             float(any(vit[8:14] > 0))],
        ]).astype(np.float32)
        assert vec.shape[0] == VEC_DIM, vec.shape

        ef = np.zeros((feats.shape[0], 5), np.float32)
        ef[:, 0] = np.minimum(feats[:, 0], 4.0)
        ef[:, 1:3] = feats[:, 1:3]
        ef[:, 3:5] = np.clip(feats[:, 3:5] / 16.0, -2, 2)
        ef *= mask[:, None]
        return {"rays": rays, "map": img, "vec": vec, "ent": ef, "ent_type": (etype * mask).astype(np.int64),
                "ent_mask": mask.astype(np.float32)}

    def pose(self) -> tuple[float, float, float]:
        return self.x, self.y, self.th


def _rot(fx: float, ly: float, th: float) -> tuple[float, float]:
    c, s = math.cos(th), math.sin(th)
    return c * fx - s * ly, s * fx + c * ly
