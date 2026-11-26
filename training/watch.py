"""Watch the agent play in a live window (game view + its spatial memory + stats).

    python training/watch.py                         # generated stage-2 level, current weights
    python training/watch.py --stage 4 --episodes 3
    python training/watch.py --tour                  # combat, explore, keys, gauntlet in turn
    python training/watch.py --wad maps/two_rooms.wad
    python training/watch.py --record run.mp4        # also save a video

Keys: q / Esc quits, space pauses, f toggles real-time pacing.
"""

from __future__ import annotations

import argparse
import random
import sys
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "submission"))
sys.path.insert(0, str(ROOT / "training"))

from agent import DoomAgent  # noqa: E402
from vec_env import _hires_timer  # noqa: E402

PANEL = 600


def _hires(width_res="RES_800X600"):
    """Render the *viewer* at high resolution. The agent never reads pixels (rays, entities, vitals
    only), so its behaviour is identical to evaluation; only the picture you see is sharper."""
    import types

    import vizdoom as vd
    import doom_env.wrapper as W

    res = types.SimpleNamespace(**{k: getattr(vd.ScreenResolution, k) for k in dir(vd.ScreenResolution) if k.startswith("RES_")})
    res.RES_160X120 = getattr(vd.ScreenResolution, width_res)
    proxy = types.SimpleNamespace(**{k: getattr(vd, k) for k in dir(vd) if not k.startswith("__")})
    proxy.ScreenResolution = res
    W.vd = proxy


def memory_view(mem, size=PANEL):
    """Render the agent's own ego map (fine scale): free=grey, wall=white, door=green, lock=red,
    key=yellow, visited=blue. Agent at centre, facing up."""
    from spatial_memory import C_DOOR, C_FREE, C_KEY, C_LOCK, C_VISIT, C_WALL, FINE, G_FINE

    R = 30
    u, v = np.meshgrid(np.arange(R, -R, -1), np.arange(R, -R, -1), indexing="ij")
    c, s = np.cos(mem.th), np.sin(mem.th)
    wx = mem.x + (c * u - s * v) * FINE
    wy = mem.y + (s * u + c * v) * FINE
    ix = np.floor(wx / FINE).astype(int) % G_FINE
    iy = np.floor(wy / FINE).astype(int) % G_FINE
    f = mem.fine[:, ix, iy]
    img = np.zeros((2 * R, 2 * R, 3), np.float32)
    img[f[C_FREE] > 0] = (60, 60, 60)
    img[f[C_VISIT] > 0] = (150, 80, 30)
    img[f[C_WALL] > 0] = (230, 230, 230)
    img[f[C_DOOR] > 0] = (40, 200, 40)
    img[f[C_LOCK] > 0] = (40, 40, 230)
    img[f[C_KEY] > 0] = (0, 220, 255)
    img = cv2.resize(img.astype(np.uint8), (size, size), interpolation=cv2.INTER_NEAREST)
    cv2.circle(img, (size // 2, size // 2), 5, (0, 165, 255), -1)
    cv2.line(img, (size // 2, size // 2), (size // 2, size // 2 - 14), (0, 165, 255), 2)
    return img


class SlamView:
    """Occupancy-grid SLAM display, the standard robotics picture: every cell holds the log-odds of
    being a wall, updated from each scan (cells a ray passed through become more free, the cell it
    ended in more occupied). Grey = never observed, white = free, black = occupied. On top: the
    current laser scan (red), the trajectory (blue) and the pose (arrow). Everything comes from the
    agent's own pose estimate and ray hits; nothing from the real level."""

    R, N = 8.0, 1024            # map units per cell, cells per side (8192 units around the start)
    L_FREE, L_OCC, L_MAX = -0.35, 0.9, 4.0
    SUB = 4                     # rays are 5.6 degrees apart; neighbours on one surface are interpolated

    def __init__(self, layout=None, start=None):
        self.lo = np.zeros((self.N, self.N), np.float32)   # log-odds, 0 = unknown
        self.path = []
        self.doors = []          # one entry per physical door: [x, y, hits, locked, opened]
        self.scan = np.zeros((0, 2))
        self.steps = 0

    def _px(self, x, y):
        return np.stack([self.N / 2 + np.asarray(x) / self.R, self.N / 2 - np.asarray(y) / self.R], -1)

    def add(self, mem, obs):
        d = np.asarray(obs["rays"]["distance"], np.float64) * 1024.0
        h = np.asarray(obs["rays"]["hit_type"])
        n = len(d)
        ang = mem.th + np.arange(n) * (2 * np.pi / n)
        ex, ey = mem.x + np.cos(ang) * d, mem.y + np.sin(ang) * d
        pts = np.rint(self._px(ex, ey)).astype(np.int32)
        x0, y0 = np.maximum(pts.min(0) - 2, 0)
        x1, y1 = np.minimum(pts.max(0) + 3, self.N)
        if x1 <= x0 or y1 <= y0:
            return
        loc = pts - (x0, y0)
        free = np.zeros((y1 - y0, x1 - x0), np.uint8)
        cv2.fillPoly(free, [loc], 1)                     # the area swept by this scan
        occ = np.zeros_like(free)
        static = (h >= 1) & (h <= 3) & (d < 1000)
        gap = np.hypot(np.roll(ex, -1) - ex, np.roll(ey, -1) - ey)
        same = gap < 0.22 * np.minimum(d, np.roll(d, -1)) + 14.0
        for i in np.flatnonzero(static):
            a = (int(loc[i][0]), int(loc[i][1]))
            k = (i + 1) % n
            if static[k] and same[i]:
                cv2.line(occ, a, (int(loc[k][0]), int(loc[k][1])), 1, 1)   # same surface: fill between hits
            else:
                occ[min(max(a[1], 0), occ.shape[0] - 1), min(max(a[0], 0), occ.shape[1] - 1)] = 1
        roi = self.lo[y0:y1, x0:x1]
        roi += self.L_FREE * ((free > 0) & (occ == 0))
        roi += self.L_OCC * (occ > 0)
        np.clip(roi, -self.L_MAX, self.L_MAX, out=roi)

        self.scan = np.stack([ex[h > 0], ey[h > 0]], 1)
        self.path.append((mem.x, mem.y))
        self.steps += 1
        for t in (2, 3):                                  # both faces of a door are one door
            sel = (h == t) & (d < 1000)
            for x, y in zip(ex[sel], ey[sel]):
                for dr in self.doors:
                    if np.hypot(dr[0] - x, dr[1] - y) < 110:
                        dr[0] += (x - dr[0]) / (dr[2] + 1)
                        dr[1] += (y - dr[1]) / (dr[2] + 1)
                        dr[2] = min(dr[2] + 1, 400)
                        dr[3] = dr[3] or t == 3
                        break
                else:
                    self.doors.append([float(x), float(y), 1, t == 3, False])
        for dr in self.doors:
            if np.hypot(dr[0] - mem.x, dr[1] - mem.y) < 44:
                dr[4] = True

    def render(self, mem, size=PANEL):
        known = np.abs(self.lo) > 0.2
        ys, xs = np.nonzero(known)
        if len(xs) == 0:
            return np.full((size, size, 3), 205, np.uint8)
        x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
        side = int(max(x1 - x0, y1 - y0, 150) + 24)
        cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
        ox, oy = int(np.clip(cx - side // 2, 0, self.N - side)), int(np.clip(cy - side // 2, 0, self.N - side))
        lo = self.lo[oy:oy + side, ox:ox + side]
        k = size / side
        # log-odds -> grey: unknown 205, free up to 254, occupied down to 0 (the usual map colours)
        t = np.clip(lo / 2.0, -1.0, 1.0)
        g = np.where(t < 0, 205 + (-t) * 49, 205 * (1 - t)).astype(np.uint8)
        img = cv2.cvtColor(cv2.resize(g, (size, size), interpolation=cv2.INTER_NEAREST), cv2.COLOR_GRAY2BGR)

        def out(x, y):
            return np.rint((self._px(x, y) - (ox, oy)) * k).astype(np.int32)

        step = 256.0                                       # metric grid, every 256 map units
        gx0 = (ox - self.N / 2) * self.R
        gy1 = (self.N / 2 - oy) * self.R
        for v in np.arange(np.ceil(gx0 / step) * step, gx0 + side * self.R, step):
            X = int((v - gx0) / self.R * k)
            cv2.line(img, (X, 0), (X, size), (188, 188, 188), 1)
        for v in np.arange(np.floor(gy1 / step) * step, gy1 - side * self.R, -step):
            Y = int((gy1 - v) / self.R * k)
            cv2.line(img, (0, Y), (size, Y), (188, 188, 188), 1)
        if len(self.path) > 1:
            p = np.array(self.path)
            cv2.polylines(img, [out(p[:, 0], p[:, 1])], False, (200, 90, 0), 1, cv2.LINE_AA)
        c = out(mem.x, mem.y)
        c = (int(c[0]), int(c[1]))
        if len(self.scan):                                 # the current scan: beams and hit points
            q = out(self.scan[:, 0], self.scan[:, 1])
            ov = img.copy()
            for a in q:
                cv2.line(ov, c, (int(a[0]), int(a[1])), (60, 60, 255), 1, cv2.LINE_AA)
            img = cv2.addWeighted(ov, 0.18, img, 0.82, 0)
            for a in q:
                cv2.circle(img, (int(a[0]), int(a[1])), 2, (0, 0, 230), -1, cv2.LINE_AA)
        for x, y, n, locked, opened in self.doors:
            if n < 3:
                continue
            a = out(x, y)
            col = (0, 0, 200) if locked else (0, 150, 0)
            cv2.rectangle(img, (int(a[0]) - 5, int(a[1]) - 5), (int(a[0]) + 5, int(a[1]) + 5), col, 1 if opened else -1)
        for kx, ky in mem.keys.values():
            a = out(kx, ky)
            cv2.drawMarker(img, (int(a[0]), int(a[1])), (0, 170, 230), cv2.MARKER_DIAMOND, 10, 2, cv2.LINE_AA)
        # pose: heading triangle
        th = mem.th
        tri = np.array([[14 * np.cos(th), -14 * np.sin(th)], [9 * np.cos(th + 2.5), -9 * np.sin(th + 2.5)],
                        [9 * np.cos(th - 2.5), -9 * np.sin(th - 2.5)]]) + c
        cv2.fillPoly(img, [np.rint(tri).astype(np.int32)], (0, 0, 255), cv2.LINE_AA)
        cv2.polylines(img, [np.rint(tri).astype(np.int32)], True, (0, 0, 0), 1, cv2.LINE_AA)
        # header, scale bar, legend
        cv2.rectangle(img, (0, 0), (size, 20), (40, 40, 40), -1)
        cells = int(known.sum())
        cv2.putText(img, f"occupancy grid  {self.R:.0f} u/cell  scans {self.steps}  known {cells} cells  "
                         f"pose ({mem.x:.0f}, {mem.y:.0f}, {np.degrees(mem.th) % 360:.0f} deg)", (6, 14),
                    cv2.FONT_HERSHEY_PLAIN, 0.9, (230, 230, 230), 1, cv2.LINE_AA)
        bar = int(256 / self.R * k)
        cv2.rectangle(img, (0, size - 20), (size, size), (40, 40, 40), -1)
        cv2.line(img, (8, size - 10), (8 + bar, size - 10), (255, 255, 255), 2)
        cv2.putText(img, "256 u", (14 + bar, size - 5), cv2.FONT_HERSHEY_PLAIN, 0.9, (230, 230, 230), 1, cv2.LINE_AA)
        x = 70 + bar
        for name, col in (("free", (254, 254, 254)), ("occupied", (0, 0, 0)), ("unknown", (205, 205, 205)),
                          ("scan", (0, 0, 230)), ("path", (200, 90, 0)), ("door", (0, 150, 0)), ("locked", (0, 0, 200))):
            cv2.rectangle(img, (x, size - 15), (x + 9, size - 6), col, -1)
            cv2.rectangle(img, (x, size - 15), (x + 9, size - 6), (120, 120, 120), 1)
            cv2.putText(img, name, (x + 13, size - 5), cv2.FONT_HERSHEY_PLAIN, 0.9, (230, 230, 230), 1, cv2.LINE_AA)
            x += 20 + cv2.getTextSize(name, cv2.FONT_HERSHEY_PLAIN, 0.9, 1)[0][0]
        return img


def _fit_screen(view):
    """Scale the window down if the screen is narrower than the picture."""
    try:
        import ctypes

        sw = ctypes.windll.user32.GetSystemMetrics(0)
    except Exception:
        return view
    if view.shape[1] > 0.96 * sw:
        f = 0.96 * sw / view.shape[1]
        view = cv2.resize(view, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
    return view


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wad", default=None)
    ap.add_argument("--map", default="MAP01")
    ap.add_argument("--stage", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--episodes", type=int, default=2)
    ap.add_argument("--ticks", type=int, default=4200)
    ap.add_argument("--record", default=None)
    ap.add_argument("--lowres", action="store_true", help="show the raw 160x120 evaluation frame")
    ap.add_argument("--tour", action="store_true", help="play benchmark tiers 1-4 one after another")
    ap.add_argument("--truth", action="store_true", help="debug: draw the real level faintly under the agent's map")
    ap.add_argument("--live", action="store_true", help="play fresh random levels forever, reloading --live-weights "
                    "whenever that file changes (training/aws/live_sync.sh keeps it fresh from the cloud box)")
    ap.add_argument("--live-weights", default=str(ROOT / "training" / "checkpoints" / "live_student.pt"))
    ap.add_argument("--weights", default=None, help="checkpoint to load (default: submission/weights/doomnet.pt)")
    a = ap.parse_args()
    _hires_timer()
    from doom_env import make_env

    levels = []  # (wad, map, tags, layout, ticks)
    if a.live:
        from evaluate import TIER_KIND, _make_level

        def endless():
            rng = random.Random()
            while True:
                tier = rng.choice((1, 2, 3, 3, 4, 4))
                wad, name, L, T = _make_level(("gen", tier, rng.randrange(1000)))
                yield wad, name, [f"tier {tier}", TIER_KIND[tier]], L, T

        levels = endless()
    elif a.tour:  # the benchmark tiers one after another, a different map each time
        from evaluate import TIER_KIND, _make_level

        for tier in (1, 2, 3, 4):
            wad, name, L, T = _make_level(("gen", tier, a.seed))
            levels.append((wad, name, [f"tier {tier}", TIER_KIND[tier]], L, T))
    elif a.wad is None:
        from mapgen import ascii_map, spec_for_stage, write_level

        for ep in range(a.episodes):
            spec = spec_for_stage(a.stage, random.Random(a.seed + ep))
            wad = str(Path(tempfile.gettempdir()) / f"watch_{a.seed + ep}.wad")
            L = write_level(wad, spec, a.seed + ep)
            levels.append((wad, a.map, list(spec.tags), L, a.ticks))
    else:
        levels = [(a.wad, a.map, ["custom"], None, a.ticks)] * a.episodes
    if not a.lowres:
        _hires()
    from metrics import EpisodeMetrics

    total = "" if a.live else f"/{len(levels)}"
    live_mtime, live_step = 0.0, "?"
    agent, writer, quit_ = None, None, False
    realtime, paused = True, False
    names = ["FWD", "BACK", "SL", "SR", "TL", "TR", "FIRE", "USE", "WPN"]
    for ep, (wad, mapname, tags, L, ticks) in enumerate(levels):
        env = make_env(wad, mapname, episode_ticks=ticks, render_buffers=True)
        # no status bar or engine crosshair: the agent reads this same picture to recognise the exit
        # pad, and the scorer renders without them. A crosshair is drawn on the displayed copy only.
        if agent is None:
            agent = DoomAgent(env.observation_space, env.action_space, "cpu")
            if a.weights:
                from common import load_checkpoint
                from model import load_compatible

                ck = load_checkpoint(a.weights)
                load_compatible(agent.net, ck["model"])
                print(f"loaded {a.weights} (step {ck.get('step', '?')})")
        if a.live and Path(a.live_weights).exists() and Path(a.live_weights).stat().st_mtime != live_mtime:
            from common import load_checkpoint
            from model import load_compatible

            try:
                ck = load_checkpoint(a.live_weights)
                load_compatible(agent.net, ck["model"])
                live_mtime, live_step = Path(a.live_weights).stat().st_mtime, ck.get("step", "?")
                print(f"new weights from training: step {live_step}", flush=True)
            except Exception as e:  # file mid-copy: keep the old weights for this game
                print("weights not readable yet:", e)
        H0, W0 = env._game.get_screen_height(), env._game.get_screen_width()
        scale = max(1, 800 // W0)
        W, H = W0 * scale, H0 * scale
        if a.record and writer is None:
            writer = cv2.VideoWriter(a.record, cv2.VideoWriter_fourcc(*"mp4v"), 35 / 4, (W + H, H))
        obs, info = env.reset(seed=a.seed + ep)
        agent.reset()
        met = EpisodeMetrics(env, L)
        ps0 = env.privileged_state()
        slam = SlamView(L if a.truth else None, (ps0["x"], ps0["y"], np.radians(ps0["angle"])))
        done, t, hold = False, 0, 0
        act = np.zeros(len(names), int)
        while not done or hold > 0:
            t0 = time.time()
            if done:
                hold -= 1
            elif not paused:
                act = agent.act(obs)
                slam.add(agent.memory, obs)
                obs, r, term, trunc, info = env.step(act)
                met.step(act)
                done = term or trunc
                t += 1
                if done:
                    hold = 25  # keep the final frame and stats on screen for ~3 s
            ps = env.privileged_state()
            m = met.result(ps["kills"])
            frame = cv2.resize(cv2.cvtColor(obs["rgb"], cv2.COLOR_RGB2BGR), (W, H), interpolation=cv2.INTER_NEAREST)
            panel = cv2.resize(slam.render(agent.memory), (H, H), interpolation=cv2.INTER_LINEAR)
            v = obs["vitals"]
            rooms = f"{m['rooms_seen']}/{m['rooms_total']}" if L is not None else f"{m['rooms_seen']}"
            aim = f"{100 * m['acc']:.0f}%" if m["shots"] else "-"
            lines = [f"{' '.join(tags)}  game {ep + 1}{total}  step {t}" + (f"  train step {live_step}" if a.live else ""),
                     f"health {v[0]:.0f}  armor {v[1]:.0f}  ammo {v[3]:.0f}",
                     f"kills {ps['kills']}/{m['monsters']}  aim {aim} ({m['shots']} shots)",
                     f"rooms {rooms}  cleared {m['rooms_cleared']}/{m['rooms_hostile']}",
                     f"keys {int(v[8:14].sum())}  gun swaps {m['wpn_switches']}  secrets {m['secrets']}",
                     "act: " + " ".join(n for n, b in zip(names, act) if b),
                     "EXIT REACHED" if info.get("exit_reached") else ("DEAD" if ps["dead"] else ("TIME UP" if done else ""))]
            cv2.drawMarker(frame, (W // 2, H // 2), (0, 255, 0), cv2.MARKER_CROSS, 16, 1, cv2.LINE_AA)
            ov = frame.copy()  # stats box on top of the game picture
            cv2.rectangle(ov, (0, 0), (430, 14 + 24 * len(lines)), (0, 0, 0), -1)
            frame = cv2.addWeighted(ov, 0.6, frame, 0.4, 0)
            for i, txt in enumerate(lines):
                cv2.putText(frame, txt, (10, 24 + 24 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
            view = np.concatenate([frame, panel], 1)
            cv2.imshow("StrawHats DoomAgent", _fit_screen(view))
            if writer is not None:
                writer.write(view)
            k = cv2.waitKey(max(1, int(1000 * (4 / 35 - (time.time() - t0)))) if realtime else 1) & 0xFF
            if k in (ord("q"), 27):
                quit_ = True
                break
            if k == ord(" "):
                paused = not paused
            if k == ord("f"):
                realtime = not realtime
        out = "exit" if info.get("exit_reached") else ("dead" if ps["dead"] else "time up")
        print(f"game {ep + 1} [{' '.join(tags)}]: {out}  steps {t}  kills {ps['kills']}/{m['monsters']}  aim {aim} of "
              f"{m['shots']} shots  rooms {rooms} cleared {m['rooms_cleared']}/{m['rooms_hostile']}  "
              f"gun swaps {m['wpn_switches']}  keys {int(obs['vitals'][8:14].sum())}", flush=True)
        env.close()
        if quit_:
            break
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
