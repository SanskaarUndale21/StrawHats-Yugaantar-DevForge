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
    """The map the agent has built so far, drawn north-up from nothing but its own pose estimate and
    its own ray hits. Explored floor is filled in, walls are joined into lines, doors are green,
    locked doors red, keys yellow, the path it walked blue. The real level is drawn faintly underneath
    (viewer only, the agent never sees it) so you can judge how right the map is."""

    R, N = 8.0, 1024  # map units per pixel, canvas side (covers 8192 units around the start)
    BG, FLOOR, WALL, DOOR, LOCK = (20, 18, 16), (74, 66, 58), (245, 245, 245), (60, 210, 60), (50, 50, 235)

    def __init__(self, layout=None, start=None):
        n = self.N
        self.free, self.wall, self.door, self.lock = (np.zeros((n, n), np.uint8) for _ in range(4))
        self.path = []
        self.truth = []
        if layout is not None and start is not None:  # outline of the real level, in the agent's frame
            x0, y0, a0 = start
            c0, s0 = np.cos(a0), np.sin(a0)
            g = layout.g
            for r in range(layout.h):
                for c in range(layout.w):
                    if g[r][c] == "#":
                        continue
                    X0, X1, Y0, Y1 = c * 64.0, (c + 1) * 64.0, -(r + 1) * 64.0, -r * 64.0
                    for (dr, dc), seg in (((-1, 0), (X0, Y1, X1, Y1)), ((1, 0), (X0, Y0, X1, Y0)),
                                          ((0, -1), (X0, Y0, X0, Y1)), ((0, 1), (X1, Y0, X1, Y1))):
                        rr, cc = r + dr, c + dc
                        if not (0 <= rr < layout.h and 0 <= cc < layout.w) or g[rr][cc] == "#":
                            w = np.array([[seg[0] - x0, seg[1] - y0], [seg[2] - x0, seg[3] - y0]])
                            self.truth.append(np.stack([c0 * w[:, 0] + s0 * w[:, 1], -s0 * w[:, 0] + c0 * w[:, 1]], 1))

    def _px(self, x, y):
        return np.stack([self.N / 2 + np.asarray(x) / self.R, self.N / 2 - np.asarray(y) / self.R], -1)

    def add(self, mem, obs):
        d = np.asarray(obs["rays"]["distance"], np.float64) * 1024.0
        h = np.asarray(obs["rays"]["hit_type"])
        ang = mem.th + np.arange(len(d)) * (2 * np.pi / len(d))
        ex, ey = mem.x + np.cos(ang) * d, mem.y + np.sin(ang) * d
        pts = np.rint(self._px(ex, ey)).astype(np.int32)
        cv2.fillPoly(self.free, [pts], 255)  # everything the rays swept this step is seen floor
        nxt = np.roll(pts, -1, 0)
        gap = np.hypot(np.roll(ex, -1) - ex, np.roll(ey, -1) - ey)
        same = gap < 0.5 * np.minimum(d, np.roll(d, -1)) + 24.0
        for t, layer, th in ((1, self.wall, 1), (2, self.door, 3), (3, self.lock, 3)):
            sel = (h == t) & (d < 1000)
            for i in np.flatnonzero(sel):
                a = (int(pts[i][0]), int(pts[i][1]))
                if sel[(i + 1) % len(d)] and same[i]:
                    cv2.line(layer, a, (int(nxt[i][0]), int(nxt[i][1])), 255, th)
                else:
                    cv2.circle(layer, a, th // 2, 255, -1)
        self.path.append((mem.x, mem.y))

    def render(self, mem, size=PANEL):
        ys, xs = np.nonzero(self.free)
        if len(xs) == 0:
            return np.full((size, size, 3), self.BG, np.uint8)
        x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
        side = int(max(x1 - x0, y1 - y0, 160) + 28)
        cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
        ox, oy = int(np.clip(cx - side // 2, 0, self.N - side)), int(np.clip(cy - side // 2, 0, self.N - side))
        sl = (slice(oy, oy + side), slice(ox, ox + side))
        k = size / side
        img = np.full((side, side, 3), self.BG, np.uint8)
        img[self.free[sl] > 0] = self.FLOOR
        img = cv2.resize(img, (size, size), interpolation=cv2.INTER_NEAREST)

        def out(x, y):
            return np.rint((self._px(x, y) - (ox, oy)) * k).astype(np.int32)

        for seg in self.truth:  # the real level, faint
            a, b = out(seg[:, 0], seg[:, 1])
            cv2.line(img, (int(a[0]), int(a[1])), (int(b[0]), int(b[1])), (120, 96, 70), 1, cv2.LINE_AA)
        if len(self.path) > 1:
            p = np.array(self.path)
            cv2.polylines(img, [out(p[:, 0], p[:, 1])], False, (230, 150, 60), 2, cv2.LINE_AA)
        for layer, col, grow in ((self.wall, self.WALL, 1), (self.door, self.DOOR, 2), (self.lock, self.LOCK, 2)):
            m = cv2.resize(layer[sl], (size, size), interpolation=cv2.INTER_LINEAR) > 110
            if grow > 1:
                m = cv2.dilate(m.astype(np.uint8), np.ones((grow, grow), np.uint8)) > 0
            img[m] = col
        for kx, ky in mem.keys.values():
            c = out(kx, ky)
            cv2.circle(img, (int(c[0]), int(c[1])), 6, (0, 220, 255), -1, cv2.LINE_AA)
            cv2.circle(img, (int(c[0]), int(c[1])), 6, (0, 0, 0), 1, cv2.LINE_AA)
        # the agent: a field-of-view wedge and an arrow
        c = out(mem.x, mem.y)
        fov = [out(mem.x + 260 * np.cos(mem.th + a), mem.y + 260 * np.sin(mem.th + a)) for a in np.linspace(-0.785, 0.785, 9)]
        ov = img.copy()
        cv2.fillPoly(ov, [np.array([c] + fov, np.int32)], (0, 200, 255))
        img = cv2.addWeighted(ov, 0.22, img, 0.78, 0)
        tip = out(mem.x + 18 / k * self.R * np.cos(mem.th), mem.y + 18 / k * self.R * np.sin(mem.th))
        cv2.arrowedLine(img, (int(c[0]), int(c[1])), (int(tip[0]), int(tip[1])), (0, 140, 255), 3, cv2.LINE_AA, tipLength=0.5)
        cv2.circle(img, (int(c[0]), int(c[1])), 5, (0, 140, 255), -1, cv2.LINE_AA)
        # title and legend
        cv2.rectangle(img, (0, 0), (size, 22), (0, 0, 0), -1)
        cv2.putText(img, "MAP BUILT BY THE AGENT (from its own position estimate)", (8, 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.rectangle(img, (0, size - 22), (size, size), (0, 0, 0), -1)
        x = 8
        for name, col in (("wall", self.WALL), ("door", self.DOOR), ("locked", self.LOCK), ("key", (0, 220, 255)),
                          ("path", (230, 150, 60)), ("seen floor", self.FLOOR), ("real level", (120, 96, 70))):
            cv2.rectangle(img, (x, size - 16), (x + 10, size - 6), col, -1)
            cv2.putText(img, name, (x + 14, size - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (220, 220, 220), 1, cv2.LINE_AA)
            x += 22 + cv2.getTextSize(name, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0][0]
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
        if not a.lowres:
            for f, v in (("set_render_hud", True), ("set_render_crosshair", True)):
                try:
                    getattr(env._game, f)(v)
                except Exception:
                    pass
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
        slam = SlamView(L, (ps0["x"], ps0["y"], np.radians(ps0["angle"])))
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
