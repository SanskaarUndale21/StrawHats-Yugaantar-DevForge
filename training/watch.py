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

PANEL = 300


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
    ap.add_argument("--weights", default=None, help="checkpoint to load (default: submission/weights/doomnet.pt)")
    a = ap.parse_args()
    _hires_timer()
    from doom_env import make_env

    levels = []  # (wad, map, tags, layout, ticks)
    if a.tour:  # the benchmark tiers one after another, a different map each time
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
        H0, W0 = env._game.get_screen_height(), env._game.get_screen_width()
        scale = max(1, 800 // W0)
        W, H = W0 * scale, H0 * scale
        if a.record and writer is None:
            writer = cv2.VideoWriter(a.record, cv2.VideoWriter_fourcc(*"mp4v"), 35 / 4, (W + PANEL, H))
        obs, info = env.reset(seed=a.seed + ep)
        agent.reset()
        met = EpisodeMetrics(env, L)
        done, t, hold = False, 0, 0
        act = np.zeros(len(names), int)
        while not done or hold > 0:
            t0 = time.time()
            if done:
                hold -= 1
            elif not paused:
                act = agent.act(obs)
                obs, r, term, trunc, info = env.step(act)
                met.step(act)
                done = term or trunc
                t += 1
                if done:
                    hold = 25  # keep the final frame and stats on screen for ~3 s
            ps = env.privileged_state()
            m = met.result(ps["kills"])
            frame = cv2.resize(cv2.cvtColor(obs["rgb"], cv2.COLOR_RGB2BGR), (W, H), interpolation=cv2.INTER_NEAREST)
            panel = np.zeros((H, PANEL, 3), np.uint8)
            panel[:PANEL] = memory_view(agent.memory)
            v = obs["vitals"]
            rooms = f"{m['rooms_seen']}/{m['rooms_total']}" if L is not None else f"{m['rooms_seen']}"
            aim = f"{100 * m['acc']:.0f}%" if m["shots"] else "-"
            lines = [f"{' '.join(tags)}  game {ep + 1}/{len(levels)}  step {t}",
                     f"health {v[0]:.0f}  armor {v[1]:.0f}  ammo {v[3]:.0f}",
                     f"kills {ps['kills']}/{m['monsters']}  aim {aim} ({m['shots']} shots)",
                     f"rooms {rooms}  cleared {m['rooms_cleared']}/{m['rooms_hostile']}",
                     f"keys {int(v[8:14].sum())}  gun swaps {m['wpn_switches']}  secrets {m['secrets']}",
                     "act: " + " ".join(n for n, b in zip(names, act) if b),
                     "EXIT REACHED" if info.get("exit_reached") else ("DEAD" if ps["dead"] else ("TIME UP" if done else ""))]
            for i, s in enumerate(lines):
                cv2.putText(panel, s, (8, PANEL + 26 + 26 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
            view = np.concatenate([frame, panel], 1)
            cv2.imshow("StrawHats DoomAgent", view)
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
