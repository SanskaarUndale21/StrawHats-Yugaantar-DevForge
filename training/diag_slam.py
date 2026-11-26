"""Measure how wrong the agent's own map is while it really plays (training-time tool).

Runs the submission agent on random generated levels and compares its dead-reckoned pose with the
engine's ground truth every step, and its remembered walls with the real level layout.

    python training/diag_slam.py --episodes 24 --seed 1
"""

from __future__ import annotations

import argparse
import math
import multiprocessing as mp
import os
import random
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "submission"))
sys.path.insert(0, str(ROOT / "training"))


def run(job):
    seed, stage, weights = job
    import torch

    torch.set_num_threads(1)
    from agent import DoomAgent
    from doom_env import make_env
    from mapgen import CELL, spec_for_stage, write_level
    from spatial_memory import C_FREE, C_WALL, FINE, G_FINE
    from vec_env import _hires_timer

    _hires_timer()
    spec = spec_for_stage(stage, random.Random(seed))
    wad = str(Path(tempfile.gettempdir()) / f"slam_{os.getpid()}_{seed}.wad")
    L = write_level(wad, spec, seed)
    env = make_env(wad, episode_ticks=4200, render_buffers=True)
    obs, _ = env.reset(seed=seed)
    agent = DoomAgent(env.observation_space, env.action_space, "cpu")
    if weights:
        from common import load_checkpoint
        from model import load_compatible

        load_compatible(agent.net, load_checkpoint(weights)["model"])
    agent.reset()
    p0 = env.privileged_state()
    x0, y0, a0 = p0["x"], p0["y"], math.radians(p0["angle"])
    c0, s0 = math.cos(a0), math.sin(a0)
    drift, herr, done, t = [], [], False, 0
    while not done:
        a = agent.act(obs)
        obs, _, te, tr, _ = env.step(a)
        done = te or tr
        t += 1
        if done:
            break
        q = env.privileged_state()
        m = agent.memory
        # the pose the memory will hold after it integrates this step is only known next act();
        # compare the pose it used for the previous observation with the previous truth instead
        drift.append((q["x"], q["y"], q["angle"]))
        herr.append((m.x, m.y, m.th))
    tr_ = np.array(drift[:-1])
    me = np.array(herr[1:])
    if len(tr_) < 5:
        env.close()
        return None
    wx = x0 + c0 * me[:, 0] - s0 * me[:, 1]
    wy = y0 + s0 * me[:, 0] + c0 * me[:, 1]
    d = np.hypot(wx - tr_[:, 0], wy - tr_[:, 1])
    h = np.abs((np.degrees(me[:, 2]) + math.degrees(a0) - tr_[:, 2] + 180) % 360 - 180)
    # map check: remembered wall / free cells against the generated layout
    m = agent.memory
    ix, iy = np.nonzero(m.fine[C_WALL] > 0)
    fx, fy = np.nonzero((m.fine[C_FREE] > 0) & (m.fine[C_WALL] == 0))

    def truth(ix, iy):  # is the centre of this memory cell solid in the real level? (memory frame -> world)
        ix = np.where(ix > G_FINE // 2, ix - G_FINE, ix)
        iy = np.where(iy > G_FINE // 2, iy - G_FINE, iy)
        mx, my = (ix + 0.5) * FINE, (iy + 0.5) * FINE
        X, Y = x0 + c0 * mx - s0 * my, y0 + s0 * mx + c0 * my
        col, row = np.floor(X / CELL).astype(int), np.floor(-Y / CELL).astype(int)
        ok = (row >= 0) & (row < L.h) & (col >= 0) & (col < L.w)
        solid = np.ones(len(ix), bool)
        g = np.array([[ch == "#" for ch in r] for r in L.g])
        solid[ok] = g[row[ok], col[ok]]
        return solid

    wall_ok = truth(ix, iy).mean() if len(ix) else float("nan")
    free_ok = (~truth(fx, fy)).mean() if len(fx) else float("nan")
    inner = d[len(d) // 2:]
    env.close()
    return dict(seed=seed, stage=stage, steps=t, drift_end=d[-1], drift_max=d.max(), drift_mean=d.mean(), drift_p95=float(np.percentile(d, 95)),
                head_end=h[-1], head_max=h.max(), wall_ok=wall_ok, free_ok=free_ok)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=24)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--weights", default=None)
    ap.add_argument("--procs", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    a = ap.parse_args()
    jobs = [(a.seed * 1000 + i, 1 + i % 3, a.weights) for i in range(a.episodes)]
    with mp.get_context("spawn").Pool(a.procs) as pool:
        res = [r for r in pool.map(run, jobs, chunksize=1) if r]
    for r in sorted(res, key=lambda r: -r["drift_end"])[:8]:
        print("  worst: seed {seed} stage {stage} steps {steps}  drift end {drift_end:.0f} max {drift_max:.0f}  "
              "heading end {head_end:.1f} max {head_max:.1f} deg  walls right {wall_ok:.2f} free right {free_ok:.2f}".format(**r))
    M = lambda k: float(np.nanmean([r[k] for r in res]))  # noqa: E731
    P = lambda k, q: float(np.nanpercentile([r[k] for r in res], q))  # noqa: E731
    print(f"n={len(res)}  steps {M('steps'):.0f}  drift avg over game {M('drift_mean'):.0f}  p95 {M('drift_p95'):.0f}  end mean {M('drift_end'):.0f} p90 {P('drift_end', 90):.0f} units "
          f"(1 map cell = 64)  heading end mean {M('head_end'):.1f} p90 {P('head_end', 90):.1f} deg  "
          f"walls right {100 * M('wall_ok'):.0f}%  free right {100 * M('free_ok'):.0f}%")


if __name__ == "__main__":
    main()
