"""Calibrate SpatialMemory odometry against engine ground truth (training-time tool).

Fits per-step displacement = a * v_prev + b * v_now (world frame) by least squares, and checks the
heading model (keyboard turn: 5 slow tics at 1.7578 deg, then 3.5156 deg per tic) and pose drift.

    python training/calibrate_odometry.py --wad maps/two_rooms.wad
"""

import argparse
import math
import random
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "submission"))
sys.path.insert(0, str(ROOT / "training"))
from spatial_memory import SpatialMemory  # noqa: E402
from vec_env import _hires_timer  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wad", default=None, help="map to use (default: a generated stage-2 level)")
    ap.add_argument("--episodes", type=int, default=3)
    a = ap.parse_args()
    _hires_timer()
    from doom_env import make_env

    wad = a.wad
    if wad is None:
        from mapgen import spec_for_stage, write_level
        import tempfile

        wad = str(Path(tempfile.gettempdir()) / "calib.wad")
        write_level(wad, spec_for_stage(2, random.Random(3)), 3)
    env = make_env(wad, render_buffers=False)
    rng = np.random.default_rng(0)
    A, B, drift, herr = [], [], [], []
    for ep in range(a.episodes):
        o, _ = env.reset(seed=ep)
        m = SpatialMemory()
        m.update(o, None)
        p0 = env.privileged_state()
        p, pv = p0, o["vitals"][14:16].copy()
        x0, y0, a0 = p0["x"], p0["y"], math.radians(p0["angle"])
        act = np.zeros(9, np.int8)
        for t in range(500):
            if t % 5 == 0:
                act = np.zeros(9, np.int8)
                act[0] = rng.random() < 0.7
                act[2 + rng.integers(2)] = rng.random() < 0.3
                tr = rng.integers(3)
                act[4], act[5] = tr == 1, tr == 2
            o, _, te, trn, _ = env.step(act)
            m.update(o, act)
            q, v = env.privileged_state(), o["vitals"][14:16]
            h0, h1 = math.radians(p["angle"]), math.radians(q["angle"])
            w0 = (math.cos(h0) * pv[0] + math.sin(h0) * pv[1], math.sin(h0) * pv[0] - math.cos(h0) * pv[1])
            w1 = (math.cos(h1) * v[0] + math.sin(h1) * v[1], math.sin(h1) * v[0] - math.cos(h1) * v[1])
            for k, key in enumerate("xy"):
                A.append([w0[k], w1[k]])
                B.append(q[key] - p[key])
            dx, dy = q["x"] - x0, q["y"] - y0
            c, s = math.cos(-a0), math.sin(-a0)
            drift.append(math.hypot(c * dx - s * dy - m.x, s * dx + c * dy - m.y))
            herr.append(abs((math.degrees(m.th) - (q["angle"] - math.degrees(a0)) + 180) % 360 - 180))
            p, pv = q, v.copy()
            if te or trn:
                break
    coef = np.linalg.lstsq(np.array(A), np.array(B), rcond=None)[0]
    print(f"fitted VEL_A, VEL_B = {coef[0]:.3f}, {coef[1]:.3f}")
    print(f"pose drift: mean {np.mean(drift):.1f} max {np.max(drift):.1f} map units; heading err max {max(herr):.3f} deg")


if __name__ == "__main__":
    main()
