"""Pre-train the exit detector (ExitEye + exit_head) on labelled frames, then hand it to the RL run.

Inside RL the detector sees a positive on about 1% of frames and gets a few gradient steps per
minute, so it stayed at 0% recall for a million steps. Here it is plain supervised learning: the
scripted teacher (with some random detours) plays generated levels, every frame is labelled from
ground truth (pad in view, bearing, range), and the small CNN is fitted to that.

    python training/pretrain_exit.py --episodes 160 --out training/checkpoints/exit_eye.pt
"""

from __future__ import annotations

import argparse
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


def gen(seed):
    """One episode -> (frames uint8 (n,3,60,80), targets float32 (n,3)); most negatives are dropped."""
    from doom_env import make_env
    from mapgen import spec_for_stage, write_level
    from rewards import RewardShaper, priv
    from teacher import Teacher
    from vec_env import _hires_timer

    _hires_timer()
    rng = random.Random(seed)
    for _ in range(100):
        spec = spec_for_stage(rng.randint(1, 4), rng)
        if spec.exit:
            break
    wad = str(Path(tempfile.gettempdir()) / f"exitpre_{os.getpid()}.wad")
    L = write_level(wad, spec, seed)
    env = make_env(wad, episode_ticks=2800, render_buffers=True)
    obs, _ = env.reset(seed=seed)
    sh = RewardShaper()
    sh.reset(priv(env), L)
    t = Teacher(L, seed=seed)
    X, Y, detour, done = [], [], 0, False
    while not done:
        a = t.act(obs, priv(env))
        if detour > 0 or rng.random() < 0.03:  # wander off the teacher's line: other views of the pad
            detour = detour - 1 if detour > 0 else rng.randint(4, 14)
            a = np.zeros(9, np.int8)
            a[rng.choice([0, 0, 2, 3])] = 1
            a[rng.choice([4, 5])] = rng.random() < 0.7
        obs, _, te, tr, _ = env.step(a)
        done = te or tr
        if done:
            break
        sh._ps = priv(env)
        y = sh.exit_targets(obs)
        if y[0] > 0 or rng.random() < 0.2:
            rgb = np.asarray(obs["rgb"], np.float32)
            X.append(rgb.reshape(60, 2, 80, 2, 3).mean((1, 3)).transpose(2, 0, 1).astype(np.uint8))
            Y.append(y)
    env.close()
    if not X:
        return np.zeros((0, 3, 60, 80), np.uint8), np.zeros((0, 3), np.float32)
    return np.stack(X), np.stack(Y)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=160)
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--procs", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    ap.add_argument("--seed", type=int, default=500_000)
    ap.add_argument("--out", default=str(ROOT / "training" / "checkpoints" / "exit_eye.pt"))
    a = ap.parse_args()

    import torch
    import torch.nn.functional as F
    from model import DoomNet

    with mp.get_context("spawn").Pool(a.procs) as pool:
        data = pool.map(gen, [a.seed + i for i in range(a.episodes)], chunksize=1)
    n_val = max(1, a.episodes // 6)  # whole episodes held out: the validation levels are unseen
    Xt, Yt = np.concatenate([d[0] for d in data[n_val:]]), np.concatenate([d[1] for d in data[n_val:]])
    Xv, Yv = np.concatenate([d[0] for d in data[:n_val]]), np.concatenate([d[1] for d in data[:n_val]])
    print(f"train {len(Xt)} frames ({Yt[:, 0].mean():.1%} with a pad in view), validation {len(Xv)} ({Yv[:, 0].mean():.1%})", flush=True)

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = DoomNet()
    eye, head = net.eye.to(dev), net.exit_head.to(dev)
    opt = torch.optim.Adam(list(eye.parameters()) + list(head.parameters()), lr=1e-3)
    Xt_t, Yt_t = torch.as_tensor(Xt), torch.as_tensor(Yt)
    pw = torch.tensor(float((1 - Yt[:, 0].mean()) / max(Yt[:, 0].mean(), 1e-3)), device=dev).clamp(max=20.0)

    def evaluate():
        eye.eval()
        with torch.no_grad():
            out = torch.cat([head(eye(torch.as_tensor(Xv[i:i + 1024]).to(dev))).cpu() for i in range(0, len(Xv), 1024)])
        eye.train()
        p = torch.sigmoid(out[:, 0]).numpy() > 0.6
        y = Yv[:, 0] > 0.5
        tp = (p & y).sum()
        rec, prec = tp / max(y.sum(), 1), tp / max(p.sum(), 1)
        hit = p & y
        bear = np.abs(out[:, 1].numpy() - Yv[:, 1])[hit].mean() * 45 if hit.any() else float("nan")
        rng_ = np.abs(out[:, 2].numpy() - Yv[:, 2])[hit].mean() * 1024 if hit.any() else float("nan")
        return rec, prec, bear, rng_

    for ep in range(a.epochs):
        perm = torch.randperm(len(Xt_t))
        for i in range(0, len(perm), 256):
            idx = perm[i:i + 256]
            x, y = Xt_t[idx].to(dev), Yt_t[idx].to(dev)
            if torch.rand(1).item() < 0.5:  # mirror the picture: the bearing flips sign
                x, y = x.flip(-1), y * torch.tensor([1.0, -1.0, 1.0], device=dev)
            out = head(eye(x))
            seen = y[:, 0]
            loss = F.binary_cross_entropy_with_logits(out[:, 0], seen, pos_weight=pw) + \
                3.0 * (seen * ((out[:, 1] - y[:, 1]) ** 2 + (out[:, 2] - y[:, 2]) ** 2)).sum() / seen.sum().clamp(min=1.0)
            opt.zero_grad()
            loss.backward()
            opt.step()
        if ep % 5 == 4 or ep == a.epochs - 1:
            rec, prec, bear, rng_ = evaluate()
            print(f"epoch {ep + 1}: on unseen levels  recall {rec:.0%}  precision {prec:.0%}  bearing error {bear:.1f} deg  "
                  f"range error {rng_:.0f} units", flush=True)
    torch.save({"eye": {k: v.cpu() for k, v in eye.state_dict().items()},
                "exit_head": {k: v.cpu() for k, v in head.state_dict().items()}}, a.out)
    print("saved", a.out)


if __name__ == "__main__":
    main()
