"""Pre-train the picture reader (ExitEye + exit_head + color_head) on labelled frames, then hand it
to the RL run.

Three things the structured observation never says have to be read from the picture: where the exit
pad is, what colour the locked door ahead is, and what colour the keycard in view is. Inside RL these
are positives on about 1% of frames and the reader gets a few gradient steps a minute (the exit
detector sat at 0% recall for seven million steps). Here it is plain supervised learning: the scripted
teacher (with random detours) plays generated levels, every frame is labelled from ground truth, and
the small CNN is fitted to that.

    python training/pretrain_exit.py --episodes 500 --out training/checkpoints/exit_eye.pt
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
    """One episode -> (frames uint8 (n,3,60,80), targets float32 (n,5)); most empty frames are dropped."""
    from doom_env import make_env
    from mapgen import spec_for_stage, write_level
    from rewards import RewardShaper, priv
    from teacher import Teacher
    from vec_env import _hires_timer

    _hires_timer()
    rng = random.Random(seed)
    want_locks = seed % 2 == 0  # half the episodes on levels with locked doors and keycards
    for _ in range(300):
        spec = spec_for_stage(rng.randint(1, 4), rng)
        if spec.exit and (not want_locks or spec.locks >= 1):
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
        if detour > 0 or rng.random() < 0.03:  # wander off the teacher's line: other views of pads, doors, keys
            detour = detour - 1 if detour > 0 else rng.randint(4, 14)
            a = np.zeros(9, np.int8)
            a[rng.choice([0, 0, 2, 3])] = 1
            a[rng.choice([4, 5])] = rng.random() < 0.7
        obs, _, te, tr, _ = env.step(a)
        done = te or tr
        if done:
            break
        sh._ps = priv(env)
        y = np.concatenate([sh.exit_targets(obs), sh.color_targets(obs)])
        if y[0] > 0 or y[3] > 0 or y[4] > 0 or rng.random() < 0.3:
            rgb = np.asarray(obs["rgb"], np.float32)
            X.append(rgb.reshape(60, 2, 80, 2, 3).mean((1, 3)).transpose(2, 0, 1).astype(np.uint8))
            Y.append(y)
    env.close()
    if not X:
        return np.zeros((0, 3, 60, 80), np.uint8), np.zeros((0, 5), np.float32)
    return np.stack(X), np.stack(Y)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=500)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--procs", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    ap.add_argument("--threads", type=int, default=0, help="CPU threads for the fitting (0 = library default)")
    ap.add_argument("--seed", type=int, default=500_000)
    ap.add_argument("--out", default=str(ROOT / "training" / "checkpoints" / "exit_eye.pt"))
    a = ap.parse_args()

    import torch
    import torch.nn.functional as F
    from model import DoomNet

    if a.threads:
        torch.set_num_threads(a.threads)
    with mp.get_context("spawn").Pool(a.procs) as pool:
        data = pool.map(gen, [a.seed + i for i in range(a.episodes)], chunksize=1)
    n_val = max(2, a.episodes // 6)  # whole episodes held out: the validation levels are unseen
    Xt, Yt = np.concatenate([d[0] for d in data[n_val:]]), np.concatenate([d[1] for d in data[n_val:]])
    Xv, Yv = np.concatenate([d[0] for d in data[:n_val]]), np.concatenate([d[1] for d in data[:n_val]])
    print(f"train {len(Xt)} frames: exit pad in view {Yt[:, 0].mean():.1%}, locked door ahead {(Yt[:, 3] > 0).mean():.1%}, "
          f"keycard in view {(Yt[:, 4] > 0).mean():.1%}; validation {len(Xv)} frames", flush=True)

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = DoomNet()
    eye, head, chead = net.eye.to(dev), net.exit_head.to(dev), net.color_head.to(dev)
    params = list(eye.parameters()) + list(head.parameters()) + list(chead.parameters())
    opt = torch.optim.Adam(params, lr=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs)
    Xt_t, Yt_t = torch.as_tensor(Xt), torch.as_tensor(Yt)
    pw = torch.tensor(float((1 - Yt[:, 0].mean()) / max(Yt[:, 0].mean(), 1e-3)), device=dev).clamp(max=8.0)
    cw = torch.tensor([1.0, 6.0, 6.0, 6.0], device=dev)

    def predict():
        eye.eval()
        with torch.no_grad():
            f = [eye(torch.as_tensor(Xv[i:i + 1024]).to(dev)) for i in range(0, len(Xv), 1024)]
            ex = torch.cat([head(x).cpu() for x in f])
            co = torch.cat([chead(x).cpu() for x in f])
        eye.train()
        return ex, co

    def report():
        ex, co = predict()
        y = Yv[:, 0] > 0.5
        for thr in (0.6, 0.8, 0.9):
            p = torch.sigmoid(ex[:, 0]).numpy() > thr
            tp = (p & y).sum()
            hit = p & y
            bear = np.abs(ex[:, 1].numpy() - Yv[:, 1])[hit].mean() * 45 if hit.any() else float("nan")
            print(f"  exit pad, confidence > {thr}: catches {tp / max(y.sum(), 1):.0%} of the frames it is in, {tp / max(p.sum(), 1):.0%} of its "
                  f"detections are real, bearing error {bear:.1f} deg")
        for name, lo, col in (("locked door", 0, 3), ("keycard", 4, 4)):
            pr = torch.softmax(co[:, lo:lo + 4], 1).numpy()
            c = pr.argmax(1)
            sure = pr.max(1) > 0.7
            said = np.where(sure, c, 0)  # what the agent would store (0 = not sure)
            truth = Yv[:, col].astype(int)
            has = truth > 0
            print(f"  {name} colour: right on {(said[has] == truth[has]).mean():.0%} of the frames with one, wrong colour on "
                  f"{((said[has] > 0) & (said[has] != truth[has])).mean():.0%}, a colour claimed on {(said[~has] > 0).mean():.1%} of the frames with none")

    for ep in range(a.epochs):
        perm = torch.randperm(len(Xt_t))
        for i in range(0, len(perm), 256):
            idx = perm[i:i + 256]
            x, y = Xt_t[idx].to(dev), Yt_t[idx].to(dev)
            if torch.rand(1).item() < 0.5:  # mirror the picture: the bearing flips sign, colours do not change
                x, y = x.flip(-1), y * torch.tensor([1.0, -1.0, 1.0, 1.0, 1.0], device=dev)
            f = eye(x)
            out, co = head(f), chead(f)
            seen = y[:, 0]
            loss = F.binary_cross_entropy_with_logits(out[:, 0], seen, pos_weight=pw) + \
                3.0 * (seen * ((out[:, 1] - y[:, 1]) ** 2 + (out[:, 2] - y[:, 2]) ** 2)).sum() / seen.sum().clamp(min=1.0) + \
                F.cross_entropy(co[:, :4], y[:, 3].long(), weight=cw) + F.cross_entropy(co[:, 4:], y[:, 4].long(), weight=cw)
            opt.zero_grad()
            loss.backward()
            opt.step()
        sched.step()
        if ep % 5 == 4 or ep == a.epochs - 1:
            print(f"epoch {ep + 1} (unseen levels):", flush=True)
            report()
    torch.save({"eye": {k: v.cpu() for k, v in eye.state_dict().items()},
                "exit_head": {k: v.cpu() for k, v in head.state_dict().items()},
                "color_head": {k: v.cpu() for k, v in chead.state_dict().items()}}, a.out)
    print("saved", a.out)


if __name__ == "__main__":
    main()
