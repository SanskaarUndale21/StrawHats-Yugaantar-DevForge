"""Teacher -> student distillation with DAgger (Ross et al. 2011) for the recurrent DoomNet.

The privileged teacher (training/teacher.py) labels every state the *student* visits; the executed
action is the teacher's with probability beta (annealed 1 -> beta_min) and the student's own sample
otherwise, so the student learns to recover from its own mistakes instead of only copying perfect
trajectories. The loss is cross-entropy on the six factorised action heads through the GRU
(truncated BPTT over 64-step sequences) plus the auxiliary perception losses.

    python training/distill.py --workers 11 --envs-per-worker 2 --steps 6_000_000
    python training/distill.py --init training/checkpoints/latest.pt     # warm-start from an RL run

Then fine-tune with RL, keeping a decaying behaviour-cloning term:
    python training/train.py --init training/checkpoints/student.pt --bc-coef 0.5 --critic-warmup 30
"""

from __future__ import annotations

import argparse
import collections
import csv
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "submission"))
sys.path.insert(0, str(ROOT / "training"))

from model import DoomNet, split_logits  # noqa: E402
from train import RunningStat, sample, to_t  # noqa: E402

# RL checkpoints pickle RunningStat as __main__.RunningStat (train.py run as a script)
setattr(sys.modules["__main__"], "RunningStat", RunningStat)
from vec_env import FEAT_KEYS, VecDoom, _hires_timer  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=11)
    ap.add_argument("--envs-per-worker", type=int, default=2)
    ap.add_argument("--steps", type=int, default=6_000_000)
    ap.add_argument("--rollout", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--minibatches", type=int, default=2)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--beta-steps", type=int, default=3_000_000, help="steps for beta to anneal 1 -> beta-min")
    ap.add_argument("--beta-min", type=float, default=0.1)
    ap.add_argument("--stages", default="0,1,2,3,4", help="curriculum stages cycled during distillation")
    ap.add_argument("--stage-every", type=int, default=8, help="updates per stage before cycling")
    ap.add_argument("--init", default=None)
    ap.add_argument("--ckdir", default=None)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()

    _hires_timer()
    torch.manual_seed(a.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if dev.type == "cpu":  # cloud CPU box: leave most cores to the env workers
        import os

        torch.set_num_threads(max(2, (os.cpu_count() or 8) // 4))
    ckdir = Path(a.ckdir) if a.ckdir else ROOT / "training" / "checkpoints"
    ckdir.mkdir(parents=True, exist_ok=True)
    net = DoomNet().to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=a.lr, eps=1e-5)
    step, update = 0, 0
    if a.resume and (ckdir / "distill.pt").exists():
        ck = torch.load(ckdir / "distill.pt", map_location=dev, weights_only=False)
        net.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        step, update = ck["step"], ck["update"]
        print(f"resumed distillation at step {step}")
    elif a.init:
        ck = torch.load(a.init, map_location=dev, weights_only=False)
        net.load_state_dict(ck["model"] if "model" in ck else ck)
        print(f"student initialised from {a.init}")

    stages = [int(x) for x in a.stages.split(",")]
    venv = VecDoom(a.workers, a.envs_per_worker, seed=a.seed * 7717, mode=2)
    N, T = venv.n, a.rollout
    venv.set_stage(stages[0])
    feats, aux = venv.reset()
    h = net.init_state(N, dev)
    starts = np.ones(N, np.float32)
    eps_hist = collections.deque(maxlen=300)
    log = open(ckdir / "distill_log.csv", "a", newline="")
    logw = csv.writer(log)
    t0, step0 = time.time(), step
    print(f"DAgger distillation: device {dev}, envs {N}")

    while step < a.steps:
        beta = max(a.beta_min, 1.0 - step / a.beta_steps)
        venv.set_beta(beta)
        venv.set_stage(stages[(update // a.stage_every) % len(stages)])
        buf = {k: [] for k in FEAT_KEYS}
        B_lab, B_start, B_aux = [], [], []
        h0 = h.clone()
        net.eval()
        for _ in range(T):
            for k in FEAT_KEYS:
                buf[k].append(feats[k])
            B_start.append(starts.copy())
            B_aux.append(aux)
            with torch.no_grad():
                logits, _, h, _ = net(to_t(feats, dev), h, torch.as_tensor(starts, device=dev)[None])
                choice, _, _ = sample(logits)
            feats, _, done, eps, aux, label = venv.step(choice.cpu().numpy())
            B_lab.append(label)
            starts = done.copy()
            eps_hist.extend(eps)
        step += N * T

        net.train()
        net.gru.flatten_parameters()
        Bt = {k: torch.as_tensor(np.stack(v), device=dev) for k, v in buf.items()}
        LAB = torch.as_tensor(np.stack(B_lab), device=dev)
        ST = torch.as_tensor(np.stack(B_start), device=dev)
        AUX = torch.as_tensor(np.stack(B_aux), device=dev)
        mb = max(1, N // a.minibatches)
        stats = collections.defaultdict(list)
        for _ in range(a.epochs):
            perm = np.random.permutation(N)
            for i in range(0, N, mb):
                idx = torch.as_tensor(perm[i : i + mb], device=dev)
                o = {k: v[:, idx].reshape(T * len(idx), *v.shape[2:]) for k, v in Bt.items()}
                logits, _, _, core = net(o, h0[:, idx], ST[:, idx])
                lab = LAB[:, idx].reshape(T * len(idx), -1)
                m = (lab[:, 0] >= 0).float()
                ce, acc = 0.0, []
                for j, l in enumerate(split_logits(logits)):
                    ce = ce + F.cross_entropy(l, lab[:, j].clamp(min=0), reduction="none")
                    acc.append((((l.argmax(-1) == lab[:, j]).float() * m).sum() / m.sum().clamp(min=1)).item())
                ce = (ce * m).sum() / m.sum().clamp(min=1)
                aux_l = F.binary_cross_entropy_with_logits(net.aux(core), AUX[:, idx].reshape(-1, 2))
                loss = ce + 0.1 * aux_l
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                opt.step()
                stats["ce"].append(ce.item())
                for name, v in zip(("move", "strafe", "turn", "fire", "use", "wpn"), acc):
                    stats["acc_" + name].append(v)
        update += 1

        if eps_hist:
            E = [e for e in eps_hist]
            ex = [e["exit"] for e in E if "noexit" not in e["tags"]]
            row = dict(step=step, beta=round(beta, 3), sps=int((step - step0) / (time.time() - t0)),
                       exit=float(np.mean(ex)) if ex else 0.0, death=float(np.mean([e["death"] for e in E])),
                       kills=float(np.mean([e["kills"] for e in E])), keys=float(np.mean([e["keys"] for e in E])),
                       unlocks=float(np.mean([e["unlocks"] for e in E])), cells=float(np.mean([e["cells"] for e in E])),
                       **{k: float(np.mean(v)) for k, v in stats.items()})
            if update % 5 == 0:
                print(" ".join(f"{k}={v:.3g}" if isinstance(v, float) else f"{k}={v}" for k, v in row.items()), flush=True)
            if log.tell() == 0:
                logw.writerow(row.keys())
            logw.writerow(row.values())
            log.flush()
        if update % 20 == 0:
            torch.save(dict(model=net.state_dict(), opt=opt.state_dict(), step=step, update=update), ckdir / "distill.pt")
            torch.save({"model": {k: v.detach().cpu() for k, v in net.state_dict().items()}, "step": step}, ckdir / "student.pt")
    torch.save({"model": {k: v.detach().cpu() for k, v in net.state_dict().items()}, "step": step}, ckdir / "student.pt")
    venv.close()


if __name__ == "__main__":
    main()
