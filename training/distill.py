"""Teacher -> student distillation with DAgger (Ross et al. 2011) for the recurrent DoomNet.

The privileged teacher (training/teacher.py) labels every state the *student* visits; the executed
action is the teacher's with probability beta (annealed 1 -> beta_min) and the student's own sample
otherwise, so the student learns to recover from its own mistakes instead of only copying perfect
trajectories. Loss: cross-entropy on the six factorised action heads through the GRU (truncated
BPTT over 64-step sequences) + the auxiliary perception losses. Runs on the async actor pool.

    python training/distill.py --workers 11 --envs-per-worker 2 --steps 6_000_000
    python training/distill.py --init training/checkpoints/student_v1.pt     # warm start

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

from common import head_ce, learner_device, load_checkpoint, to_t  # noqa: E402
from model import DoomNet, load_compatible, split_logits  # noqa: E402
from sampler import AsyncSampler  # noqa: E402
from vec_env import _hires_timer  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=11)
    ap.add_argument("--envs-per-worker", type=int, default=2)
    ap.add_argument("--batch-rollouts", type=int, default=0)
    ap.add_argument("--steps", type=int, default=6_000_000)
    ap.add_argument("--rollout", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--minibatches", type=int, default=2)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--beta-steps", type=int, default=3_000_000, help="steps for beta to anneal 1 -> beta-min")
    ap.add_argument("--beta-min", type=float, default=0.1)
    ap.add_argument("--stages", default="0,1,2,3,4", help="curriculum stages cycled during distillation")
    ap.add_argument("--stage-every", type=int, default=60, help="updates per stage before cycling")
    ap.add_argument("--init", default=None)
    ap.add_argument("--ckdir", default=None)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()

    _hires_timer()
    torch.manual_seed(a.seed)
    dev = learner_device()
    ckdir = Path(a.ckdir) if a.ckdir else ROOT / "training" / "checkpoints"
    ckdir.mkdir(parents=True, exist_ok=True)
    net = DoomNet().to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=a.lr, eps=1e-5)
    step, update = 0, 0
    if a.resume and (ckdir / "distill.pt").exists():
        ck = load_checkpoint(ckdir / "distill.pt", dev)
        net.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        step, update = ck["step"], ck["update"]
        print(f"resumed distillation at step {step}")
    elif a.init:
        skipped = load_compatible(net, load_checkpoint(a.init, dev)["model"])
        print(f"student initialised from {a.init} (new tensors: {len(skipped)})")

    stages = [int(x) for x in a.stages.split(",")]
    B_roll = a.batch_rollouts or a.workers
    smp = AsyncSampler(a.workers, a.envs_per_worker, a.rollout, seed=a.seed * 7717, mode=2)
    smp.set_stage(stages[0])
    smp.set_beta(max(a.beta_min, 1.0 - step / a.beta_steps))
    smp.set_weights(net.state_dict())
    T = a.rollout
    eps_hist = collections.deque(maxlen=400)
    log = open(ckdir / "distill_log.csv", "a", newline="")
    logw = csv.writer(log)
    t0, step0 = time.time(), step
    print(f"DAgger distillation: device {dev}, actors {a.workers}x{a.envs_per_worker}", flush=True)

    def save():
        torch.save(dict(model=net.state_dict(), opt=opt.state_dict(), step=step, update=update), ckdir / "distill.pt")
        torch.save({"model": {k: v.detach().cpu() for k, v in net.state_dict().items()}, "step": step}, ckdir / "student.pt")

    while step < a.steps:
        beta = max(a.beta_min, 1.0 - step / a.beta_steps)
        smp.set_beta(beta)
        smp.set_stage(stages[(update // a.stage_every) % len(stages)])
        R = smp.get(B_roll)
        N = R["rew"].shape[1]
        step += N * T
        eps_hist.extend(R["eps"])

        net.train()
        if dev.type == "cuda":
            net.gru.flatten_parameters()
        Bt = to_t(R["feats"], dev)
        LAB = torch.as_tensor(R["label"], device=dev)
        ST = torch.as_tensor(R["starts"], device=dev)
        AUX = torch.as_tensor(R["aux"], device=dev)
        h0 = torch.as_tensor(R["h0"], device=dev)
        mb = max(1, N // a.minibatches)
        stats = collections.defaultdict(list)
        for _ in range(a.epochs):
            perm = np.random.permutation(N)
            for i in range(0, N, mb):
                idx = torch.as_tensor(perm[i : i + mb], device=dev)
                n = len(idx)
                o = {k: v[:, idx].reshape(T * n, *v.shape[2:]) for k, v in Bt.items()}
                logits, _, _, core = net(o, h0[:, idx], ST[:, idx])
                lab = LAB[:, idx].reshape(T * n, -1)
                m = (lab[:, 0] >= 0).float()
                ce, acc = head_ce(split_logits(logits), lab, m)
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
        smp.set_weights(net.state_dict())
        update += 1

        if eps_hist:
            E = list(eps_hist)
            ex = [e["exit"] for e in E if "noexit" not in e["tags"]]
            row = dict(step=step, beta=round(beta, 3), sps=int((step - step0) / (time.time() - t0)),
                       exit=float(np.mean(ex)) if ex else 0.0, death=float(np.mean([e["death"] for e in E])),
                       kills=float(np.mean([e["kills"] for e in E])), keys=float(np.mean([e["keys"] for e in E])),
                       unlocks=float(np.mean([e["unlocks"] for e in E])), cells=float(np.mean([e["cells"] for e in E])),
                       **{k: float(np.mean(v)) for k, v in stats.items()})
            if update % 10 == 0:
                print(" ".join(f"{k}={v:.3g}" if isinstance(v, float) else f"{k}={v}" for k, v in row.items()), flush=True)
            if log.tell() == 0:
                logw.writerow(row.keys())
            logw.writerow(row.values())
            log.flush()
        if update % 25 == 0:
            save()
    save()
    smp.close()


if __name__ == "__main__":
    main()
