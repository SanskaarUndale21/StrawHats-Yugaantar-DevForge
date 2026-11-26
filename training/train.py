"""Recurrent PPO + RND curiosity + auxiliary self-supervision + curriculum, on the async actor pool.

    python training/train.py --workers 11 --envs-per-worker 2
    python training/train.py --init training/checkpoints/student.pt --bc-coef 0.5 --critic-warmup 30

Pipeline
  * AsyncSampler actors (training/sampler.py) play 64-step rollouts on fresh procedural levels with
    their own policy copy; the learner trains on whichever rollouts are ready (truncated BPTT through
    the GRU, values recomputed with the current critic, GAE, clipped PPO objective).
  * Exploration: shaped reward for unique areas + Random Network Distillation novelty.
  * Auxiliary heads: "enemy in crosshair", "door ahead".
  * Optional behaviour-cloning term against the privileged teacher that decays to zero, so RL starts
    from the distilled student without forgetting it, then goes beyond it.
  * Curriculum: stage rises when the exit rate on generated maps clears a threshold.
Checkpoints: training/checkpoints/latest.pt; policy export: submission/weights/doomnet.pt.
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

from common import RunningStat, evaluate_actions, head_ce, learner_device, load_checkpoint, to_t  # noqa: E402
from model import DoomNet, load_compatible, split_logits  # noqa: E402
from sampler import AsyncSampler  # noqa: E402
from vec_env import _hires_timer  # noqa: E402

RND_DIM = 9 * 64 + 53


class RND(nn.Module):
    """Random Network Distillation over the ray ring + vitals vector."""

    def __init__(self, d_in=RND_DIM):
        super().__init__()

        def mlp():
            return nn.Sequential(nn.Linear(d_in, 256), nn.LeakyReLU(), nn.Linear(256, 256), nn.LeakyReLU(), nn.Linear(256, 128))

        self.target, self.pred = mlp(), mlp()
        for p in self.target.parameters():
            p.requires_grad_(False)

    def forward(self, x):
        return (self.pred(x) - self.target(x)).pow(2).mean(-1)


def rnd_input(f):
    r = f["rays"]
    return np.concatenate([r.reshape(*r.shape[:-2], -1), f["vec"]], -1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=11)
    ap.add_argument("--envs-per-worker", type=int, default=2)
    ap.add_argument("--batch-rollouts", type=int, default=0, help="actor rollouts per update (default: workers)")
    ap.add_argument("--steps", type=int, default=50_000_000)
    ap.add_argument("--rollout", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--minibatches", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2.5e-4)
    ap.add_argument("--gamma", type=float, default=0.995)
    ap.add_argument("--lam", type=float, default=0.95)
    ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--ent", type=float, default=0.01)
    ap.add_argument("--int-coef", type=float, default=0.03)
    ap.add_argument("--aux-coef", type=float, default=0.1)
    ap.add_argument("--stage", type=int, default=0)
    ap.add_argument("--stage-steps", type=int, default=10_000_000, help="force next stage after this many steps")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--ckdir", default=None)
    ap.add_argument("--init", default=None, help="start from these policy weights (e.g. distilled student)")
    ap.add_argument("--bc-coef", type=float, default=0.0, help="teacher behaviour-cloning regulariser weight")
    ap.add_argument("--bc-decay", type=int, default=20_000_000, help="steps over which bc-coef decays to 0")
    ap.add_argument("--critic-warmup", type=int, default=0, help="updates with the policy frozen (value head only)")
    ap.add_argument("--target-kl", type=float, default=0.03,
                    help="stop updating on a batch once the policy moved this far from the one that played it")
    ap.add_argument("--exit-coef", type=float, default=0.5, help="weight of the exit detector's side loss")
    ap.add_argument("--fresh-lr-mult", type=float, default=10.0, help="learning-rate multiplier for newly added modules")
    ap.add_argument("--load-eye", default=None, help="pre-trained exit detector from training/pretrain_exit.py")
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()

    _hires_timer()
    torch.manual_seed(a.seed)
    dev = learner_device()
    ckdir = Path(a.ckdir) if a.ckdir else ROOT / "training" / "checkpoints"
    ckdir.mkdir(parents=True, exist_ok=True)
    wpath = ROOT / "submission" / "weights" / "doomnet.pt" if not a.ckdir else ckdir / "doomnet.pt"

    net = DoomNet().to(dev)
    rnd = RND().to(dev)
    # the freshly added parts (exit detector, critics, curiosity predictor) start from nothing and get
    # a ten times larger step than the already trained policy, which is only being fine-tuned
    fresh = [p_ for n_, p_ in net.named_parameters() if n_.split(".")[0] in ("eye", "exit_head", "vpriv", "v")]
    tuned = [p_ for n_, p_ in net.named_parameters() if n_.split(".")[0] not in ("eye", "exit_head", "vpriv", "v")]
    opt = torch.optim.Adam([{"params": tuned}, {"params": fresh + list(rnd.pred.parameters()), "lr": a.lr * a.fresh_lr_mult}],
                           lr=a.lr, eps=1e-5)
    rnd_obs, int_stat, ret_stat = RunningStat(RND_DIM), RunningStat(), RunningStat()
    step, stage, update = 0, a.stage, 0
    if a.resume and (ckdir / "latest.pt").exists():
        ck = load_checkpoint(ckdir / "latest.pt", dev)
        net.load_state_dict(ck["model"])
        rnd.load_state_dict(ck["rnd"])
        try:
            opt.load_state_dict(ck["opt"])
        except ValueError:  # checkpoint from before the optimiser had two groups: start its statistics afresh
            print("optimiser state not reused")
        rnd_obs.load(ck["rnd_obs"]), int_stat.load(ck["int_stat"]), ret_stat.load(ck["ret_stat"])
        step, stage, update = ck["step"], ck["stage"], ck["update"]
        print(f"resumed at step {step} stage {stage}")
    elif a.init:
        skipped = load_compatible(net, load_checkpoint(a.init, dev)["model"])
        print(f"initialised policy from {a.init} (new tensors: {len(skipped)})")
    if a.load_eye and Path(a.load_eye).exists():
        eye_ck = torch.load(a.load_eye, map_location=dev)
        net.eye.load_state_dict(eye_ck["eye"])
        net.exit_head.load_state_dict(eye_ck["exit_head"])
        print(f"exit detector loaded from {a.load_eye}", flush=True)
    stage_start, step0, warm0 = step, step, update

    B_roll = a.batch_rollouts or a.workers
    smp = AsyncSampler(a.workers, a.envs_per_worker, a.rollout, seed=a.seed * 100003, mode=1 if a.bc_coef > 0 else 0)
    smp.set_stage(stage)
    smp.set_weights(net.state_dict())
    T = a.rollout
    eps_hist = collections.deque(maxlen=600)
    stage_hist = collections.deque(maxlen=400)
    log = open(ckdir / "log.csv", "a", newline="")
    logw = csv.writer(log)
    t0 = time.time()
    print(f"device {dev}, actors {a.workers}x{a.envs_per_worker}, batch {B_roll * a.envs_per_worker}x{T}, "
          f"params {sum(p.numel() for p in net.parameters()) / 1e6:.2f}M", flush=True)

    while step < a.steps:
        t_wait = time.time()
        R = smp.get(B_roll)
        t_wait = time.time() - t_wait
        t_upd = time.time()
        N = R["rew"].shape[1]
        step += N * T
        for e in R["eps"]:
            eps_hist.append(e)
            if e["tags"] and e["tags"][0] not in ("freedoom", "fixture") and "noexit" not in e["tags"]:
                stage_hist.append(e["exit"])
        Bt = to_t(R["feats"], dev)  # T,N,...
        ST = torch.as_tensor(R["starts"], device=dev)
        h0 = torch.as_tensor(R["h0"], device=dev)

        # ---- intrinsic reward (RND on the next observation)
        nxt = np.concatenate([rnd_input(R["feats"])[1:], rnd_input(R["last_feats"])[None]], 0)
        rnd_obs.update(nxt.reshape(-1, RND_DIM))
        with torch.no_grad():
            x = torch.as_tensor(np.clip((nxt - rnd_obs.mean) / np.sqrt(rnd_obs.var + 1e-8), -5, 5), dtype=torch.float32, device=dev)
            r_int = rnd(x.view(-1, RND_DIM)).view(T, N).cpu().numpy()
        int_stat.update(r_int.reshape(-1))
        r_int = r_int / np.sqrt(int_stat.var + 1e-8)

        rew, dones = R["rew"], R["done"]
        ret_stat.update(R["retacc"].reshape(-1))
        r_tot = rew / np.sqrt(ret_stat.var + 1e-8) + a.int_coef * r_int

        # ---- values from the current critic, GAE
        net.eval()
        with torch.no_grad():
            flat = {k: v.reshape(T * N, *v.shape[2:]) for k, v in Bt.items()}
            _, vals, hT, _ = net(flat, h0, ST)
            _, last_v, _, _ = net(to_t(R["last_feats"], dev), hT, torch.as_tensor(R["last_starts"], device=dev)[None])
        vals = vals.view(T, N).cpu().numpy()
        last_v = last_v.cpu().numpy()
        adv = np.zeros((T, N), np.float32)
        gae = 0.0
        for t in reversed(range(T)):
            nv = last_v if t == T - 1 else vals[t + 1]
            nonterm = 1.0 - dones[t]
            delta = r_tot[t] + a.gamma * nv * nonterm - vals[t]
            gae = delta + a.gamma * a.lam * nonterm * gae
            adv[t] = gae
        rets = adv + vals

        # ---- PPO update over env-sliced sequences
        net.train()
        if dev.type == "cuda":
            net.gru.flatten_parameters()
        C = torch.as_tensor(R["choice"], device=dev)
        LP = torch.as_tensor(R["logp"], device=dev)
        ADV = torch.as_tensor(adv, device=dev)
        RET = torch.as_tensor(rets, device=dev)
        AUX = torch.as_tensor(R["aux"], device=dev)
        LAB = torch.as_tensor(R["label"], device=dev)
        bc_w = a.bc_coef * max(0.0, 1.0 - (step - step0) / max(a.bc_decay, 1))
        if a.bc_coef > 0 and bc_w == 0.0 and smp.mode.value != 0:
            smp.mode.value = 0  # the kick-start is over: stop running the teacher in the actors
            print("==> teacher switched off: rewards only from here", flush=True)
        pg_on = 0.0 if update - warm0 < a.critic_warmup else 1.0
        frac = min(step / a.steps, 1.0)
        mb = max(1, N // a.minibatches)
        stats = collections.defaultdict(list)
        moved = False
        for _ in range(a.epochs):
            if moved:
                break
            perm = np.random.permutation(N)
            for i in range(0, N, mb):
                idx = torch.as_tensor(perm[i : i + mb], device=dev)
                n = len(idx)
                o = {k: v[:, idx].reshape(T * n, *v.shape[2:]) for k, v in Bt.items()}
                logits, v, _, core = net(o, h0[:, idx], ST[:, idx])
                logp, ent = evaluate_actions(logits, C[:, idx].reshape(T * n, -1))
                adv_b = ADV[:, idx].reshape(-1)
                adv_b = (adv_b - adv_b.mean()) / (adv_b.std() + 1e-8)
                ratio = torch.exp(logp - LP[:, idx].reshape(-1))
                pg = -torch.min(ratio * adv_b, ratio.clamp(1 - a.clip, 1 + a.clip) * adv_b).mean()
                vl = 0.5 * (v - RET[:, idx].reshape(-1)).pow(2).mean()
                aux_b = AUX[:, idx].reshape(-1, AUX.shape[-1])
                aux_l = F.binary_cross_entropy_with_logits(net.aux(core), aux_b[:, :2])
                # exit detector: is a pad in view, and where (bearing and range only count when it is)
                ex, seen = net.exit_out, aux_b[:, 2]
                # a pad is in view on about 1% of frames: without the weight the detector just says "never"
                exit_l = F.binary_cross_entropy_with_logits(ex[:, 0], seen, pos_weight=torch.tensor(20.0, device=dev)) + \
                    (seen * ((ex[:, 1] - aux_b[:, 3]) ** 2 + (ex[:, 2] - aux_b[:, 4]) ** 2)).sum() / seen.sum().clamp(min=1.0)
                with torch.no_grad():
                    hit = ((ex[:, 0] > 0) == (seen > 0.5)).float().mean().item()
                    recall = (((ex[:, 0] > 0) & (seen > 0.5)).float().sum() / seen.sum().clamp(min=1.0)).item()
                rnd_l = rnd(x[:, idx].reshape(-1, RND_DIM)).mean()
                loss = pg_on * (pg - a.ent * (1 - 0.5 * frac) * ent.mean()) + 0.5 * vl + a.aux_coef * aux_l + rnd_l + a.exit_coef * exit_l
                if bc_w > 0:
                    lab = LAB[:, idx].reshape(T * n, -1)
                    m = (lab[:, 0] >= 0).float()
                    if m.sum() > 0:
                        bc, _ = head_ce(split_logits(logits), lab, m)
                        bc = (bc * m).sum() / m.sum()
                        loss = loss + bc_w * bc
                        stats["bc"].append(bc.item())
                kl_now = (LP[:, idx].reshape(-1) - logp).mean().item()
                if pg_on and kl_now > 1.5 * a.target_kl and stats["pg"]:
                    moved = True  # trust region: this batch has been used enough
                    stats["early_stop"].append(1.0)
                    break
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), 0.5)
                opt.step()
                stats["pg"].append(pg.item())
                stats["vl"].append(vl.item())
                stats["ent"].append(ent.mean().item())
                stats["aux"].append(aux_l.item())
                stats["exit_loss"].append(exit_l.item())
                stats["exit_acc"].append(hit)
                stats["exit_recall"].append(recall)
                stats["kl"].append((LP[:, idx].reshape(-1) - logp).mean().item())
                stats["clipfrac"].append(((ratio - 1).abs() > a.clip).float().mean().item())
        smp.set_weights(net.state_dict())
        update += 1
        t_upd = time.time() - t_upd

        # ---- curriculum
        if stage < 4 and (len(stage_hist) >= 300 and np.mean(stage_hist) > 0.55 - 0.05 * stage
                          or step - stage_start > a.stage_steps):
            stage += 1
            stage_start = step
            stage_hist.clear()
            smp.set_stage(stage)
            print(f"==> curriculum stage {stage}", flush=True)

        # ---- logging / checkpoints
        if eps_hist:
            E = list(eps_hist)
            ex = [e["exit"] for e in E if "noexit" not in e["tags"]]
            row = dict(step=step, stage=stage, sps=int((step - step0) / max(time.time() - t0, 1e-6)),
                       ret=np.mean([e["ret"] for e in E]), exit=float(np.mean(ex)) if ex else 0.0,
                       death=np.mean([e["death"] for e in E]), kills=np.mean([e["kills"] for e in E]),
                       cells=np.mean([e["cells"] for e in E]), keys=np.mean([e["keys"] for e in E]),
                       doors=np.mean([e["doors"] for e in E]), unlocks=np.mean([e["unlocks"] for e in E]),
                       progress=np.mean([e.get("progress", 0.0) for e in E]),
                       flips100=100 * np.sum([e.get("flips", 0) for e in E]) / max(np.sum([e["len"] for e in E]), 1),
                       idle=np.sum([e.get("idle", 0) for e in E]) / max(np.sum([e["len"] for e in E]), 1),
                       revisits=np.mean([e.get("revisits", 0) for e in E]),
                       lag=R["lag"], t_wait=t_wait, t_upd=t_upd, **{k: float(np.mean(v)) for k, v in stats.items()})
            if update % 10 == 0:
                print(" ".join(f"{k}={v:.3g}" if isinstance(v, float) else f"{k}={v}" for k, v in row.items()), flush=True)
            if log.tell() == 0:
                logw.writerow(row.keys())
            logw.writerow(row.values())
            log.flush()
        if update % 25 == 0:
            ck = dict(model=net.state_dict(), rnd=rnd.state_dict(), opt=opt.state_dict(), rnd_obs=rnd_obs.state(),
                      int_stat=int_stat.state(), ret_stat=ret_stat.state(), step=step, stage=stage, update=update)
            torch.save(ck, ckdir / "latest.pt")
            wpath.parent.mkdir(exist_ok=True)
            torch.save({"model": {k: v.detach().cpu() for k, v in net.state_dict().items()}, "step": step, "stage": stage}, wpath)
    smp.close()


if __name__ == "__main__":
    main()
