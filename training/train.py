"""Recurrent PPO + RND curiosity + auxiliary self-supervision + procedural curriculum (written from scratch).

    python training/train.py --workers 10 --envs-per-worker 2 --steps 50_000_000

Pipeline
  * VecDoom workers generate fresh levels per curriculum stage and compute SpatialMemory features.
  * DoomNet (submission/model.py) acts on a GPU in batch; GRU state is carried across rollouts and
    reset at episode starts; PPO trains with truncated BPTT over whole rollout sequences.
  * Exploration: shaped extrinsic reward (unique areas) + Random Network Distillation novelty over
    the ray/vitals encoding, so the agent is pulled toward observations it has not seen before.
  * Auxiliary losses: predict "enemy in crosshair" and "door ahead" from the recurrent core.
  * Curriculum: stage goes up when the exit rate on generated maps clears a threshold (or after a
    step budget), mixing combat arenas, mazes, keycard puzzles and gauntlets (training/mapgen.py).
Checkpoints go to training/checkpoints/ and the policy weights to submission/weights/doomnet.pt.
"""

from __future__ import annotations

import argparse
import collections
import copy
import csv
import queue
import threading
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

from model import HEADS, DoomNet, split_logits  # noqa: E402
from vec_env import FEAT_KEYS, VecDoom, _hires_timer  # noqa: E402


class RunningStat:
    def __init__(self, shape=()):
        self.mean = np.zeros(shape, np.float64)
        self.var = np.ones(shape, np.float64)
        self.count = 1e-4

    def update(self, x):
        x = np.asarray(x, np.float64)
        bm, bv, bc = x.mean(0), x.var(0), x.shape[0]
        d = bm - self.mean
        tot = self.count + bc
        self.mean = self.mean + d * bc / tot
        self.var = (self.var * self.count + bv * bc + d**2 * self.count * bc / tot) / tot
        self.count = tot


class RND(nn.Module):
    """Random Network Distillation over the ray ring + vitals vector."""

    def __init__(self, d_in):
        super().__init__()

        def mlp():
            return nn.Sequential(nn.Linear(d_in, 256), nn.LeakyReLU(), nn.Linear(256, 256), nn.LeakyReLU(), nn.Linear(256, 128))

        self.target, self.pred = mlp(), mlp()
        for p in self.target.parameters():
            p.requires_grad_(False)

    def forward(self, x):
        return (self.pred(x) - self.target(x)).pow(2).mean(-1)


def rnd_input(f):
    return np.concatenate([f["rays"].reshape(f["rays"].shape[0], -1), f["vec"]], -1)


def rnd_input_seq(f):
    """(T, N, ...) features -> (T, N, D) RND input."""
    r = f["rays"]
    return np.concatenate([r.reshape(*r.shape[:2], -1), f["vec"]], -1)


class Collector:
    """Plays T steps in every env with the actor network and returns the whole rollout."""

    def __init__(self, venv, actor, lock, T, dev):
        self.venv, self.actor, self.lock, self.T, self.dev = venv, actor, lock, T, dev
        self.feats, self.aux = venv.reset()
        self.h = actor.init_state(venv.n, dev)
        self.starts = np.ones(venv.n, np.float32)

    def collect(self):
        T = self.T
        buf = {k: [] for k in FEAT_KEYS}
        out = {k: [] for k in ("choice", "logp", "rew", "done", "starts", "aux", "label")}
        eps = []
        h0 = self.h.clone()
        t_roll, t_env = time.time(), 0.0
        for _ in range(T):
            for k in FEAT_KEYS:
                buf[k].append(self.feats[k])
            out["starts"].append(self.starts.copy())
            out["aux"].append(self.aux)
            with torch.no_grad(), self.lock:
                o = to_t(self.feats, self.dev)
                st = torch.as_tensor(self.starts, device=self.dev)[None]
                logits, _, self.h, _ = self.actor(o, self.h, st)
                choice, logp, _ = sample(logits)
                choice_np, logp_np = choice.cpu().numpy(), logp.cpu().numpy()
            te = time.time()
            out["label"].append(None)
            self.feats, rew, done, ep, self.aux, label = self.venv.step(choice_np)
            out["label"][-1] = label
            t_env += time.time() - te
            out["choice"].append(choice_np)
            out["logp"].append(logp_np)
            out["rew"].append(rew)
            out["done"].append(done)
            self.starts = done.copy()
            eps += ep
        R = {k: np.stack(v) for k, v in out.items()}
        R["feats"] = {k: np.stack(v) for k, v in buf.items()}
        R.update(h0=h0, eps=eps, last_feats=self.feats, last_starts=self.starts.copy(),
                 t_roll=time.time() - t_roll, t_env=t_env)
        return R


def to_t(d, device):
    out = {}
    for k, v in d.items():
        out[k] = torch.as_tensor(v, device=device)
    return out


def sample(logits):
    parts = split_logits(logits)
    choice, logp, ent = [], 0.0, 0.0
    for l in parts:
        dist = torch.distributions.Categorical(logits=l)
        a = dist.sample()
        choice.append(a)
        logp = logp + dist.log_prob(a)
        ent = ent + dist.entropy()
    return torch.stack(choice, -1), logp, ent


def evaluate(logits, choice):
    parts = split_logits(logits)
    logp, ent = 0.0, 0.0
    for i, l in enumerate(parts):
        dist = torch.distributions.Categorical(logits=l)
        logp = logp + dist.log_prob(choice[..., i])
        ent = ent + dist.entropy()
    return logp, ent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--envs-per-worker", type=int, default=2)
    ap.add_argument("--steps", type=int, default=50_000_000)
    ap.add_argument("--rollout", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--sync", action="store_true", help="collect then update (no overlap)")
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
    ap.add_argument("--ckdir", default=None, help="checkpoint dir (default training/checkpoints); "
                    "weights are exported to submission/ only with the default dir")
    ap.add_argument("--init", default=None, help="start from these policy weights (e.g. distilled student)")
    ap.add_argument("--bc-coef", type=float, default=0.0, help="teacher behaviour-cloning regulariser weight")
    ap.add_argument("--bc-decay", type=int, default=20_000_000, help="steps over which bc-coef decays to 0")
    ap.add_argument("--critic-warmup", type=int, default=0, help="updates with policy frozen (value head only)")
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()

    _hires_timer()
    torch.manual_seed(a.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if dev.type == "cpu":  # cloud CPU box: leave most cores to the env workers
        import os

        torch.set_num_threads(int(os.environ.get("LEARNER_THREADS", max(2, (os.cpu_count() or 8) // 4))))
    ckdir = Path(a.ckdir) if a.ckdir else ROOT / "training" / "checkpoints"
    ckdir.mkdir(parents=True, exist_ok=True)
    wpath = ROOT / "submission" / "weights" / "doomnet.pt" if not a.ckdir else ckdir / "doomnet.pt"

    net = DoomNet().to(dev)
    rnd = RND(9 * 64 + 49).to(dev)
    opt = torch.optim.Adam(list(net.parameters()) + list(rnd.pred.parameters()), lr=a.lr, eps=1e-5)
    rnd_obs, int_stat, ret_stat = RunningStat(9 * 64 + 49), RunningStat(), RunningStat()
    step, stage, stage_start, update = 0, a.stage, 0, 0
    if a.resume and (ckdir / "latest.pt").exists():
        ck = torch.load(ckdir / "latest.pt", map_location=dev, weights_only=False)
        net.load_state_dict(ck["model"])
        rnd.load_state_dict(ck["rnd"])
        opt.load_state_dict(ck["opt"])
        rnd_obs, int_stat, ret_stat = ck["rnd_obs"], ck["int_stat"], ck["ret_stat"]
        step, stage, update = ck["step"], ck["stage"], ck["update"]
        stage_start = step
        print(f"resumed at step {step} stage {stage}")
    elif a.init:
        sd = torch.load(a.init, map_location=dev, weights_only=False)
        net.load_state_dict(sd["model"] if "model" in sd else sd)
        print(f"initialised policy from {a.init}")

    venv = VecDoom(a.workers, a.envs_per_worker, seed=a.seed * 100003, mode=1 if a.bc_coef > 0 else 0)
    venv.set_stage(stage)
    N, T = venv.n, a.rollout
    ret_acc = np.zeros(N)
    eps_hist = collections.deque(maxlen=400)
    stage_hist = collections.deque(maxlen=300)
    log = open(ckdir / "log.csv", "a", newline="")
    logw = csv.writer(log)
    t0, step0 = time.time(), step
    print(f"device {dev}, envs {N}, params {sum(p.numel() for p in net.parameters())/1e6:.2f}M, "
          f"mode {'sync' if a.sync else 'async'}")

    # actor: a copy of the policy that keeps playing while the learner updates (at most one update of
    # lag; the stored behaviour log-probs make the PPO importance ratio correct for it)
    actor = copy.deepcopy(net).eval()
    actor_lock = threading.Lock()
    collector = Collector(venv, actor, actor_lock, T, dev)
    if a.sync:
        next_rollout = collector.collect
    else:
        q = queue.Queue(maxsize=1)
        stop = threading.Event()

        def loop():
            while not stop.is_set():
                try:
                    r = collector.collect()
                except (EOFError, BrokenPipeError, OSError):
                    return  # workers shut down
                while not stop.is_set():
                    try:
                        q.put(r, timeout=1.0)
                        break
                    except queue.Full:
                        pass

        th = threading.Thread(target=loop, daemon=True)
        th.start()
        next_rollout = q.get

    while step < a.steps:
        t_wait = time.time()
        R = next_rollout()
        t_wait = time.time() - t_wait
        t_upd = time.time()
        step += N * T
        for e in R["eps"]:
            eps_hist.append(e)
            if e["tags"] and e["tags"][0] not in ("freedoom", "fixture") and "noexit" not in e["tags"]:
                stage_hist.append(e["exit"])
        B = {k: torch.as_tensor(v, device=dev) for k, v in R["feats"].items()}  # T,N,...
        ST = torch.as_tensor(R["starts"], device=dev)
        h0 = R["h0"]

        # ---- intrinsic reward (RND on the next observation)
        last = {k: R["last_feats"][k][None] for k in ("rays", "vec")}
        nxt = np.concatenate([rnd_input_seq(R["feats"])[1:], rnd_input_seq(last)], 0)
        rnd_obs.update(nxt.reshape(-1, nxt.shape[-1]))
        with torch.no_grad():
            x = torch.as_tensor(np.clip((nxt - rnd_obs.mean) / np.sqrt(rnd_obs.var + 1e-8), -5, 5),
                                dtype=torch.float32, device=dev)
            r_int = rnd(x.view(-1, x.shape[-1])).view(T, N).cpu().numpy()
        int_stat.update(r_int.reshape(-1))
        r_int = r_int / np.sqrt(int_stat.var + 1e-8)

        rew, dones = R["rew"], R["done"]
        for t in range(T):  # reward scaling by running std of discounted returns
            ret_acc = ret_acc * a.gamma + rew[t]
            ret_stat.update(ret_acc)
            ret_acc = ret_acc * (1 - dones[t])
        r_tot = rew / np.sqrt(ret_stat.var + 1e-8) + a.int_coef * r_int

        # values from the *current* network (fresh critic for GAE) + bootstrap value
        net.eval()
        with torch.no_grad():
            flat = {k: v.reshape(T * N, *v.shape[2:]) for k, v in B.items()}
            _, vals, hT, _ = net(flat, h0, ST)
            lf = to_t(R["last_feats"], dev)
            _, last_v, _, _ = net(lf, hT, torch.as_tensor(R["last_starts"], device=dev)[None])
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
        net.gru.flatten_parameters()
        C = torch.as_tensor(R["choice"], device=dev)
        LP = torch.as_tensor(R["logp"], device=dev)
        ADV = torch.as_tensor(adv, device=dev)
        RET = torch.as_tensor(rets, device=dev)
        AUX = torch.as_tensor(R["aux"], device=dev)
        LAB = torch.as_tensor(R["label"], device=dev)  # T,N,6 teacher labels (-1 = none)
        bc_w = a.bc_coef * max(0.0, 1.0 - (step - step0) / max(a.bc_decay, 1))
        pg_on = 0.0 if update < a.critic_warmup else 1.0
        XR = x
        mb = max(1, N // a.minibatches)
        stats = collections.defaultdict(list)
        frac = min(step / a.steps, 1.0)
        for _ in range(a.epochs):
            perm = np.random.permutation(N)
            for i in range(0, N, mb):
                idx = torch.as_tensor(perm[i : i + mb], device=dev)
                o = {k: v[:, idx].reshape(T * len(idx), *v.shape[2:]) for k, v in B.items()}
                logits, v, _, core = net(o, h0[:, idx], ST[:, idx])
                logp, ent = evaluate(logits, C[:, idx].reshape(T * len(idx), -1))
                adv_b = ADV[:, idx].reshape(-1)
                adv_b = (adv_b - adv_b.mean()) / (adv_b.std() + 1e-8)
                ratio = torch.exp(logp - LP[:, idx].reshape(-1))
                pg = -torch.min(ratio * adv_b, ratio.clamp(1 - a.clip, 1 + a.clip) * adv_b).mean()
                vl = 0.5 * (v - RET[:, idx].reshape(-1)).pow(2).mean()
                aux_l = F.binary_cross_entropy_with_logits(net.aux(core), AUX[:, idx].reshape(-1, 2))
                rnd_l = rnd(XR[:, idx].reshape(-1, XR.shape[-1])).mean()
                loss = pg_on * (pg - a.ent * (1 - 0.5 * frac) * ent.mean()) + 0.5 * vl + a.aux_coef * aux_l + rnd_l
                if bc_w > 0:
                    lab = LAB[:, idx].reshape(T * len(idx), -1)
                    m = (lab[:, 0] >= 0).float()
                    if m.sum() > 0:
                        bc = sum(F.cross_entropy(l, lab[:, j].clamp(min=0), reduction="none")
                                 for j, l in enumerate(split_logits(logits)))
                        bc = (bc * m).sum() / m.sum()
                        loss = loss + bc_w * bc
                        stats["bc"].append(bc.item())
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), 0.5)
                opt.step()
                stats["pg"].append(pg.item())
                stats["vl"].append(vl.item())
                stats["ent"].append(ent.mean().item())
                stats["aux"].append(aux_l.item())
                stats["rnd"].append(rnd_l.item())
                stats["kl"].append((LP[:, idx].reshape(-1) - logp).mean().item())
                stats["clipfrac"].append(((ratio - 1).abs() > a.clip).float().mean().item())
        with actor_lock:  # hand the new weights to the actor
            actor.load_state_dict(net.state_dict())
            actor.gru.flatten_parameters()
        update += 1
        t_upd = time.time() - t_upd
        t_roll, t_env = R["t_roll"], R["t_env"]

        # ---- curriculum
        if len(stage_hist) >= 200 and np.mean(stage_hist) > 0.5 - 0.05 * stage and stage < 4 or \
                step - stage_start > a.stage_steps and stage < 4:
            stage += 1
            stage_start = step
            stage_hist.clear()
            venv.set_stage(stage)
            print(f"==> curriculum stage {stage}")

        # ---- logging / checkpoints
        if eps_hist:
            E = list(eps_hist)
            row = dict(step=step, stage=stage, sps=int((step - step0) / max(time.time() - t0, 1e-6)),
                       ret=np.mean([e["ret"] for e in E]), exit=np.mean([e["exit"] for e in E]),
                       death=np.mean([e["death"] for e in E]), kills=np.mean([e["kills"] for e in E]),
                       cells=np.mean([e["cells"] for e in E]), keys=np.mean([e["keys"] for e in E]),
                       doors=np.mean([e["doors"] for e in E]), unlocks=np.mean([e["unlocks"] for e in E]),
                       t_roll=t_roll, t_env=t_env, t_upd=t_upd, t_wait=t_wait, **{k: float(np.mean(v)) for k, v in stats.items()})
            if update % 5 == 0:
                print(" ".join(f"{k}={v:.3g}" if isinstance(v, float) else f"{k}={v}" for k, v in row.items()), flush=True)
            if update == 1 or log.tell() == 0:
                logw.writerow(row.keys())
            logw.writerow(row.values())
            log.flush()
        if update % 20 == 0:
            ck = dict(model=net.state_dict(), rnd=rnd.state_dict(), opt=opt.state_dict(), rnd_obs=rnd_obs,
                      int_stat=int_stat, ret_stat=ret_stat, step=step, stage=stage, update=update)
            torch.save(ck, ckdir / "latest.pt")
            wpath.parent.mkdir(exist_ok=True)
            torch.save({"model": {k: v.detach().cpu() for k, v in net.state_dict().items()}, "step": step, "stage": stage}, wpath)
    if not a.sync:
        stop.set()
    venv.close()


if __name__ == "__main__":
    main()
