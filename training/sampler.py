"""Asynchronous actor pool (IMPALA / Sample-Factory style, written for this project).

Every actor process owns a few DOOM environments *and its own CPU copy of the policy*. It plays
T-step rollouts on its own and pushes them to a queue; the learner pulls whichever rollouts are ready,
updates, and broadcasts new weights. No actor ever waits for another one, so a slow level load or a
long episode reset in one process no longer stalls the whole batch (the synchronous version lost
~70% of its time that way). Policy lag is at most a couple of updates; the stored behaviour log-probs
keep PPO's importance ratio correct, and values are recomputed by the learner.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import queue
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "submission"))
sys.path.insert(0, str(ROOT / "training"))

FEAT_KEYS = ("rays", "map", "vec", "ent", "ent_type", "ent_mask", "rgb")
GAMMA = 0.995


def _pack(f):
    f = dict(f)
    f["map"] = np.clip(np.rint(f["map"] * 255.0), 0, 255).astype(np.uint8)  # 4x smaller rollouts
    return f


def _actor(wid, k, T, ctl, seed, out_q, w_conn):
    time.sleep(0.25 * wid)  # stagger engine start-up
    import torch

    torch.set_num_threads(1)
    from common import sample
    from model import DoomNet
    from rewards import aux_targets
    from vec_env import _hires_timer, _Slot

    _hires_timer()
    torch.manual_seed(seed)
    net = DoomNet().eval()
    slots = [_Slot(wid, wid * k + i, ctl, seed + 1000 * i) for i in range(k)]
    feats, aux = [], []
    for s in slots:
        obs, f = s.reset()
        feats.append(_pack(f))
        aux.append(aux_targets(obs))
    h = net.init_state(k)
    starts = np.ones(k, np.float32)
    retacc = np.zeros(k)
    version = -1
    try:
        while True:
            msg = None
            while w_conn.poll():
                msg = w_conn.recv()
            if msg is not None:
                if msg == "close":
                    break
                version, sd = msg
                net.load_state_dict({n: torch.from_numpy(v) for n, v in sd.items()})
            buf = {key: [] for key in FEAT_KEYS}
            out = {key: [] for key in ("choice", "logp", "rew", "done", "starts", "aux", "label", "retacc")}
            eps = []
            h0 = h.numpy().copy()
            for _ in range(T):
                st = {key: np.stack([f[key] for f in feats]) for key in FEAT_KEYS}
                for key in FEAT_KEYS:
                    buf[key].append(st[key])
                out["starts"].append(starts.copy())
                out["aux"].append(np.stack(aux))
                with torch.no_grad():
                    logits, _, h, _ = net({key: torch.from_numpy(v) for key, v in st.items()}, h,
                                          torch.from_numpy(starts)[None])
                    choice, logp = sample(logits)
                choice = choice.numpy()
                rew, done, lab = np.zeros(k, np.float32), np.zeros(k, np.float32), []
                for i, s in enumerate(slots):
                    f, r, d, ep, a, label = s.step(choice[i])
                    feats[i], aux[i] = _pack(f), a
                    rew[i], done[i] = r, float(d)
                    lab.append(label)
                    if ep is not None:
                        eps.append(ep)
                retacc = retacc * GAMMA + rew
                out["retacc"].append(retacc.copy())
                retacc = retacc * (1 - done)
                out["choice"].append(choice)
                out["logp"].append(logp.numpy())
                out["rew"].append(rew)
                out["done"].append(done)
                out["label"].append(np.stack(lab))
                starts = done.copy()
            R = {key: np.stack(v) for key, v in out.items()}
            R["feats"] = {key: np.stack(v) for key, v in buf.items()}
            R["last_feats"] = {key: np.stack([f[key] for f in feats]) for key in FEAT_KEYS}
            R.update(h0=h0, last_starts=starts.copy(), eps=eps, version=version)
            out_q.put(R)
    except (EOFError, BrokenPipeError, KeyboardInterrupt):
        pass
    finally:
        for s in slots:
            s.close()


class AsyncSampler:
    def __init__(self, workers: int, envs_per_worker: int, rollout: int, seed: int = 0, mode: int = 0):
        """mode 0: RL only; 1: RL + teacher labels; 2: DAgger (teacher acts with probability beta)."""
        ctx = mp.get_context("spawn")
        os.makedirs("_vizdoom", exist_ok=True)
        self.stage = ctx.Value("i", 0)
        self.beta = ctx.Value("d", 0.0)
        self.mode = ctx.Value("i", mode)
        self.k, self.T, self.workers = envs_per_worker, rollout, workers
        self.q = ctx.Queue(maxsize=max(4, workers))
        self.conns, self.procs = [], []
        self.version = 0
        for w in range(workers):
            a, b = ctx.Pipe()
            p = ctx.Process(target=_actor, daemon=True,
                            args=(w, envs_per_worker, rollout, (self.stage, self.beta, self.mode), seed + 7919 * w, self.q, b))
            p.start()
            self.conns.append(a)
            self.procs.append(p)

    def set_weights(self, state_dict) -> None:
        self.version += 1
        sd = {n: v.detach().cpu().numpy() for n, v in state_dict.items()}
        for c in self.conns:
            try:
                c.send((self.version, sd))
            except (BrokenPipeError, OSError):
                pass

    def get(self, n: int) -> dict:
        """Block until n actor rollouts are available and merge them into one batch (T, n*k, ...)."""
        rs = []
        while len(rs) < n:
            try:
                rs.append(self.q.get(timeout=60))
            except queue.Empty:
                if not any(p.is_alive() for p in self.procs):
                    raise RuntimeError("all actor processes died")
        B = {key: np.concatenate([r[key] for r in rs], 1)
             for key in ("choice", "logp", "rew", "done", "starts", "aux", "label", "retacc", "h0")}
        B["feats"] = {key: np.concatenate([r["feats"][key] for r in rs], 1) for key in FEAT_KEYS}
        B["last_feats"] = {key: np.concatenate([r["last_feats"][key] for r in rs], 0) for key in FEAT_KEYS}
        B["last_starts"] = np.concatenate([r["last_starts"] for r in rs], 0)
        B["eps"] = [e for r in rs for e in r["eps"]]
        B["lag"] = self.version - float(np.mean([r["version"] for r in rs]))
        return B

    def set_stage(self, s: int):
        self.stage.value = int(s)

    def set_beta(self, b: float):
        self.beta.value = float(b)

    def close(self):
        for c in self.conns:
            try:
                c.send("close")
            except Exception:
                pass
        time.sleep(1.0)
        for p in self.procs:
            if p.is_alive():
                p.terminate()
