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

FEAT_KEYS = ("rays", "map", "vec", "ent", "ent_type", "ent_mask", "rgb", "rgb2", "priv")
GAMMA = 0.995


def _pack(f):
    f = dict(f)
    f["map"] = np.clip(np.rint(f["map"] * 255.0), 0, 255).astype(np.uint8)  # 4x smaller rollouts
    return f


def _actor(wid, k, T, ctl, seed, out_q, shared):
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
        aux.append(np.concatenate([aux_targets(obs), s.exit_t]))
    h = net.init_state(k)
    starts = np.ones(k, np.float32)
    retacc = np.zeros(k)
    version = -1
    wbuf, wver, wlock = shared
    layout = [(n, tuple(v.shape), v.dtype, v.numel()) for n, v in net.state_dict().items()]
    try:
        while True:
            if wver.value != version:  # newer weights published: copy them out of shared memory
                with wlock:
                    flat = np.frombuffer(wbuf, np.float32).copy()
                    version = wver.value
                sd, o = {}, 0
                for n, shape, dtype, size in layout:
                    sd[n] = torch.from_numpy(flat[o:o + size].reshape(shape)).to(dtype)
                    o += size
                net.load_state_dict(sd)
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
                    exit_pred = net.exit_out.numpy()
                    color_pred = net.color_out.numpy()
                choice = choice.numpy()
                rew, done, lab = np.zeros(k, np.float32), np.zeros(k, np.float32), []
                for i, s in enumerate(slots):
                    f, r, d, ep, a, label = s.step(choice[i], exit_pred[i], color_pred[i])
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
        self.kindw = ctx.RawArray("d", 4)  # all zero = the stage's default mix of level types
        self.q = ctx.Queue(maxsize=max(4, workers))
        self.procs = []
        self.version = 0
        self._closing = False
        # weights live in shared memory: publishing is one memcpy and every actor picks the newest
        # version up at its next rollout. (Sending them down 50 pipes one after another left most
        # actors a hundred versions behind: a pipe write blocks until that actor reads it.)
        from model import DoomNet

        self._n = sum(v.numel() for v in DoomNet().state_dict().values())
        self._wbuf = ctx.RawArray("f", self._n)
        self._wver = ctx.Value("l", 0, lock=False)
        self._wlock = ctx.Lock()
        for w in range(workers):
            p = ctx.Process(target=_actor, daemon=True,
                            args=(w, envs_per_worker, rollout, (self.stage, self.beta, self.mode, self.kindw), seed + 7919 * w, self.q,
                                  (self._wbuf, self._wver, self._wlock)))
            p.start()
            self.procs.append(p)

    def set_weights(self, state_dict) -> None:
        """Publish new weights to all actors (never blocks on an actor)."""
        flat = np.concatenate([v.detach().cpu().numpy().astype(np.float32).ravel() for v in state_dict.values()])
        assert flat.size == self._n, (flat.size, self._n)
        self.version += 1
        with self._wlock:
            np.frombuffer(self._wbuf, np.float32)[:] = flat
            self._wver.value = self.version

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

    def set_kind_weights(self, w) -> None:
        for i, v in enumerate(w):
            self.kindw[i] = float(v)

    def set_stage(self, s: int):
        self.stage.value = int(s)

    def set_beta(self, b: float):
        self.beta.value = float(b)

    def close(self):
        """Stop the actors. They are terminated, not asked: an actor blocked handing over a rollout
        never reads its pipe, and a polite "close" message then blocks the learner forever once the
        pipe is full of weights (this hung the end of a distillation run for an hour)."""
        self._closing = True
        for p in self.procs:
            if p.is_alive():
                p.terminate()
        for p in self.procs:
            p.join(timeout=3)
