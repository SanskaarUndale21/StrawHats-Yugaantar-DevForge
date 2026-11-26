"""Shared helpers for the learners (PPO and DAgger)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "submission"))
sys.path.insert(0, str(ROOT / "training"))

from model import split_logits  # noqa: E402


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

    def state(self):
        return dict(mean=self.mean, var=self.var, count=self.count)

    def load(self, st):
        self.mean, self.var, self.count = st["mean"], st["var"], st["count"]
        return self


def load_checkpoint(path, device="cpu") -> dict:
    """torch.load that also opens checkpoints written by older versions of train.py."""
    setattr(sys.modules["__main__"], "RunningStat", RunningStat)
    ck = torch.load(path, map_location=device, weights_only=False)
    return ck if "model" in ck else {"model": ck}


def to_t(d, device):
    return {k: torch.as_tensor(v, device=device) for k, v in d.items()}


def sample(logits):
    choice, logp = [], 0.0
    for l in split_logits(logits):
        dist = torch.distributions.Categorical(logits=l)
        a = dist.sample()
        choice.append(a)
        logp = logp + dist.log_prob(a)
    return torch.stack(choice, -1), logp


def evaluate_actions(logits, choice):
    logp, ent = 0.0, 0.0
    for i, l in enumerate(split_logits(logits)):
        dist = torch.distributions.Categorical(logits=l)
        logp = logp + dist.log_prob(choice[..., i])
        ent = ent + dist.entropy()
    return logp, ent


def learner_device():
    import os

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if dev.type == "cpu":  # cloud CPU box: most cores belong to the actor processes
        torch.set_num_threads(int(os.environ.get("LEARNER_THREADS", max(2, (os.cpu_count() or 8) // 4))))
    return dev
