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


# heads: 0 move, 1 strafe, 2 turn, 3 fire, 4 use, 5 weapon
# cap of the inverse-frequency class weight per rare head. The policy *samples* its actions, so a
# weight of w makes it press roughly w times as often as the teacher: 25 turned one gun swap per
# game into fifty. Small caps keep the learning signal without the button spam.
RARE_HEADS = {4: 4.0, 5: 2.0}


def head_ce(logits_list, lab, m):
    """Per-head cross-entropy, lightly class-weighted for the rarely pressed heads (USE, weapon).
    Returns (per-sample loss, per-head accuracy list)."""
    import torch.nn.functional as F

    total, acc = 0.0, []
    for j, l in enumerate(logits_list):
        y = lab[:, j].clamp(min=0)
        nc = l.shape[-1]
        with torch.no_grad():
            if j in RARE_HEADS:
                cnt = torch.bincount(y[m > 0], minlength=nc).float()
                w = (cnt.sum() / (nc * cnt.clamp(min=1.0))).clamp(min=1.0, max=RARE_HEADS[j])
            else:
                w = torch.ones(nc, device=l.device)
        total = total + F.cross_entropy(l, y, weight=w, reduction="none")
        acc.append((((l.argmax(-1) == lab[:, j]).float() * m).sum() / m.sum().clamp(min=1)).item())
    return total, acc
