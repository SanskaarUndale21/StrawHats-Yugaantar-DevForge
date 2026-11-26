"""DoomNet: spatial-memory recurrent actor-critic (bespoke, ~1.4M params, CPU-friendly).

Inputs (built by spatial_memory.SpatialMemory):
  rays     (9, 64)        per-ray distance, proximity, hit-type one-hot, already-visited fraction
  map      (14, 21, 21)   ego-centric, heading-aligned crops of the remembered map at 2 scales
  vec      (49,)          vitals, previous action, pointers to remembered locked door/door/keycard
  ent      (32, 5) + ent_type (32,) + ent_mask (32,)   variable-size set of visible entities

Modules:
  RingEncoder    circular 1-D convolutions over the 360 degree ray ring (rotation-equivariant,
                 wrap-around padding: "behind me" is next to "behind me").
  EntitySetPool  per-entity MLP with class embeddings, then K learned queries attend over the masked
                 set (threats, pickups, keys ...): handles any number of enemies, order invariant.
  MapEncoder     small CNN over the two-scale ego map (spatial memory read-out).
  GRU core       recurrent memory for the short-term tactical state.
  Heads          factorised categorical policy: move (fwd/none/back), strafe (L/none/R),
                 turn (L/none/R), attack, use, next-weapon; value head; auxiliary heads predict
                 "enemy in crosshair" and "a door is ahead" from the core state (self-supervised).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

RAY_CH, NUM_RAYS, MAP_CH, CROP, VEC_DIM, N_ENT, N_ENT_TYPES = 9, 64, 14, 21, 49, 32, 12
HEADS = (3, 3, 3, 2, 2, 2)  # move, strafe, turn, attack, use, weapon
HIDDEN = 256


class RingEncoder(nn.Module):
    def __init__(self, out=192):
        super().__init__()
        self.c1 = nn.Conv1d(RAY_CH, 32, 5, padding=2, padding_mode="circular")
        self.c2 = nn.Conv1d(32, 48, 5, stride=2, padding=2, padding_mode="circular")
        self.c3 = nn.Conv1d(48, 48, 3, stride=2, padding=1, padding_mode="circular")
        self.fc = nn.Linear(48 * 16, out)

    def forward(self, x):
        x = F.relu(self.c1(x))
        x = F.relu(self.c2(x))
        x = F.relu(self.c3(x))
        return F.relu(self.fc(x.flatten(1)))


class EntitySetPool(nn.Module):
    def __init__(self, d=64, queries=4, out=128):
        super().__init__()
        self.emb = nn.Embedding(N_ENT_TYPES, 16)
        self.mlp = nn.Sequential(nn.Linear(5 + 16, d), nn.ReLU(), nn.Linear(d, d), nn.ReLU())
        self.q = nn.Parameter(torch.randn(queries, d) * 0.1)
        self.k = nn.Linear(d, d)
        self.v = nn.Linear(d, d)
        self.out = nn.Linear(queries * d + 1, out)
        self.scale = d ** -0.5

    def forward(self, ent, etype, mask):
        h = self.mlp(torch.cat([ent, self.emb(etype)], -1))            # B,N,d
        att = torch.einsum("qd,bnd->bqn", self.q, self.k(h)) * self.scale  # B,Q,N
        att = att.masked_fill(mask[:, None, :] < 0.5, -1e9)
        w = torch.softmax(att, -1) * (mask.sum(-1, keepdim=True)[:, :, None] > 0)
        pooled = torch.einsum("bqn,bnd->bqd", w, self.v(h)).flatten(1)
        count = mask.sum(-1, keepdim=True) / 8.0
        return F.relu(self.out(torch.cat([pooled, count], -1)))


class MapEncoder(nn.Module):
    def __init__(self, out=192):
        super().__init__()
        self.c1 = nn.Conv2d(MAP_CH, 32, 3, padding=1)
        self.c2 = nn.Conv2d(32, 48, 3, stride=2, padding=1)   # 11
        self.c3 = nn.Conv2d(48, 64, 3, stride=2, padding=1)   # 6
        self.fc = nn.Linear(64 * 6 * 6, out)

    def forward(self, x):
        x = F.relu(self.c1(x))
        x = F.relu(self.c2(x))
        x = F.relu(self.c3(x))
        return F.relu(self.fc(x.flatten(1)))


class DoomNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.rays = RingEncoder(192)
        self.ents = EntitySetPool(64, 4, 128)
        self.map = MapEncoder(192)
        self.vec = nn.Sequential(nn.Linear(VEC_DIM, 128), nn.ReLU())
        self.fuse = nn.Sequential(nn.Linear(192 + 128 + 192 + 128, HIDDEN), nn.ReLU())
        self.gru = nn.GRU(HIDDEN, HIDDEN)
        self.pi = nn.Linear(HIDDEN + HIDDEN, sum(HEADS))
        self.v = nn.Linear(HIDDEN + HIDDEN, 1)
        self.aux = nn.Linear(HIDDEN, 2)  # [enemy near crosshair, door/locked door ahead]
        nn.init.orthogonal_(self.pi.weight, 0.01)
        nn.init.zeros_(self.pi.bias)
        nn.init.orthogonal_(self.v.weight, 1.0)

    def encode(self, o):
        e = torch.cat([self.rays(o["rays"]), self.ents(o["ent"], o["ent_type"], o["ent_mask"]),
                       self.map(o["map"]), self.vec(o["vec"])], -1)
        return self.fuse(e)

    def forward(self, o, h, done=None):
        """o: dict of (T*B, ...) tensors; h: (1, B, H); done: (T, B) episode-start mask. Returns
        logits (T*B, sum(HEADS)), value (T*B,), new h, core (T*B, H)."""
        x = self.encode(o)
        B = h.shape[1]
        T = x.shape[0] // B
        x = x.view(T, B, -1)
        if done is None:
            y, h = self.gru(x, h)
            core = y.reshape(T * B, -1)
        else:
            # run the GRU over whole segments between time steps where some episode restarts
            # (identical to stepping one tic at a time with resets, much fewer kernel launches)
            cuts = [0] + [t for t in range(1, T) if bool(done[t].any())] + [T]
            outs = []
            for s, e in zip(cuts[:-1], cuts[1:]):
                h = h * (1.0 - done[s]).view(1, B, 1)
                y, h = self.gru(x[s:e], h)
                outs.append(y)
            core = torch.cat(outs, 0).reshape(T * B, -1)
        z = torch.cat([core, x.view(T * B, -1)], -1)
        return self.pi(z), self.v(z).squeeze(-1), h, core

    def init_state(self, b=1, device="cpu"):
        return torch.zeros(1, b, HIDDEN, device=device)


def split_logits(logits):
    return torch.split(logits, HEADS, dim=-1)


def to_buttons(choice):
    """choice: (..., 6) int tensor/array of head indices -> (..., 9) MultiBinary buttons."""
    import numpy as np

    c = np.asarray(choice)
    out = np.zeros(c.shape[:-1] + (9,), np.int8)
    out[..., 0] = c[..., 0] == 0  # forward
    out[..., 1] = c[..., 0] == 2  # backward
    out[..., 2] = c[..., 1] == 0  # strafe left
    out[..., 3] = c[..., 1] == 2  # strafe right
    out[..., 4] = c[..., 2] == 0  # turn left
    out[..., 5] = c[..., 2] == 2  # turn right
    out[..., 6] = c[..., 3] == 1
    out[..., 7] = c[..., 4] == 1
    out[..., 8] = c[..., 5] == 1
    return out


def obs_to_torch(feats: dict, device) -> dict:
    """Single feature dict (numpy) -> batch-of-1 tensors."""
    return {k: torch.as_tensor(v, device=device).unsqueeze(0) for k, v in feats.items()}
