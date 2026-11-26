"""DoomNet: spatial-memory recurrent actor-critic (bespoke, 1.57M parameters, about 5 ms per decision on one CPU thread).

Inputs (built by spatial_memory.SpatialMemory):
  rays     (9, 64)        per-ray distance, proximity, hit-type one-hot, already-visited fraction
  map      (18, 21, 21)   heading-aligned crops of the agent's own occupancy map at 2 scales (9 layers each)
  rgb      (3, 30, 40)    low-res colour thumbnail (uint8)
  rgb2     (3, 60, 80)    sharper colour picture for the exit and door-colour reader (uint8)
  vec      (77,)          vitals, previous action, pointers to remembered doors, locked doors by colour, keycards, exit
  ent      (32, 5) + ent_type (32,) + ent_mask (32,)   variable-size set of visible entities

Modules:
  RingEncoder    circular 1-D convolutions over the 360 degree ray ring (rotation-equivariant,
                 wrap-around padding: "behind me" is next to "behind me").
  EntitySetPool  per-entity MLP with class embeddings, then K learned queries attend over the masked
                 set (threats, pickups, keys ...): handles any number of enemies, order invariant.
  MapEncoder     small CNN over the two-scale ego map (spatial memory read-out).
  PixelEncoder   small CNN over the thumbnail.
  ExitEye        CNN over the sharper picture with two side heads: exit pad in view (with bearing and
                 range) and the colour of the locked door ahead / keycard in view. The observation gives
                 neither; confident sightings are written into the map by the agent.
  GRU core       recurrent memory for the short-term tactical state.
  Heads          factorised categorical policy: move (fwd/none/back), strafe (L/none/R),
                 turn (L/none/R), attack, use, next-weapon; auxiliary heads predict "enemy in crosshair"
                 and "a door is ahead" from the core state (self-supervised).
  vpriv          training-only critic that also reads ground truth; not stored in the shipped weights
                 and never called at evaluation (the plain value head is unused there as well).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

RAY_CH, NUM_RAYS, MAP_CH, CROP, VEC_DIM, N_ENT, N_ENT_TYPES = 9, 64, 18, 21, 77, 32, 12
PRIV_DIM = 12  # training-only ground truth for the critic (never part of the policy's input)
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


class PixelEncoder(nn.Module):
    """Tiny CNN over a 30x40 colour thumbnail. The rays say nothing about floor textures, so this is
    how the agent recognises the exit pad (and keycard colours) when they are in view."""

    def __init__(self, out=64):
        super().__init__()
        self.c1 = nn.Conv2d(3, 16, 5, stride=2, padding=2)   # 15x20
        self.c2 = nn.Conv2d(16, 32, 3, stride=2, padding=1)  # 8x10
        self.c3 = nn.Conv2d(32, 32, 3, stride=2, padding=1)  # 4x5
        self.fc = nn.Linear(32 * 4 * 5, out)

    def forward(self, x):
        x = x.float() / 255.0 - 0.5
        x = F.relu(self.c1(x))
        x = F.relu(self.c2(x))
        x = F.relu(self.c3(x))
        return F.relu(self.fc(x.flatten(1)))


class ExitEye(nn.Module):
    """CNN over a 60x80 colour picture. Trained with a side task (is an exit pad in view, at what
    bearing and range) so that the largest reward in the game does not depend on a 40x30 thumbnail."""

    def __init__(self, out=64):
        super().__init__()
        self.c0 = nn.Conv2d(3, 16, 5, stride=2, padding=2)   # 30x40
        self.c1 = nn.Conv2d(16, 32, 3, stride=2, padding=1)  # 15x20
        self.c2 = nn.Conv2d(32, 32, 3, stride=2, padding=1)  # 8x10
        self.c3 = nn.Conv2d(32, 32, 3, stride=2, padding=1)  # 4x5
        self.fc = nn.Linear(32 * 4 * 5, out)

    def forward(self, x):
        x = x.float() / 255.0 - 0.5
        for c in (self.c0, self.c1, self.c2, self.c3):
            x = F.relu(c(x))
        return F.relu(self.fc(x.flatten(1)))


class DoomNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.rays = RingEncoder(192)
        self.ents = EntitySetPool(64, 4, 128)
        self.map = MapEncoder(192)
        self.vec = nn.Sequential(nn.Linear(VEC_DIM, 128), nn.ReLU())
        self.pix = PixelEncoder(64)
        self.eye = ExitEye(64)
        self.exit_head = nn.Linear(64, 3)  # [exit pad in view (logit), bearing / 45 deg, range / 1024]
        # colour reader: 4 logits for the locked door ahead and 4 for the keycard in view
        # (none / blue / yellow / red). The structured observation never gives these colours.
        self.color_head = nn.Linear(64, 8)
        self.fuse = nn.Sequential(nn.Linear(192 + 128 + 192 + 128 + 64 + 64, HIDDEN), nn.ReLU())
        self.gru = nn.GRU(HIDDEN, HIDDEN)
        self.pi = nn.Linear(HIDDEN + HIDDEN, sum(HEADS))
        self.v = nn.Linear(HIDDEN + HIDDEN, 1)
        self.aux = nn.Linear(HIDDEN, 2)  # [enemy near crosshair, door/locked door ahead]
        # asymmetric critic: during training the value estimate also reads ground truth (distance to
        # the objective, where the route goes). The policy never does, and the scorer never calls it.
        self.vpriv = nn.Sequential(nn.Linear(HIDDEN + HIDDEN + PRIV_DIM, 256), nn.ReLU(), nn.Linear(256, 1))
        nn.init.orthogonal_(self.pi.weight, 0.01)
        nn.init.zeros_(self.pi.bias)
        nn.init.orthogonal_(self.v.weight, 1.0)

    def encode(self, o):
        e = torch.cat([self.rays(o["rays"]), self.ents(o["ent"], o["ent_type"], o["ent_mask"]),
                       self.map(o["map"].float() if o["map"].dtype != torch.uint8 else o["map"].float() / 255.0),
                       self.vec(o["vec"]), self.pix(o["rgb"])], -1)
        eye = self.eye(o["rgb2"])
        self.exit_out = self.exit_head(eye)  # read by the agent (to remember the exit) and by the side loss
        self.color_out = self.color_head(eye)
        return self.fuse(torch.cat([e, eye], -1))

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
        if "priv" in o:
            # the critic reads the policy's features but may not reshape them: a fresh critic's large
            # early errors otherwise flow into the shared layers and undo what the policy knows
            # (measured: benchmark 231 -> 155 in 0.4M steps with the policy loss switched off)
            v = self.vpriv(torch.cat([z.detach(), o["priv"].float()], -1)).squeeze(-1)
        else:
            v = self.v(z).squeeze(-1)
        return self.pi(z), v, h, core

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


def load_compatible(net: nn.Module, state: dict) -> list:
    """Load every tensor whose name and shape match (lets a checkpoint from an older input layout
    warm-start a newer network). Returns the names that were skipped."""
    own = net.state_dict()
    ok = {k: v for k, v in state.items() if k in own and own[k].shape == v.shape}
    for k, v in state.items():  # a layer that only gained inputs: keep the old ones, start the new at zero
        if k in own and k not in ok and v.dim() == own[k].dim() >= 2 and v.shape[0] == own[k].shape[0] \
                and v.shape[2:] == own[k].shape[2:] and v.shape[1] < own[k].shape[1]:
            w = torch.zeros_like(own[k])
            w[:, : v.shape[1]] = v
            ok[k] = w
    own.update(ok)
    net.load_state_dict(own)
    return [k for k in own if k not in ok]


def decode_colors(color_logits, confidence=0.7):
    """Colour-head output for one observation -> (locked-door colour, keycard colour), 0 = not sure."""
    import numpy as np

    z = np.asarray(color_logits, np.float64).reshape(2, 4)
    p = np.exp(z - z.max(1, keepdims=True))
    p /= p.sum(1, keepdims=True)
    c = p.argmax(1)
    return tuple(int(ci) if pi[ci] > confidence else 0 for ci, pi in zip(c, p))


def obs_to_torch(feats: dict, device) -> dict:
    """Single feature dict (numpy) -> batch-of-1 tensors."""
    return {k: torch.as_tensor(v, device=device).unsqueeze(0) for k, v in feats.items()}
