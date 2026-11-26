"""Evaluation entry point: DoomAgent = SpatialMemory (odometry + ego map) -> DoomNet (recurrent policy).

Contract: docs/CONTRACT.md section 6. All decisions come from the trained network; the memory module
only turns raw observations into features (map painting, pose integration).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import torch

SUBMISSION_DIR = Path(__file__).resolve().parent
WEIGHTS_DIR = SUBMISSION_DIR / "weights"
if str(SUBMISSION_DIR) not in sys.path:
    sys.path.insert(0, str(SUBMISSION_DIR))

from model import DoomNet, load_compatible, obs_to_torch, split_logits, to_buttons  # noqa: E402
from spatial_memory import SpatialMemory  # noqa: E402


TEMPERATURE = 1.0  # < 1 makes the sampled actions less jittery; chosen on the held-out benchmark


class DoomAgent:
    def __init__(self, observation_space, action_space, device: str):
        self.observation_space = observation_space
        self.action_space = action_space
        torch.set_num_threads(1)
        # the net is small: one CPU thread is faster than a GPU round-trip per decision
        self.device = torch.device("cpu")
        self.net = DoomNet().to(self.device).eval()
        path = WEIGHTS_DIR / "doomnet.pt"
        if path.exists():
            ckpt = torch.load(path, map_location=self.device, weights_only=True)
            load_compatible(self.net, ckpt["model"] if "model" in ckpt else ckpt)
        self.gen = torch.Generator().manual_seed(0)
        self.temperature = float(os.environ.get("DOOM_TEMP", TEMPERATURE))  # env override: experiments only
        self.memory = SpatialMemory()
        self.reset()

    def reset(self) -> None:
        self.memory.reset()
        self.h = self.net.init_state(1, self.device)
        self.last_action = None

    @torch.no_grad()
    def act(self, observation: dict) -> np.ndarray:
        feats = self.memory.update(observation, self.last_action)
        logits, _, self.h, _ = self.net(obs_to_torch(feats, self.device), self.h)
        # sample each action head from the network's own distribution, sharpened by TEMPERATURE
        choice = [torch.multinomial(torch.softmax(l / self.temperature, -1), 1, generator=self.gen).item()
                  for l in split_logits(logits)]
        action = to_buttons(np.array(choice))
        self.last_action = action
        return action
