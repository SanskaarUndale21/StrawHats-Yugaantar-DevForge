"""Evaluation entry point: DoomAgent = SpatialMemory (odometry + ego map) -> DoomNet (recurrent policy).

Contract: docs/CONTRACT.md section 6. All decisions come from the trained network; the memory module
only turns raw observations into features (map painting, pose integration).
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import torch

SUBMISSION_DIR = Path(__file__).resolve().parent
WEIGHTS_DIR = SUBMISSION_DIR / "weights"
if str(SUBMISSION_DIR) not in sys.path:
    sys.path.insert(0, str(SUBMISSION_DIR))

from model import DoomNet, decode_colors, load_compatible, obs_to_torch, split_logits, to_buttons  # noqa: E402
from spatial_memory import SpatialMemory  # noqa: E402


# Sampling temperature, chosen on 112 held-out levels with the same weights: 1.0 scored 190 with 23
# left-right turn reversals per 100 steps, 0.35 scored 218 with 11 (the scripted teacher: 9).
TEMPERATURE = 0.35
EXIT_CONFIDENCE = 0.6


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
        # every button comes from the network: each action head is sampled from the distribution the
        # network outputs (sharpened by TEMPERATURE). No rule adds, blocks or overrides an action.
        choice = [torch.multinomial(torch.softmax(l / TEMPERATURE, -1), 1, generator=self.gen).item()
                  for l in split_logits(logits)]
        ex = self.net.exit_out[0]
        if torch.sigmoid(ex[0]).item() > EXIT_CONFIDENCE:  # the learned detector sees an exit pad: store it
            self.memory.note_exit(float(ex[1].clamp(-1, 1)) * math.radians(45.0), float(ex[2].clamp(0.02, 1.2)) * 1024.0)
        self.memory.note_colors(*decode_colors(self.net.color_out[0].numpy()))  # colours it recognised
        action = to_buttons(np.array(choice))
        self.last_action = action
        return action
