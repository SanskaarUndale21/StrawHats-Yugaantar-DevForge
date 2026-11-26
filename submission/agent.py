"""Evaluation entry point: DoomAgent = SpatialMemory (odometry + ego map) -> DoomNet (recurrent policy).

Contract: docs/CONTRACT.md section 6. All decisions come from the trained network; the memory module
only turns raw observations into features (map painting, pose integration).
"""

from __future__ import annotations

import math
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


# Sampling temperature, chosen on 112 held-out levels with the same weights: 1.0 scored 190 with 23
# left-right turn reversals per 100 steps, 0.35 scored 218 with 11 (the scripted teacher: 9).
TEMPERATURE = 0.35
EXIT_CONFIDENCE = 0.6
TEMP_MOVE = 0.35  # move / strafe / turn heads
USE_REFIRE = 0    # decision steps USE stays released after a press


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
        # environment overrides exist for experiments only; the scorer runs the constants above
        self.temperature = float(os.environ.get("DOOM_TEMP", TEMPERATURE))
        self.temp_move = float(os.environ.get("DOOM_TEMP_MOVE", os.environ.get("DOOM_TEMP", TEMP_MOVE)))
        self.use_refire = int(os.environ.get("DOOM_USE_REFIRE", USE_REFIRE))
        self.memory = SpatialMemory()
        self.reset()

    def reset(self) -> None:
        self.memory.reset()
        self.h = self.net.init_state(1, self.device)
        self.last_action = None
        self.since_use = 10_000

    @torch.no_grad()
    def act(self, observation: dict) -> np.ndarray:
        feats = self.memory.update(observation, self.last_action)
        logits, _, self.h, _ = self.net(obs_to_torch(feats, self.device), self.h)
        heads = list(split_logits(logits))
        if self.since_use < self.use_refire:
            # USE was pressed a moment ago: pressing it again while the door is still moving sends it
            # shut. Hold the button off until the door has had time to open.
            heads[4] = heads[4] + torch.tensor([0.0, -1e9])
        # sample each action head from the network's own distribution, sharpened by its temperature
        temps = (self.temp_move, self.temp_move, self.temp_move, self.temperature, self.temperature, self.temperature)
        choice = [torch.multinomial(torch.softmax(l / t, -1), 1, generator=self.gen).item() for l, t in zip(heads, temps)]
        self.since_use = 0 if choice[4] == 1 else self.since_use + 1
        ex = self.net.exit_out[0]
        if torch.sigmoid(ex[0]).item() > EXIT_CONFIDENCE:  # the learned detector sees an exit pad: store it
            self.memory.note_exit(float(ex[1].clamp(-1, 1)) * math.radians(45.0), float(ex[2].clamp(0.02, 1.2)) * 1024.0)
        action = to_buttons(np.array(choice))
        self.last_action = action
        return action
