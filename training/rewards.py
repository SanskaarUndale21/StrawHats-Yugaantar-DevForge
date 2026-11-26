"""Reward shaping from privileged ground truth (training only, never visible to the agent).

Mirrors the evaluation categories: exit (largest), progression (keys, doors, unlocks), exploration
(unique areas), combat & survival (kills, damage, early death), plus a small time cost.
"""

from __future__ import annotations

import math

import numpy as np

AREA = 128.0  # map units per exploration cell


class RewardShaper:
    W = dict(explore=0.04, kill=1.0, item=0.1, damage=0.01, key=4.0, door=0.5, unlock=4.0,
             exit=12.0, exit_health=3.0, exit_time=4.0, death=3.0, step=0.001, use_spam=0.003)

    def reset(self, ps: dict) -> None:
        self.cells = {self._cell(ps)}
        self.kills, self.items, self.damage = ps["kills"], ps["items"], ps["damage_taken"]
        self.keys = np.asarray(ps["keys"]).copy()
        dl = ps["door_lock"]
        self.door_ids = np.flatnonzero(dl >= 0) if dl is not None else np.zeros(0, int)
        self.door_lock = dl
        self.opened = set()
        self.stats = dict(kills=0, keys=0, doors=0, unlocks=0, cells=1, exit=0, death=0, damage=0.0, items=0)
        self.last_use = 0

    @staticmethod
    def _cell(ps):
        return (math.floor(ps["x"] / AREA), math.floor(ps["y"] / AREA))

    def __call__(self, ps: dict, action: np.ndarray) -> float:
        W, st = self.W, self.stats
        r = -W["step"]
        c = self._cell(ps)
        if c not in self.cells:
            self.cells.add(c)
            st["cells"] += 1
            r += W["explore"]
        dk = ps["kills"] - self.kills
        if dk > 0:
            r += W["kill"] * dk
            st["kills"] += dk
        di = ps["items"] - self.items
        if di > 0:
            r += W["item"] * di
            st["items"] += di
        dd = ps["damage_taken"] - self.damage
        if dd > 0:
            r -= W["damage"] * dd
            st["damage"] += dd
        self.kills, self.items, self.damage = ps["kills"], ps["items"], ps["damage_taken"]
        keys = np.asarray(ps["keys"])
        nk = int(((keys > 0) & (self.keys == 0)).sum())
        if nk:
            r += W["key"] * nk
            st["keys"] += nk
        self.keys = keys.copy()
        if len(self.door_ids) and ps["sector_ceiling"] is not None:
            gap = ps["sector_ceiling"][self.door_ids] - ps["sector_floor"][self.door_ids]
            for sid in self.door_ids[gap > 40]:
                if sid not in self.opened:
                    self.opened.add(int(sid))
                    if self.door_lock[sid] > 0:
                        r += W["unlock"]
                        st["unlocks"] += 1
                    else:
                        r += W["door"]
                        st["doors"] += 1
        # discourage hammering USE every step (it bounces doors back)
        if action[7] and self.last_use:
            r -= W["use_spam"]
        self.last_use = int(action[7])
        if ps["finished"]:
            frac = ps["tic"] / max(ps["episode_ticks"], 1)
            if ps["exit_reached"]:
                r += W["exit"] + W["exit_health"] * min(ps["health"], 100) / 100 + W["exit_time"] * (1 - frac)
                st["exit"] = 1
            elif ps["dead"]:
                r -= W["death"] * (1.5 - frac)
                st["death"] = 1
        return r


def aux_targets(obs: dict) -> np.ndarray:
    """Self-supervised targets: [enemy near the crosshair, door/locked door straight ahead]."""
    e = obs["visible_entities"]
    m = np.asarray(e["mask"]).astype(bool)
    t = np.asarray(e["entity_type"])
    f = np.asarray(e["features"])
    enemy = m & (t >= 1) & (t <= 3) & (np.abs(f[:, 1]) < 0.08) & (f[:, 2] > 0)
    hit = np.asarray(obs["rays"]["hit_type"])
    d = np.asarray(obs["rays"]["distance"])
    front = [62, 63, 0, 1, 2]
    door = np.any(((hit[front] == 2) | (hit[front] == 3)) & (d[front] < 0.25))
    return np.array([float(enemy.any()), float(door)], np.float32)
