"""Diagnose the scripted teacher: where does it fail (stuck, no goal, deaths, exploration)?"""

import random
import sys
import tempfile
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "training"))
from evaluate import EVAL_SEED_BASE, TIER_KIND  # noqa: E402
from mapgen import spec_for_stage, write_level  # noqa: E402
from teacher import Teacher  # noqa: E402
from vec_env import _hires_timer  # noqa: E402


def main(tier, n):
    _hires_timer()
    from doom_env import make_env

    for k in range(n):
        rng = random.Random(EVAL_SEED_BASE + tier * 1000 + k)
        for _ in range(200):
            ls = spec_for_stage(2 if tier < 4 else 3, rng)
            if ls.tags[0] == TIER_KIND[tier] and ls.exit:
                break
        wad = str(Path(tempfile.gettempdir()) / "diag.wad")
        L = write_level(wad, ls, EVAL_SEED_BASE + tier * 1000 + k)
        T = 2800 if tier < 3 else 4200
        env = make_env(wad, episode_ticks=T, render_buffers=False)
        obs, _ = env.reset(seed=0)
        t = Teacher(L, seed=0)
        goals, escapes, fights, done, steps = Counter(), 0, 0, False, 0
        hp_lost_in_fight = 0.0
        while not done:
            ps = env.privileged_state()
            esc_before = t.escape
            a = t.act(obs, ps)
            if t.escape > esc_before:
                escapes += 1
            m = np.asarray(obs["visible_entities"]["mask"]).astype(bool) & np.isin(obs["visible_entities"]["entity_type"], (1, 2, 3))
            if m.any():
                fights += 1
            goals[t.goal if not m.any() else "combat"] += 1
            obs, r, te, tr, info = env.step(a)
            steps += 1
            done = te or tr
        ps = env.privileged_state()
        floor = sum(1 for row in L.g for ch in row if ch != "#")
        print(f"tier{tier}#{k} size {L.w}x{L.h} exit={info['exit_reached']} dead={ps['dead']} steps={steps} "
              f"seen={t.seen.sum()}/{floor} exit_seen={bool(t.exit and t.seen[t.exit])} escapes={escapes} "
              f"kills={ps['kills']} goals={dict(goals)}")
        env.close()


if __name__ == "__main__":
    main(int(sys.argv[1]), int(sys.argv[2]))
