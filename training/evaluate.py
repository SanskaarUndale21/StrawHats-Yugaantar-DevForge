"""Fixed benchmark: score checkpoints on held-out levels the way the organisers describe.

Runs the real submission `DoomAgent` (same code path as evaluation) on a fixed set of levels:
  tier 1 combat, tier 2 exploration, tier 3 keycard->lock->exit, tier 4 gauntlet (generated with
  seeds never used in training), plus the template's two_rooms map and Freedoom 2 MAP01..MAP03.
Approximate score per map (weights are our guess of the hidden scorer):
  exit 100 (+ up to 20 for health/time left), key 15, unlock 15, door 3, area 1 (128-unit cells),
  kill 5, damage -0.05/hp, early death -30 * (1 - t/T).  Tier weights 1, 1.5, 2, 3.

    python training/evaluate.py --weights submission/weights/doomnet.pt training/checkpoints/student.pt
    python training/evaluate.py --teacher           # reference: the scripted teacher on the same maps
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import random
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "submission"))
sys.path.insert(0, str(ROOT / "training"))

TIER_W = {1: 1.0, 2: 1.5, 3: 2.0, 4: 3.0}
TIER_KIND = {1: "combat", 2: "explore", 3: "keys", 4: "gauntlet"}
EVAL_SEED_BASE = int(os.environ.get("DOOM_EVAL_SEED", 900_000_000))  # far from anything the training RNGs produce


def level_list(per_tier: int, freedoom: int):
    """(id, tier, wad-spec) tuples. Generated levels are re-created deterministically per worker."""
    out = []
    for tier in (1, 2, 3, 4):
        for k in range(per_tier):
            out.append((f"t{tier}_{k}", tier, ("gen", tier, k)))
    out.append(("two_rooms", 2, ("file", str(ROOT / "maps" / "two_rooms.wad"), "MAP01")))
    for i in range(1, freedoom + 1):
        out.append((f"freedoom_MAP{i:02d}", 4, ("iwad", f"MAP{i:02d}")))
    return out


def _make_level(spec):
    import vizdoom as vd
    from mapgen import LevelSpec, spec_for_stage, write_level

    if spec[0] == "file":
        return spec[1], spec[2], None, 2100
    if spec[0] == "iwad":
        return os.path.join(os.path.dirname(vd.__file__), "freedoom2.wad"), spec[1], None, 4200
    _, tier, k = spec
    rng = random.Random(EVAL_SEED_BASE + tier * 1000 + k)
    kind = TIER_KIND[tier]
    for _ in range(200):  # draw specs from mid/high stages until the requested kind appears
        ls = spec_for_stage(2 if tier < 4 else 3, rng)
        if ls.tags[0] == kind and ls.exit:
            break
    path = str(Path(tempfile.gettempdir()) / f"eval_{os.getpid()}_{tier}_{k}.wad")
    L = write_level(path, ls, EVAL_SEED_BASE + tier * 1000 + k)
    return path, "MAP01", L, 4200 if tier >= 3 else 2800


def _score(ps, stats, tic, T):
    s = 0.0
    if ps["exit_reached"]:
        s += 100 + 10 * min(ps["health"], 100) / 100 + 10 * (1 - tic / T)
    s += 15 * stats["keys"] + 15 * stats["unlocks"] + 3 * stats["doors"] + 1.0 * stats["cells"]
    s += 5 * stats["kills"] - 0.05 * stats["damage"]
    if ps["dead"]:
        s -= 30 * (1 - tic / T)
    return s


def _run_one(job):
    who, weights, (lid, tier, spec), seed = job
    if os.name == "nt":
        import ctypes

        ctypes.windll.winmm.timeBeginPeriod(1)
    import torch

    torch.set_num_threads(1)
    from doom_env import make_env
    from metrics import EpisodeMetrics
    from rewards import RewardShaper, priv

    wad, name, L, T = _make_level(spec)
    env = make_env(wad, name, episode_ticks=T, render_buffers=True)
    obs, _ = env.reset(seed=seed)
    if who == "teacher":
        if L is None:
            env.close()
            return None
        from teacher import Teacher

        agent = Teacher(L, seed=seed)
        act = lambda o: agent.act(o, priv(env))  # noqa: E731
    else:
        from agent import DoomAgent

        agent = DoomAgent(env.observation_space, env.action_space, "cpu")
        if weights:
            from common import load_checkpoint
            from model import load_compatible

            load_compatible(agent.net, load_checkpoint(weights)["model"])
        agent.reset()
        act = agent.act
    shaper = RewardShaper()
    shaper.reset(priv(env))
    met = EpisodeMetrics(env, L)
    done = False
    tic = 0
    while not done:
        a = act(obs)
        obs, _, te, tr, info = env.step(a)
        ps = priv(env)
        shaper(ps, a)
        met.step(a)
        tic = ps["tic"] if not ps["finished"] else tic + 4
        done = te or tr
    ps = env.privileged_state()
    st = dict(shaper.stats)
    extra = met.result(st["kills"])
    env.close()
    return dict(level=lid, tier=tier, exit=bool(ps["exit_reached"]), dead=bool(ps["dead"]), kills=st["kills"],
                keys=st["keys"], unlocks=st["unlocks"], doors=st["doors"], cells=st["cells"], items=st["items"],
                damage=st["damage"], score=_score(ps, st, min(tic, T), T), **extra)


def evaluate(who, weights, levels, seeds, procs):
    jobs = [(who, weights, lv, s) for lv in levels for s in seeds]
    with mp.get_context("spawn").Pool(procs) as pool:
        res = [r for r in pool.map(_run_one, jobs, chunksize=1) if r is not None]
    return res


def _detail(rs):
    """Combat / coverage detail line: sums over episodes, so empty episodes do not skew the ratios."""
    S = lambda k: float(np.nansum([r[k] for r in rs]))  # noqa: E731
    R = lambda a, b: S(a) / S(b) if S(b) else float("nan")  # noqa: E731
    n = len(rs)
    return (f"kill% {100 * R('kills', 'monsters'):.0f}  aim {100 * R('hits', 'shots'):.0f}% of {S('shots') / n:.0f} shots  "
            f"dmg dealt {S('dmg_dealt') / n:.0f} taken {S('damage') / n:.0f}  wpn switches {S('wpn_switches') / n:.1f} "
            f"(guns used {S('wpn_used') / n:.1f})  rooms seen {100 * R('rooms_seen', 'rooms_total'):.0f}%  "
            f"cleared {100 * R('rooms_cleared', 'rooms_hostile'):.0f}%  items {S('items') / n:.1f}  secrets {S('secrets') / n:.2f}\n"
            f"         movement: turn wobbles {100 * R('turn_flips', 'steps'):.1f} per 100 steps  standing still "
            f"{100 * R('idle_steps', 'steps'):.0f}%  walked {S('dist') / n:.0f} units  new ground per 1000 units "
            f"{1000 * R('new_cells', 'dist'):.1f} cells  stuck spells {S('stuck_windows') / n:.1f}/game")


def summarize(label, res):
    tot = sum(TIER_W[r["tier"]] * r["score"] for r in res) / max(len(res), 1)
    print(f"\n== {label}: weighted score {tot:.1f}  (n={len(res)})")
    for tier in (1, 2, 3, 4):
        rs = [r for r in res if r["tier"] == tier]
        if rs:
            print(f"  tier {tier} {TIER_KIND[tier]:9s} exit {np.mean([r['exit'] for r in rs]):.2f}  dead "
                  f"{np.mean([r['dead'] for r in rs]):.2f}  kills {np.mean([r['kills'] for r in rs]):.1f}  keys "
                  f"{np.mean([r['keys'] for r in rs]):.2f}  unlocks {np.mean([r['unlocks'] for r in rs]):.2f}  "
                  f"cells {np.mean([r['cells'] for r in rs]):.0f}  score {np.mean([r['score'] for r in rs]):.1f}")
            print("    " + _detail(rs))
    print("  all    " + _detail(res))
    return tot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", nargs="*", default=[])
    ap.add_argument("--teacher", action="store_true")
    ap.add_argument("--per-tier", type=int, default=6)
    ap.add_argument("--freedoom", type=int, default=3)
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--procs", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    ap.add_argument("--seed-base", type=int, default=None, help="generate a different set of levels (soak testing)")
    a = ap.parse_args()
    if a.seed_base is not None:  # read by the worker processes when they build their levels
        os.environ["DOOM_EVAL_SEED"] = str(900_000_000 + a.seed_base * 10_000)
    levels = level_list(a.per_tier, a.freedoom)
    seeds = list(range(a.seeds))
    results = {}
    if a.teacher:
        results["teacher"] = summarize("teacher (privileged, reference)", evaluate("teacher", None, levels, seeds, a.procs))
    for w in a.weights:
        results[w] = summarize(w, evaluate("agent", w, levels, seeds, a.procs))
    if len(results) > 1:
        best = max(results, key=results.get)
        print(f"\nbest: {best} ({results[best]:.1f})")


if __name__ == "__main__":
    main()
