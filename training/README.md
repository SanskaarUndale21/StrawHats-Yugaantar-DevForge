# Training pipeline (StrawHats)

```
mapgen.py              procedural UDMF level generator + curriculum specs (combat / explore / keys / gauntlet)
vec_env.py             multiprocess envs: fresh levels, SpatialMemory features, shaped rewards, aux targets
rewards.py             reward shaping from privileged ground truth (training only)
train.py               recurrent PPO + RND curiosity + auxiliary losses + curriculum (from scratch, no SB3)
calibrate_odometry.py  fits the odometry constants used by submission/spatial_memory.py
```

## Quick start

```bash
uv sync
uv pip install torch --index-url https://download.pytorch.org/whl/cu128   # training only (image has torch)
uv run python training/train.py --workers 10 --envs-per-worker 2           # writes submission/weights/doomnet.pt
uv run python training/train.py --resume                                   # continue from training/checkpoints/latest.pt
uv run python training/mapgen.py --stage 3 --seed 5 --out gen.wad          # preview a generated level (ASCII)
uv run python validate.py --map gen.wad:MAP01
```

**Windows note:** ViZDoom runs about 17x faster after `timeBeginPeriod(1)` (the default 15.6 ms timer
makes every engine sync sleep). `vec_env._hires_timer()` does this in every worker.

## Architecture (submission/)

1. **SpatialMemory** (`spatial_memory.py`): the observation has no global coordinates, so the agent
   keeps its own private frame.
   * Heading comes from its own TURN actions. The keyboard turn profile was measured exactly: 5 slow
     tics at 1.7578 deg, then 3.5156 deg per tic, with the held counter shared by both turn buttons.
     Heading error is 0.
   * Position comes from integrating the ego-frame velocity in `vitals`. Coefficients are a
     least-squares fit to ground truth, and drift stays around one or two 64-unit cells over 500 steps.
   * Each step, rays paint free space, walls, doors and locked doors into a toroidal grid at two
     scales (64 and 256 units per cell). Keycards and pickups get projected into the grid, and visit
     counts build up.
   * The policy reads a heading-aligned 21x21 crop of both scales (14 channels), the per-ray
     "already visited" fraction, and pointers to the nearest remembered locked door, door and
     keycard. This is what makes backtracking possible: once a locked door has been seen it stays in
     memory after it leaves the field of view.
2. **DoomNet** (`model.py`, about 1.4M params, about 2 ms per decision on 1 CPU thread):
   * a circular 1-D conv ring encoder over the 360 degree rays
   * an entity set encoder (per-entity MLP plus class embedding, with 4 learned attention queries
     over the masked set) that handles any number of enemies
   * a CNN over the ego map
   * an MLP over vitals and memory pointers
   * a GRU core
   * a factorised policy (move, strafe, turn, attack, use, weapon) plus value and auxiliary heads

## Training

* **Recurrent PPO:** truncated BPTT over 128-step rollouts, GAE with gamma 0.995 and lambda 0.95,
  clipped objective, entropy annealing, return-scaled rewards.
* **Curiosity:** Random Network Distillation novelty over the ray ring and vitals, plus an
  exploration bonus for unique 128-unit areas, so the agent leaves corners and explores unseen rooms.
* **Auxiliary self-supervision:** heads predict "enemy in crosshair" and "door ahead".
* **Shaped reward:** mirrors the scorer's categories.
  * exit: +12, plus bonuses for health and time left
  * keys: +4 each
  * unlocking a locked door: +4
  * opening a door: +0.5
  * new area: +0.04
  * kill: +1
  * pickup: +0.1
  * damage: -0.01 per HP
  * death: -3 to -4.5 (bigger the earlier it happens)
  * USE spam: small penalty
* **Curriculum:**
  * The stage goes from 0 to 4. It moves up when the exit rate on generated maps clears a threshold,
    or after a step budget.
  * Each stage mixes four kinds of level: combat arenas, multi-room mazes with doors, keycard to
    locked door to exit puzzles (1-2 locks), and gauntlets.
  * Levels randomise size, room count, heights, light, textures, monsters and items.
  * The 13 template fixture maps and the Freedoom 2 campaign maps are mixed in for realism.
* **Domain randomisation:** episode tick cap drawn from 2100 to 4200, and random spawn seeds.
