# StrawHats: an agent that maps, remembers and plays DOOM

Yugaantar 2026, DevForge, Track 1: "Can It Play DOOM?"

A neural network that plays unseen DOOM levels zero-shot. It builds its own map of the level as it
plays, remembers doors, keycards and the exit, and every button it presses is sampled from the
network. The two-page project brief is in [docs/PROJECT_BRIEF.pdf](docs/PROJECT_BRIEF.pdf).

## What is in the submission

```
submission/agent.py            DoomAgent: load weights once, reset() per map, act() every 4 tics
submission/model.py            DoomNet, 1.57M parameters
submission/spatial_memory.py   position tracking and the occupancy map the agent builds
submission/weights/doomnet.pt  trained weights (Git LFS)
```

Decisions come only from trained parameters. There is no rule tree, no path planner and no aiming
formula in `submission/`. `spatial_memory.py` prepares the network's input from what the sensors
report; it never chooses an action.

## How it works

**Spatial memory.** The observation has no coordinates, so the agent tracks its own pose. Heading
is integrated from its own turn actions with the engine's exact keyboard turn profile; position from
the velocity in `vitals` with the exact friction model, corrected against the previous ray scan
whenever it touches a wall. Measured on 300 random levels: 13 map units of average position error
(a map cell is 64). Rays then paint free space, walls, doors and locked doors into a two-scale
occupancy grid, and the network reads a heading-aligned crop of it.

**Network.** A circular 1-D CNN over the 360 degree ray ring, an attention pool over the entity
list (any number of enemies), a CNN over the map crop, an MLP over vitals and memory pointers, and
two small picture CNNs, fused into a GRU with six action heads (move, strafe, turn, fire, use,
next weapon).

**Learned perception of what the sensors omit.** The observation never says where the exit is or
which colour a locked door is. A picture CNN is trained to report both, and its confident sightings
are written into the map, so the exit and each door colour are remembered.

**Training.** Phase 1 copies a scripted teacher for a kick-start. Phase 2 is recurrent PPO written
for this project, with a critic that reads ground truth during training only, a reward for true
walking progress toward the next objective, and a level mix that follows the policy's weaknesses.
Details and commands are in [training/README.md](training/README.md).

## Results

Two fresh sets of 112 generated levels each (seeds never used in training). Scores are our own
estimate of the hidden scorer; exit rates are exact.

| Tier | Exit reached |
| --- | --- |
| 1 Combat and survival | 92% |
| 2 Multi-room exploration | 96% |
| 3 Keycard, locked door, exit | 85% |
| 4 Full gauntlet | 28% |

Benchmark score on the two sets: 325 and 361. On the same levels the copy-phase network alone scores
207 and 234, and the scripted teacher it was started from scores 284 and 305, so the reward phase took
the policy past its teacher. `validate.py` passes on Linux at about 5 ms per decision and 330 MB of RAM.

Known weaknesses: gauntlets are finished about one time in four, mostly lost when the exit is never
found before death or the clock (when the exit has been seen it is reached every time); aim is about
70% of shots; 36% of its moves are still onto ground already walked; keycard colour is not read from
the picture (the agent learns a key's colour when it picks it up).

## Run it

```bash
uv sync
uv run python validate.py                              # the organisers' checks
uv run python training/watch.py --tour                 # watch one level of each tier, with its map
uv run python training/evaluate.py --weights submission/weights/doomnet.pt --seeds 4 --seed-base 6161
```

## Repository layout

```
submission/   the agent that is evaluated
training/     the full training pipeline, benchmark and viewer (read by reviewers, not by the scorer)
docs/         PROJECT_BRIEF.pdf, and the organisers' contract, rules, submission and evaluation notes
env/          the official ViZDoom wrapper from the starter template (unchanged)
maps/         the template's test map
tests/        the template's tests
validate.py   the organisers' validator
```
