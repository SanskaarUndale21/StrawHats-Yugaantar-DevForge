# StrawHats: an agent that maps, remembers and plays DOOM

Yugaantar 2026, DevForge, Track 1: "Can It Play DOOM?"

A neural network that plays unseen DOOM levels zero-shot. It builds its own map of the level as it
plays, remembers doors, keycards and the exit, and every button it presses is sampled from the
network. The two-page project brief is in [docs/PROJECT_BRIEF.pdf](docs/PROJECT_BRIEF.pdf).

| | |
| --- | --- |
| Network | 1.57M parameters, 5.8 MB of weights |
| Speed | about 5 ms per decision on one CPU thread (limit 40 ms) |
| Memory | under 1 GB of RAM (limit 8 GB); no GPU needed |
| Map accuracy | 13 map units of average position error (one map cell is 64) |
| Validator | `validate.py` and `validate.py --docker` both pass on the submitted commit |

## Contents

1. [What is in the submission](#1-what-is-in-the-submission)
2. [Quick start](#2-quick-start)
3. [How the agent works, step by step](#3-how-the-agent-works-step-by-step)
4. [How it was trained](#4-how-it-was-trained)
5. [Results](#5-results)
6. [What we tried that did not help](#6-what-we-tried-that-did-not-help)
7. [Rules checklist](#7-rules-checklist)
8. [Reproduce the training](#8-reproduce-the-training)
9. [Repository layout](#9-repository-layout)
10. [Known weaknesses](#10-known-weaknesses)

## 1. What is in the submission

```
submission/agent.py            DoomAgent: load weights once, reset() per map, act() every 4 tics
submission/model.py            DoomNet, the network (1.57M parameters)
submission/spatial_memory.py   position tracking and the occupancy map the agent builds
submission/weights/doomnet.pt  trained weights (Git LFS)
submission/requirements.txt    empty: nothing beyond the official image is needed
```

Decisions come only from trained parameters. There is no rule tree, no path planner and no aiming
formula in `submission/`. `spatial_memory.py` prepares the network's input from what the sensors
report; it never chooses an action.

## 2. Quick start

You need Python 3.13 and [uv](https://docs.astral.sh/uv/). Git LFS must be installed so the weights
are downloaded with the clone.

```bash
git lfs install
git clone https://github.com/SanskaarUndale21/StrawHats-Yugaantar-DevForge.git
cd StrawHats-Yugaantar-DevForge
uv sync
```

**Check the submission the way the organisers do**

```bash
uv run python validate.py                      # interface, speed, memory, a full game on the test map

docker build -t devforge-doom .                # the official image, built from this repository
uv run python validate.py --docker --image devforge-doom            # same checks inside the sandbox
uv run python validate.py --docker --image devforge-doom --no-gpu   # on a machine without a GPU
```

**Watch it play**

```bash
uv run python training/watch.py --tour         # one level of each tier, with the map it builds
uv run python training/watch.py --live         # keeps playing fresh random levels
```

The viewer shows the game on the left and, on the right, the occupancy map the agent is drawing
from its own position estimate: walls, free space, doors, locked doors, keycards, its path and the
exit once seen. The real level layout is not shown unless `--truth` is passed.

**Score it on fresh levels**

```bash
uv run python training/evaluate.py --weights submission/weights/doomnet.pt --seeds 4 --seed-base 6161
```

This generates 112 levels (four tiers plus the template map and three Freedoom 2 maps) from a seed
never used in training, plays them all, and prints exit rate, deaths, kills, keys, aim and movement
statistics per tier. Change `--seed-base` for a different set of levels.

**Use the agent from your own code**

```python
from doom_env import make_env
from submission.agent import DoomAgent

env = make_env("maps/two_rooms.wad", "MAP01")
agent = DoomAgent(env.observation_space, env.action_space, "cpu")
obs, _ = env.reset(seed=0)
agent.reset()
done = False
while not done:
    obs, _, terminated, truncated, _ = env.step(agent.act(obs))
    done = terminated or truncated
```

## 3. How the agent works, step by step

Every 4 game tics the agent receives one observation and returns 9 button states. One decision goes
through four stages.

### 3.1 What comes in

The official `doom_env` observation: 64 distance rays around the player with a hit type each
(wall, door, locked door, enemy, item), a list of visible entities, the vitals (health, armour,
ammo, selected weapon, keys held, own velocity), and a low-resolution colour picture. There are no
coordinates and no map. The depth buffer is not used.

### 3.2 Spatial memory (`spatial_memory.py`)

The agent keeps its own estimate of where it is and draws a map as it goes.

- **Heading** is integrated from its own turn presses using the engine's exact keyboard turn
  profile, including the 7 tics at the start of every episode in which input is ignored. Ignoring
  those rotated every map by up to 16 degrees.
- **Position** is integrated from the velocity in the vitals with the engine's exact friction
  model. The engine keeps reporting speed while the player is pushing against a wall, so on wall
  contact the step is re-estimated by fitting the new ray scan onto the previous one.
- **Occupancy grid** at two scales (64 and 256 map units per cell), 9 layers each: visited, free,
  wall, door, locked door, keycard, pickup, edge of unexplored space, exit.
- **Pointers** in the agent's own frame (direction and distance) to the nearest remembered door,
  the locked door of each colour, the nearest keycard and the exit.
- **Learned sightings.** The observation never says where the exit is or what colour a locked door
  is; both are only visible in the picture. A small picture network reports them, and its confident
  sightings are stored in the map with the position they were seen from.

Measured against ground truth on 300 random levels, the position error averages 13 map units, down
from 61 before the corrections above. The memory only records. It never plans a route and never
picks an action.

### 3.3 The network (`model.py`)

Each input has an encoder that matches its shape:

| Input | Encoder |
| --- | --- |
| 64 rays, 9 channels each | circular 1-D CNN, so a wall looks the same at any bearing |
| up to 32 visible entities | per-entity embedding pooled by 4 learned attention queries |
| map crop, 18 layers of 21 by 21 cells, rotated to the heading | 2-D CNN |
| vitals, previous action and memory pointers (77 numbers) | MLP |
| picture at 30 by 40 and at 60 by 80 | two small CNNs (general view, and exit and door colour) |

The five outputs are joined (768 numbers), reduced to 256, and passed through a GRU with 256 units,
which carries state from one decision to the next. Six action heads read the GRU output.

### 3.4 What goes out

Six heads, each sampled from the network's own probabilities at temperature 0.35:

| Head | Choices |
| --- | --- |
| move | forward, none, back |
| strafe | left, none, right |
| turn | left, none, right |
| fire | yes, no |
| use (open doors) | yes, no |
| next weapon | yes, no |

So the agent can move, strafe, turn, shoot, open a door and change gun in the same decision. No rule
adds, blocks or overrides a button.

## 4. How it was trained

Everything in `training/` runs at training time only. Details, reward terms and commands are in
[training/README.md](training/README.md).

1. **Levels.** Our own generator (`mapgen.py`) writes UDMF levels of four kinds: combat arenas,
   multi-room mazes, keycard chains and full gauntlets, with doors, locks, monsters, pickups and all
   gun types. Each level is checked to be finishable before it is used; before that check, 1 in 8
   gauntlets could not be finished at all.
2. **Copy phase** (`distill.py`, 12M steps). The network imitates a scripted teacher that can see
   the level. The teacher's actions are executed less and less often, so the network learns on the
   states it reaches itself. This gives basic competence.
3. **Picture reader** (`pretrain_exit.py`). The picture network is fitted to 68,000 frames labelled
   from ground truth: exit pad in view, its bearing and range, colour of the locked door ahead. On
   unseen levels 89% of its confident exit sightings are real, and it names a door's colour
   correctly 79% of the time and wrongly 0% of the time.
4. **Reward phase** (`train.py`, about 14M steps). Recurrent PPO written for this project, with no
   RL library: asynchronous actors with weights in shared memory, sequence updates through the GRU,
   and a per-batch stop when the policy has moved far enough. The teacher is switched off.
   - The critic also reads ground truth during training (the objective, the true walking distance to
     it, the direction of the route). The policy never does, and the critic is not shipped.
   - Progress is paid per cell of true walking distance gained toward the next objective (keycard,
     then exit) and taken back when lost, so circling earns nothing.
   - New ground pays; walking over old ground, standing, pushing into walls and firing at nothing
     cost. Each of these terms came from a measured failure.
   - The level mix shifts toward whichever level type the policy is failing.
   - The newest policy is scored on 112 held-out levels every few minutes and the best is kept.

## 5. Results

All numbers are from generated levels whose seeds were never used in training. The score is our own
estimate of the hidden scorer (exit largest, then progression, exploration and combat, with tier
weights 1, 1.5, 2 and 3), useful for comparing versions. Exit rates are exact.

**Sets A and B (112 levels each), the numbers in the project brief**

| Version | Score A / B | Gauntlet exits |
| --- | --- | --- |
| Copied from the teacher only | 207 / 234 | 6% |
| Scripted teacher (sees the level) | 284 / 305 | 19% |
| Final agent (submitted) | 325 / 361 | 28% |

| Tier (final agent, sets A and B) | Exit reached |
| --- | --- |
| 1 Combat and survival | 92% |
| 2 Multi-room exploration | 96% |
| 3 Keycard, locked door, exit | 85% |
| 4 Full gauntlet | 28% |

**A larger check: ten more sets (1,120 levels)**

| Tier (final agent) | Exit reached |
| --- | --- |
| 1 Combat and survival | about 84% |
| 2 Multi-room exploration | about 90% |
| 3 Keycard, locked door, exit | about 93% |
| 4 Full gauntlet | about 31% |

Average score on these ten sets: about 329. A single set of 112 levels moves by roughly 20 points
from set to set, which is why versions were only compared across several sets.

**Behaviour** (60 to 72 games each, before and after the movement rewards)

| | Before | Final agent |
| --- | --- | --- |
| Steps spent going nowhere | 23% | 8% |
| Standing still while walking between rooms | 21% | 0% |
| Left-right turn reversals per 100 steps | 16.8 | 7.0 |
| Exit reached once the exit pad has been seen | 95% | 100% |

**Validator on the submitted commit**

| | Plain | Inside the official Docker image |
| --- | --- | --- |
| Time per decision (median) | 4.5 ms | 11.2 ms |
| Peak RAM | 365 MB | 877 MB |
| Result | passed | passed |

## 6. What we tried that did not help

Each of these was tested against the submitted agent on fresh level sets and was not adopted.

| Idea | Outcome |
| --- | --- |
| Depth buffer as a fourth image channel | 196 against 220 without it |
| Intrinsic curiosity bonus (random network distillation) | paid the agent for spinning on the spot |
| Attention memory over past frames instead of the grid map | best score 208, well below the grid map |
| "Finish the level before the exit" objective | 203 against 258: ran out of time or died while clearing |
| 4M more steps of the same reward training on three machines | level with the submitted agent over five sets |
| Stronger survival rewards with a health and armour memory | about 6 points lower over ten sets |
| An ensemble of three trained versions | about 5 points lower over ten sets |
| Sampling temperature 0.2 or 0.5 instead of 0.35 | no difference |
| Action masking of pointless FIRE presses | no difference, slightly lower when masking fire with no enemy in view |

The experiment code for the later ones is on the `train-more` branch.

## 7. Rules checklist

| Rule | Status |
| --- | --- |
| Actions come from trained network parameters | Yes: every button is sampled from the network's output |
| No hand-written rule trees, pathfinding or aim formulas in the agent | None in `submission/` |
| Core architecture and training pipeline are our own | Yes: network, memory, PPO, level generator and benchmark are written for this project |
| Training code is in the repository | `training/` |
| Runs with the official wrapper on unseen maps | Yes |
| No network or file writes at runtime | None |
| Within 40 ms per decision, 8 GB RAM, 2 GB VRAM, 1 GB repository | Yes: about 5 ms, under 1 GB RAM, CPU only, 6 MB |

The scripted teacher, the route knowledge behind the rewards and the critic's ground truth exist
only in `training/`.

## 8. Reproduce the training

```bash
uv sync
uv pip install torch --index-url https://download.pytorch.org/whl/cu128   # training only

uv run python training/distill.py --workers 10 --envs-per-worker 2 --steps 12000000
uv run python training/pretrain_exit.py --episodes 500 --out training/checkpoints/exit_eye.pt
uv run python training/train.py --workers 10 --envs-per-worker 2 --init training/checkpoints/student.pt \
    --load-eye training/checkpoints/exit_eye.pt --epochs 1 --lr 5e-5 --target-kl 0.02 --ent 0.003 \
    --int-coef 0 --stage 2 --critic-warmup 150 --adaptive-mix --ckdir training/checkpoints

uv run python training/evaluate.py --teacher           # the scripted teacher, for reference
uv run python training/diag_slam.py --episodes 300     # position and map error against ground truth
```

A laptop manages about 400 environment steps per second; a 64-core cloud machine about 1,500.
`training/aws/` has the scripts used to launch, monitor and stop such a machine.

## 9. Repository layout

```
submission/   the agent that is evaluated
training/     the full training pipeline, benchmark and viewer (read by reviewers, not by the scorer)
docs/         PROJECT_BRIEF.pdf, and the organisers' contract, rules, submission and evaluation notes
env/          the official ViZDoom wrapper from the starter template (unchanged)
maps/         the template's test map
tests/        the template's tests
docker/       the official image's pinned requirements
Dockerfile    the official image
validate.py   the organisers' validator
```

## 10. Known weaknesses

- Full gauntlets are finished about three times in ten. They are lost to early deaths in large
  fights or to the clock when the exit is never found; once the exit has been seen it is reached
  every time.
- In fights it often stands still while being shot and keeps the pistol when a stronger gun is
  available.
- Aim is about 70% of shots (the scripted teacher: 83%).
- About 36% of its moves are onto ground it has already walked.
- Keycard colour is not read from the picture; the agent learns a key's colour when it picks it up.
- It was trained mostly on generated levels, so hand-made maps that look very different are less
  tested.
