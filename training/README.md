# Training pipeline (StrawHats)

Everything here runs at training time only. The scorer loads `submission/` and nothing else.

```
mapgen.py          procedural UDMF level generator: combat / explore / keys / gauntlet, every level checked to be finishable
vec_env.py         one environment slot: fresh levels, SpatialMemory features, shaped reward, side-task targets
sampler.py         asynchronous actor pool (each actor owns envs and a policy copy; weights via shared memory)
rewards.py         reward shaping, the route knowledge behind it, and the critic's side information
teacher.py         scripted teacher that sees the level (kick-start only, never in the submission)
distill.py         phase 1: the network copies the teacher (DAgger)
pretrain_exit.py   supervised pre-training of the picture reader (exit pad, locked-door colour)
train.py           phase 2: recurrent PPO from rewards, with a critic that reads ground truth
evaluate.py        benchmark on held-out generated levels, four tiers, per-skill statistics
metrics.py         aim, kill share, rooms seen and cleared, gun swaps, movement quality
diag_slam.py       measures the agent's own position and map error against ground truth
watch.py           live viewer: the game, the occupancy map the agent builds, and its statistics
aws/               launch, run, soak-test and sync scripts for a many-core cloud box
```

## How the agent was trained

1. **Copy phase** (`distill.py`, 12M steps). The network imitates the scripted teacher on generated
   levels. The teacher's actions are executed less and less often, so the network learns on the states
   it reaches itself. This gives basic competence: 207 and 234 on our two final test sets.
2. **Picture reader** (`pretrain_exit.py`). The observation never says where the exit is or what
   colour a locked door is; both are only visible in the picture. A small CNN is fitted to frames
   labelled from ground truth. On unseen levels: 89% of its confident exit sightings are real, and
   it names a locked door's colour correctly 79% of the time and wrongly 0% of the time.
3. **Reward phase** (`train.py`, about 14M steps). Recurrent PPO written for this project, no RL
   library. The policy continues from the copy phase with the teacher switched off.
   * The critic also reads ground truth (the objective, the true walking distance to it, which way
     the route goes). The policy never does, and the critic is not shipped.
   * Progress is paid per cell of true walking distance gained toward the next objective (keycard,
     then exit) and taken back when lost, so circling earns exactly nothing.
   * The level mix shifts toward whichever level type the policy is currently failing.
   * The newest policy is scored on 112 held-out levels every few minutes and the best one is kept.

## Reward terms (`rewards.py`)

| Group | Terms |
| --- | --- |
| Scorer-aligned | exit (largest, plus health and time left), keycard, locked door opened, door opened, new 128-unit area, kill, damage taken, early death |
| Route | progress toward the objective by true walking distance |
| Combat | shot that hits, shot that misses, damage dealt, new gun picked up, health picked up while hurt |
| Movement quality | going nowhere (under 64 units net in 20 steps), reversing a turn or forward/back, re-walking a visited cell, pushing into a wall, USE with no door ahead, firing that lands nothing for 6 steps, cycling guns |

The movement terms each came from a measured failure, and the comments in `rewards.py` give the
numbers. Two lessons are recorded there: a penalty must be sized against the episode return or it
changes nothing, and each term must be checked for the cheapest way around it (standing still was
once cheaper than walking over old ground; walking in a tight circle slipped past four terms).

## Reproduce

```bash
uv sync
uv pip install torch --index-url https://download.pytorch.org/whl/cu128   # training only (the image has torch)

uv run python training/distill.py --workers 10 --envs-per-worker 2 --steps 12000000
uv run python training/pretrain_exit.py --episodes 500 --out training/checkpoints/exit_eye.pt
uv run python training/train.py --workers 10 --envs-per-worker 2 --init training/checkpoints/student.pt \
    --load-eye training/checkpoints/exit_eye.pt --epochs 1 --lr 5e-5 --target-kl 0.02 --ent 0.003 \
    --int-coef 0 --stage 2 --critic-warmup 150 --adaptive-mix --ckdir training/checkpoints

uv run python training/evaluate.py --weights submission/weights/doomnet.pt --seeds 4 --seed-base 6161
uv run python training/evaluate.py --teacher                    # the scripted teacher, for reference
uv run python training/diag_slam.py --episodes 300              # position and map error
uv run python training/watch.py --tour                          # watch it play one level of each tier
```

On a 64-core cloud box, `bash training/aws/launch.sh` runs the same pipeline (see `aws/README.md`);
the laptop manages about 400 steps per second and the box about 1,400 to 3,700.

**Windows note:** ViZDoom runs about 17 times faster after `timeBeginPeriod(1)` (with the default
15.6 ms timer every engine sync sleeps). `vec_env._hires_timer()` does this in every worker.

## Benchmark (`evaluate.py`)

Levels are generated from seeds never used in training, in the four tiers of the hidden suite, plus
the template's test map and three Freedoom 2 campaign maps. The score mirrors the scoring categories
the organisers describe (exit largest, then progression, exploration, combat) with tier weights 1,
1.5, 2 and 3; the real weights are hidden, so the number is our estimate, useful for comparing
versions and nothing more. One run of 112 levels still moves by about 20 points between level sets,
so every promotion to `main` was decided on at least two fresh sets.
