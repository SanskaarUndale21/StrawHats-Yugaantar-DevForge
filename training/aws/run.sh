#!/usr/bin/env bash
# Runs ON the instance inside tmux. Full pipeline: DAgger distillation -> RL fine-tune (resumable).
# Usage: bash training/aws/run.sh [distill|rl|full] [distill_steps] [rl_steps]
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
cd "$HOME/doom"
PHASE="${1:-full}"
DSTEPS="${2:-8000000}"
RSTEPS="${3:-150000000}"
SEED="${4:-1}"
CORES=$(nproc)
# measured on 64 cores: with 50 actors and 12 learner threads the learner was the bottleneck
# (3.6 s per update against 0.1 s waiting for data), so the learner gets a bigger share
WORKERS=$(( CORES > 32 ? CORES - 24 : (CORES > 16 ? CORES - 14 : (CORES > 4 ? CORES - 2 : CORES)) ))
export LEARNER_THREADS=$(( CORES > 32 ? 20 : (CORES > 16 ? 12 : 1) ))
CK=training/checkpoints
echo "cores=$CORES workers=$WORKERS phase=$PHASE"

distill() {
  local extra=""
  if [ -f $CK/distill.pt ]; then extra="--resume";
  elif [ -f $CK/student.pt ]; then extra="--init $CK/student.pt";
  elif [ -f $CK/latest.pt ]; then extra="--init $CK/latest.pt"; fi
  uv run python -u training/distill.py --workers $WORKERS --envs-per-worker 2 --steps $DSTEPS --beta-steps $(( DSTEPS / 2 )) --seed $SEED $extra 2>&1 | tee -a $CK/distill.out
}

rl() {
  local extra=""
  if [ -f $CK/rl_done_init ] && [ -f $CK/latest.pt ]; then extra="--resume";
  elif [ -f $CK/student.pt ]; then extra="--init $CK/student.pt --critic-warmup 150"; touch $CK/rl_done_init;
  elif [ -f $CK/latest.pt ]; then extra="--resume"; touch $CK/rl_done_init; fi
  # a fresh RL phase after distillation must not resume the old pre-distillation checkpoint
  if [[ "$extra" == --init* ]] && [ -f $CK/latest.pt ]; then mv $CK/latest.pt $CK/pre_distill_latest.pt; fi
  # First attempt (lr 1e-4, entropy 0.01, 30 warm-up updates) made the student worse: fixed benchmark
  # 245 -> 160-200 within 2.5M steps while the entropy rose. So: the value head gets 150 updates
  # (about 1M steps) before the policy moves at all, and the policy then moves slowly with almost
  # no entropy bonus and a strong teacher-label term.
  # fine-tuning a trained student: small learning rate, bounded policy change per batch, the teacher
  # still labelling the states the policy itself reaches (also after a resume), and a
  # faster march through the curriculum stages (each is forced after 6M steps at the latest)
  uv run python -u training/train.py --workers $WORKERS --envs-per-worker 2 --steps $RSTEPS --seed $SEED \
    --lr 5e-5 --target-kl 0.02 --ent 0.003 --stage-steps 6000000 --ckdir $CK --bc-coef 1.0 --bc-decay 40000000 $extra 2>&1 | tee -a $CK/train.out
}

case "$PHASE" in
  distill) distill ;;
  rl) rl ;;
  full) distill; pkill -f "[v]izdoom" 2>/dev/null; sleep 2; rl ;;
esac
