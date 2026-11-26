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
WORKERS=$(( CORES > 16 ? CORES - 14 : (CORES > 4 ? CORES - 2 : CORES) ))
export LEARNER_THREADS=$(( CORES > 16 ? 12 : 1 ))
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
  elif [ -f $CK/student.pt ]; then extra="--init $CK/student.pt --bc-coef 0.5 --critic-warmup 30"; touch $CK/rl_done_init;
  elif [ -f $CK/latest.pt ]; then extra="--resume"; touch $CK/rl_done_init; fi
  # a fresh RL phase after distillation must not resume the old pre-distillation checkpoint
  if [[ "$extra" == --init* ]] && [ -f $CK/latest.pt ]; then mv $CK/latest.pt $CK/pre_distill_latest.pt; fi
  # fine-tuning a trained student: small learning rate, bounded policy change per batch, and a
  # faster march through the curriculum stages (each is forced after 6M steps at the latest)
  uv run python -u training/train.py --workers $WORKERS --envs-per-worker 2 --steps $RSTEPS --seed $SEED \
    --lr 1e-4 --target-kl 0.03 --stage-steps 6000000 --ckdir $CK $extra 2>&1 | tee -a $CK/train.out
}

case "$PHASE" in
  distill) distill ;;
  rl) rl ;;
  full) distill; pkill -f "[v]izdoom" 2>/dev/null; sleep 2; rl ;;
esac
