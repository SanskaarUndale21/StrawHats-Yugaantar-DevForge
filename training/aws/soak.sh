#!/usr/bin/env bash
# Runs ON the instance: benchmark the newest student on fresh random levels, forever.
# One line per round is appended to training/checkpoints/soak.log.
#   tmux new -d -s soak "bash training/aws/soak.sh"
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
cd "$HOME/doom"
CK=training/checkpoints
PROCS="${1:-8}"
while true; do
  if [ ! -f $CK/student.pt ]; then sleep 60; continue; fi
  cp $CK/student.pt /tmp/soak_student.pt
  SEED=$(( (RANDOM << 15) | RANDOM ))
  OUT=$(uv run python training/evaluate.py --weights /tmp/soak_student.pt --per-tier 4 --freedoom 2 --seeds 1         --procs "$PROCS" --seed-base "$SEED" 2>&1)
  STEP=$(tail -n 1 $CK/train.out 2>/dev/null | grep -o "step=[0-9]*" || tail -n 1 $CK/distill.out | grep -o "step=[0-9]*")
  echo "$(date -u +%H:%M) seed=$SEED $STEP | $(echo "$OUT" | grep weighted | sed 's/.*weighted //') | $(echo "$OUT" | grep '  all ' | sed 's/  all  *//')" >> $CK/soak.log
  echo "$OUT" | grep -E "tier [0-9]" | sed "s/^/    /" >> $CK/soak.log
done
