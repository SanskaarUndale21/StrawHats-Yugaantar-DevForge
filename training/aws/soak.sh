#!/usr/bin/env bash
# Runs ON the instance, forever. Each round takes the newest policy (distilled student, or the RL
# policy once that phase runs) and
#   1. scores it on the fixed benchmark; the best one so far is kept as checkpoints/best.pt
#   2. plays fresh random levels with it (generalisation check), one line per round in soak.log
#   tmux new -d -s soak "bash training/aws/soak.sh"
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
cd "$HOME/doom"
CK=training/checkpoints
PROCS="${1:-8}"
score() { echo "$1" | grep weighted | sed 's/.*weighted score \([0-9.-]*\).*/\1/' | tail -n 1; }
while true; do
  SRC=$(ls -t $CK/doomnet.pt $CK/student.pt 2>/dev/null | head -n 1)
  if [ -z "$SRC" ]; then sleep 60; continue; fi
  cp "$SRC" /tmp/soak_policy.pt
  STEP=$(tail -n 1 $CK/train.out 2>/dev/null | grep -o "step=[0-9]*" || tail -n 1 $CK/distill.out | grep -o "step=[0-9]*")
  PHASE=$([ "$SRC" = "$CK/doomnet.pt" ] && echo rl || echo distill)

  OUT=$(uv run python training/evaluate.py --weights /tmp/soak_policy.pt --procs "$PROCS" --seeds 4 --seed-base 4242 2>&1)
  S=$(score "$OUT")
  BEST=$(cat $CK/best.score 2>/dev/null || echo -1e9)
  if [ -n "$S" ] && awk -v s="$S" -v b="$BEST" 'BEGIN{exit !(s>b)}'; then
    cp /tmp/soak_policy.pt $CK/best.pt.part && mv -f $CK/best.pt.part $CK/best.pt
    echo "$S" > $CK/best.score
    MARK="NEW BEST"
  else
    MARK="best $BEST"
  fi
  echo "$(date -u +%H:%M) $PHASE $STEP | 112-level test $S ($MARK) | $(echo "$OUT" | grep '  all ' | sed 's/  all  *//')" >> $CK/soak.log
  echo "$OUT" | grep -E "tier [0-9]" | sed "s/^/    /" >> $CK/soak.log

  SEED=$(( (RANDOM << 15) | RANDOM ))
  OUT=$(uv run python training/evaluate.py --weights /tmp/soak_policy.pt --per-tier 4 --freedoom 2 --seeds 1 \
        --procs "$PROCS" --seed-base "$SEED" 2>&1)
  echo "$(date -u +%H:%M) $PHASE $STEP | random seed $SEED score $(score "$OUT") | $(echo "$OUT" | grep '  all ' | sed 's/  all  *//')" >> $CK/soak.log
done
