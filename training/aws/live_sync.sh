#!/usr/bin/env bash
# Runs on the LAPTOP: keep training/checkpoints/live_student.pt equal to the newest policy on the
# cloud box, so `python training/watch.py --live` always plays what is being trained right now.
#   NAME=big bash training/aws/live_sync.sh
set -uo pipefail
cd "$(dirname "$0")/../.."
DST=training/checkpoints/live_student.pt
while true; do
  source "training/aws/.instance-${NAME:-doom-train}"   # re-read: a relaunched box has a new address
  NEWEST=$(ssh -i "$KEYFILE" -o StrictHostKeyChecking=no -o ConnectTimeout=15 ubuntu@$IP            'cd ~/doom/training/checkpoints && ls -t student.pt doomnet.pt 2>/dev/null | head -1' 2>/dev/null)
  if [ -n "$NEWEST" ]; then
    scp -q -i "$KEYFILE" -o StrictHostKeyChecking=no "ubuntu@$IP:doom/training/checkpoints/$NEWEST" "$DST.part" 2>/dev/null       && mv -f "$DST.part" "$DST"
  fi
  # the best policy found so far (by the benchmark on the box) is backed up too: a reclaimed spot
  # box then costs nothing but time
  scp -q -i "$KEYFILE" -o StrictHostKeyChecking=no "ubuntu@$IP:doom/training/checkpoints/best.pt" "${DST%live_student.pt}best_${NAME:-box}.pt.part" 2>/dev/null \
    && mv -f "${DST%live_student.pt}best_${NAME:-box}.pt.part" "${DST%live_student.pt}best_${NAME:-box}.pt"
  scp -q -i "$KEYFILE" -o StrictHostKeyChecking=no "ubuntu@$IP:doom/training/checkpoints/soak.log" "${DST%live_student.pt}soak_${NAME:-box}.log" 2>/dev/null
  sleep "${EVERY:-90}"
done
