#!/usr/bin/env bash
# Runs on the LAPTOP: keep training/checkpoints/live_student.pt equal to the newest policy on the
# cloud box, so `python training/watch.py --live` always plays what is being trained right now.
#   NAME=big bash training/aws/live_sync.sh
set -uo pipefail
cd "$(dirname "$0")/../.."
DST=training/checkpoints/live_student.pt
while true; do
  source "training/aws/.instance-${NAME:-doom-train}"   # re-read: a relaunched box has a new address
  NEWEST=$(ssh -i "$KEYFILE" -o StrictHostKeyChecking=no -o ConnectTimeout=15 ubuntu@$IP            'cd ~/doom/training/checkpoints && ls -t student.pt latest.pt 2>/dev/null | head -1' 2>/dev/null)
  if [ -n "$NEWEST" ]; then
    scp -q -i "$KEYFILE" -o StrictHostKeyChecking=no "ubuntu@$IP:doom/training/checkpoints/$NEWEST" "$DST.part" 2>/dev/null       && mv -f "$DST.part" "$DST"
  fi
  sleep "${EVERY:-90}"
done
