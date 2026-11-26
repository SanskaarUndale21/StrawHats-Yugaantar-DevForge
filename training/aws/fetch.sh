#!/usr/bin/env bash
# Copy checkpoints, logs and the exported policy back from the instance.
set -euo pipefail
cd "$(dirname "$0")/../.."
source "training/aws/.instance-${NAME:-doom-train}"
OUT="training/checkpoints/aws/${NAME:-doom-train}"
mkdir -p "$OUT"
scp -i "$KEYFILE" -o StrictHostKeyChecking=no \
  "ubuntu@$IP:~/doom/training/checkpoints/{latest.pt,student.pt,distill.pt,log.csv,distill_log.csv,train.out,distill.out}" \
  "$OUT"/ 2>/dev/null || true
scp -i "$KEYFILE" -o StrictHostKeyChecking=no "ubuntu@$IP:~/doom/submission/weights/doomnet.pt" "$OUT"/doomnet.pt 2>/dev/null || true
ls -la "$OUT"/
echo "to use the cloud weights:  cp $OUT/doomnet.pt submission/weights/doomnet.pt"
