#!/usr/bin/env bash
# Copy checkpoints, logs and the exported policy back from the instance.
set -euo pipefail
cd "$(dirname "$0")/../.."
source training/aws/.instance
mkdir -p training/checkpoints/aws
scp -i "$KEYFILE" -o StrictHostKeyChecking=no \
  "ubuntu@$IP:~/doom/training/checkpoints/{latest.pt,student.pt,distill.pt,log.csv,distill_log.csv,train.out,distill.out}" \
  training/checkpoints/aws/ 2>/dev/null || true
scp -i "$KEYFILE" -o StrictHostKeyChecking=no "ubuntu@$IP:~/doom/submission/weights/doomnet.pt" training/checkpoints/aws/doomnet.pt 2>/dev/null || true
ls -la training/checkpoints/aws/
echo "to use the cloud weights:  cp training/checkpoints/aws/doomnet.pt submission/weights/doomnet.pt"
