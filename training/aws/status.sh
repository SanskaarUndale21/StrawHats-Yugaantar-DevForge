#!/usr/bin/env bash
# Show the latest training lines from the running instance.
set -euo pipefail
cd "$(dirname "$0")/../.."
source "training/aws/.instance-${NAME:-doom-train}"
ssh -i "$KEYFILE" -o StrictHostKeyChecking=no ubuntu@$IP \
  "uptime; tail -n 2 ~/doom/training/checkpoints/distill.out 2>/dev/null; tail -n 3 ~/doom/training/checkpoints/train.out 2>/dev/null"
