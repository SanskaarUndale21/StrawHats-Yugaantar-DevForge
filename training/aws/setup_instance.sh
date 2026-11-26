#!/usr/bin/env bash
# Runs ON the EC2 instance (Ubuntu 24.04). Installs everything and checks the environment.
# Usage: bash setup_instance.sh [repo_url]
set -euo pipefail
REPO_URL="${1:-https://github.com/SanskaarUndale21/StrawHats-Yugaantar-DevForge.git}"

sudo apt-get update -qq
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq git git-lfs tmux htop libgl1 libglib2.0-0 >/dev/null
git lfs install --skip-repo >/dev/null

if ! command -v uv >/dev/null; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="$HOME/.local/bin:$PATH"

cd "$HOME"
if [ ! -d doom ]; then
  git clone -q "$REPO_URL" doom
fi
cd doom
git pull -q || true
uv sync -q
if command -v nvidia-smi >/dev/null && nvidia-smi >/dev/null 2>&1; then
  uv pip install -q torch --index-url https://download.pytorch.org/whl/cu128
else
  uv pip install -q torch --index-url https://download.pytorch.org/whl/cpu
fi
mkdir -p training/checkpoints
uv run python -c "import torch, vizdoom; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), 'cores', __import__('os').cpu_count())"
echo "setup done"
