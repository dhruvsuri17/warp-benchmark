#!/usr/bin/env bash
# GPU training launcher with tmux + logging
#
# Usage:
#   bash scripts/train_gpu.sh warp_case118 experiments/configs/warp_case118.yaml 0

set -euo pipefail

SESSION_NAME="${1:-warp_train}"
CONFIG="${2:-experiments/configs/warp_case118.yaml}"
GPU="${3:-0}"

LOG_DIR="results/logs"
mkdir -p "$LOG_DIR"

LOG_FILE="$LOG_DIR/${SESSION_NAME}.log"

echo "Starting training session: $SESSION_NAME"
echo "Config: $CONFIG"
echo "GPU: $GPU"
echo "Log: $LOG_FILE"

tmux new-session -d -s "$SESSION_NAME" \
    "python training/train_warp.py --config $CONFIG --gpu $GPU 2>&1 | tee $LOG_FILE"

echo "Training started in tmux session '$SESSION_NAME'"
echo "Attach with: tmux attach -t $SESSION_NAME"
echo "Detach with: Ctrl+B D"
echo "Check progress: tail -f $LOG_FILE"
