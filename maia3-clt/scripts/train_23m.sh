#!/usr/bin/env bash
# Train the 23M cross-layer transcoder on 2 GPUs (DDP).
# Usage:  DATA=/path/to/lichess.pgn  CUDA_VISIBLE_DEVICES=0,1  bash scripts/train_23m.sh
set -euo pipefail
cd "$(dirname "$0")/.."
DATA="${DATA:?set DATA=/path/to/lichess.pgn (.pgn or .pgn.zst)}"
OUT="${OUT:-runs/clt-23m}"
torchrun --nproc_per_node=2 --master_port=29501 \
  train.py --model 23m --data "$DATA" --out "$OUT" \
  --activation jumprelu --expansion 8 \
  --sparsity-lambda 1e-3 --tanh-c 1 --lambda-warmup 2000 \
  --lambda-anneal-start 3000 --lambda-anneal-len 4000 \
  --steps 20000 --ckpt-interval 2000 \
  --train-batch 8192 --buffer-size 500000 --capture-batch 256 \
  --pool-device cuda "$@"
