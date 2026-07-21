#!/usr/bin/env bash
# Train the 5M cross-layer transcoder (single GPU).
# Usage:  DATA=/path/to/lichess_db_standard_rated_YYYY-MM.pgn  bash scripts/train_5m.sh
set -euo pipefail
cd "$(dirname "$0")/.."
DATA="${DATA:?set DATA=/path/to/lichess.pgn (.pgn or .pgn.zst)}"
OUT="${OUT:-runs/clt-5m}"
python train.py --model 5m --data "$DATA" --out "$OUT" \
  --activation jumprelu --expansion 8 \
  --sparsity-lambda 1e-3 --tanh-c 1 --lambda-warmup 2000 \
  --lambda-anneal-start 5000 --lambda-anneal-len 5000 \
  --steps 20000 --ckpt-interval 2000 \
  --train-batch 8192 --buffer-size 500000 --capture-batch 256 \
  --pool-device cuda "$@"
