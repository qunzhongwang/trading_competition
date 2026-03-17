#!/usr/bin/env bash
# Model architecture sweep: 3 Transformer sizes + 3 LSTM sizes
# Group: sweep_0317, tagged with model size and timestamp
set -euo pipefail

PARQUET_DIR="/home/qw3460/wp/huggingface/Wrigggy/crypto-ohlcv-1m/data"
COMMON="--parquet-dir $PARQUET_DIR --seq-len 60 --epochs 50 --batch-size 256 --device cuda --amp --compile --wandb --wandb-tags sweep,test"
DATE=$(date +%m%d)
GROUP="sweep_${DATE}"

echo "=== Model Sweep: $GROUP ==="
echo "Data: $PARQUET_DIR (65 symbols, 8.4M samples)"
echo ""

# ── Transformer variants ──
echo ">>> [1/6] Transformer 4L d64 4h (~203k params)"
python -m models.train $COMMON \
  --model-type transformer --num-layers 4 --d-model 64 --nhead 4 --d-ff 256 \
  --wandb-group "$GROUP" --wandb-name "tf_4L_d64_$(date +%H%M%S)"

echo ">>> [2/6] Transformer 6L d64 4h (~280k params)"
python -m models.train $COMMON \
  --model-type transformer --num-layers 6 --d-model 64 --nhead 4 --d-ff 256 \
  --wandb-group "$GROUP" --wandb-name "tf_6L_d64_$(date +%H%M%S)"

echo ">>> [3/6] Transformer 8L d64 4h (~360k params)"
python -m models.train $COMMON \
  --model-type transformer --num-layers 8 --d-model 64 --nhead 4 --d-ff 256 \
  --wandb-group "$GROUP" --wandb-name "tf_8L_d64_$(date +%H%M%S)"

# ── LSTM variants ──
echo ">>> [4/6] LSTM h64 2L (~40k params)"
python -m models.train $COMMON \
  --model-type lstm --hidden-size 64 --num-layers 2 \
  --wandb-group "$GROUP" --wandb-name "lstm_h64_2L_$(date +%H%M%S)"

echo ">>> [5/6] LSTM h128 2L (~147k params)"
python -m models.train $COMMON \
  --model-type lstm --hidden-size 128 --num-layers 2 \
  --wandb-group "$GROUP" --wandb-name "lstm_h128_2L_$(date +%H%M%S)"

echo ">>> [6/6] LSTM h256 2L (~530k params)"
python -m models.train $COMMON \
  --model-type lstm --hidden-size 256 --num-layers 2 \
  --wandb-group "$GROUP" --wandb-name "lstm_h256_2L_$(date +%H%M%S)"

echo ""
echo "=== Sweep complete: $GROUP ==="
echo "View results: https://wandb.ai/Base-Work-Space/trading-lstm"
