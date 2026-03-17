#!/usr/bin/env bash
# Parallel model sweep: 6 runs on single A100, pre-built dataset
set -uo pipefail

DATASET="artifacts/dataset_65sym_60seq.npz"
DATE=$(date +%m%d)
GROUP="cuda_e50_8M_amp_sweep_${DATE}"
LOGDIR="logs/sweep_${DATE}"
COMMON="--load-dataset $DATASET --seq-len 60 --epochs 50 --batch-size 256 --device cuda --amp --wandb --wandb-tags sweep,test --wandb-group $GROUP"
mkdir -p "$LOGDIR"

echo "=== Parallel Sweep: $GROUP ==="
echo "Dataset: $DATASET"
echo "Logs: $LOGDIR/"
echo ""

# ── Transformer variants ──
echo "[1/6] tf_4L_d64 launching..."
python -m models.train $COMMON \
  --model-type transformer --num-layers 4 --d-model 64 --nhead 4 --d-ff 256 \
  --wandb-name "tf_4L_d64_$(date +%H%M%S)" \
  > "$LOGDIR/tf_4L_d64.log" 2>&1 &

echo "[2/6] tf_6L_d64 launching..."
python -m models.train $COMMON \
  --model-type transformer --num-layers 6 --d-model 64 --nhead 4 --d-ff 256 \
  --wandb-name "tf_6L_d64_$(date +%H%M%S)" \
  > "$LOGDIR/tf_6L_d64.log" 2>&1 &

echo "[3/6] tf_8L_d64 launching..."
python -m models.train $COMMON \
  --model-type transformer --num-layers 8 --d-model 64 --nhead 4 --d-ff 256 \
  --wandb-name "tf_8L_d64_$(date +%H%M%S)" \
  > "$LOGDIR/tf_8L_d64.log" 2>&1 &

# ── LSTM variants ──
echo "[4/6] lstm_h64_2L launching..."
python -m models.train $COMMON \
  --model-type lstm --hidden-size 64 --num-layers 2 \
  --wandb-name "lstm_h64_2L_$(date +%H%M%S)" \
  > "$LOGDIR/lstm_h64_2L.log" 2>&1 &

echo "[5/6] lstm_h128_2L launching..."
python -m models.train $COMMON \
  --model-type lstm --hidden-size 128 --num-layers 2 \
  --wandb-name "lstm_h128_2L_$(date +%H%M%S)" \
  > "$LOGDIR/lstm_h128_2L.log" 2>&1 &

echo "[6/6] lstm_h256_2L launching..."
python -m models.train $COMMON \
  --model-type lstm --hidden-size 256 --num-layers 2 \
  --wandb-name "lstm_h256_2L_$(date +%H%M%S)" \
  > "$LOGDIR/lstm_h256_2L.log" 2>&1 &

echo ""
echo "All 6 launched. Waiting..."
echo "Monitor: watch nvidia-smi"
echo "Logs: tail -f $LOGDIR/*.log"
echo ""

wait
echo ""
echo "=== All runs complete ==="
echo "Results: https://wandb.ai/Base-Work-Space/trading-lstm"

echo ""
echo "=== Summary ==="
for f in "$LOGDIR"/*.log; do
  name=$(basename "$f" .log)
  params=$(grep "Model parameters" "$f" 2>/dev/null | tail -1 | grep -oP '\d+$')
  best=$(grep "Best val_loss" "$f" 2>/dev/null | tail -1 | grep -oP 'val_loss=\S+')
  total=$(grep "Training complete" "$f" 2>/dev/null | tail -1 | grep -oP '[\d.]+s' | head -1)
  echo "  $name | params=$params | $best | time=$total"
done
