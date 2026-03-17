#!/usr/bin/env bash
# Sequential model sweep: 6 runs on A100, pre-built dataset
set -euo pipefail

DATASET="artifacts/dataset_65sym_60seq.npz"
BATCH_SIZE=$((4096*32))
DATE=$(date +%m%d)
GROUP="cuda_e20_8M_amp_sweep_${DATE}"
LOGDIR="logs/${GROUP}"
COMMON="--load-dataset $DATASET --seq-len 60 --epochs 20 --batch-size ${BATCH_SIZE} --device cuda --amp --wandb --wandb-tags sweep,test --wandb-group $GROUP"
mkdir -p "$LOGDIR"

echo "=== Sequential Sweep: $GROUP ==="
echo "Dataset: $DATASET"
echo "Logs: $LOGDIR/"
echo ""

run_one() {
  local name="$1"; shift
  echo ">>> $name launching..."
  python -m models.train $COMMON --wandb-name "$name" "$@" 2>&1 | tee "$LOGDIR/${name}.log"
  echo ""
}

source .venv/bin/activate
run_one "lstm_h64"   --model-type lstm --hidden-size 64  --num-layers 2


echo "=== Sweep complete: $GROUP ==="
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
