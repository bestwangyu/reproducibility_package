#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"
DEVICE="${DEVICE:-cuda}"
GPU_LIST="${CUDA_VISIBLE_DEVICES:-0}"
ITERATIONS="${ITERATIONS:-100}"
INSTANCES="${INSTANCES:-20}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs}"
mkdir -p "$OUTPUT_ROOT/logs"

for TRAIN_SEED in 20260805 20260807 20260808; do
  OUTPUT_DIR="$OUTPUT_ROOT/main_seed_${TRAIN_SEED}"
  LOG_FILE="$OUTPUT_ROOT/logs/main_seed_${TRAIN_SEED}.log"
  if [[ -f "$OUTPUT_DIR/summary.json" ]] && grep -q '"passed": true' "$OUTPUT_DIR/summary.json"; then
    echo "skip completed training: $OUTPUT_DIR"
    continue
  fi
  if [[ -e "$OUTPUT_DIR" ]]; then
    echo "incomplete output exists; remove or rename it before rerunning: $OUTPUT_DIR" >&2
    exit 1
  fi
  BATCH_SIZE="$INSTANCES"
  if [[ "${SMOKE:-0}" == "1" ]]; then
    ITERATIONS=2
    INSTANCES=4
    BATCH_SIZE=4
  fi
  echo "training seed=${TRAIN_SEED} device=${DEVICE} visible_gpu=${GPU_LIST}"
  CUDA_VISIBLE_DEVICES="$GPU_LIST" "$PYTHON_BIN" -u train_constrained_dynamic.py \
    --device "$DEVICE" \
    --checkpoint checkpoints/base/reference_save_10_5.pt \
    --data-dir data_dev/1005 \
    --instances "$INSTANCES" \
    --batch-size "$BATCH_SIZE" \
    --iterations "$ITERATIONS" \
    --initial-jobs 6 \
    --mean-interarrival 8 \
    --mean-time-between-failures 8 \
    --mean-repair-time 2 \
    --events-per-machine 1 \
    --stability-budget 1.7 \
    --lr 0.0001 \
    --reward-coeff 0.0 \
    --cost-advantage-mode batch_centered \
    --cost-signal raw \
    --stability-context-mode full \
    --lambda-init 1.0 \
    --lambda-lr 0.0 \
    --lambda-max 1.0 \
    --train-seed "$TRAIN_SEED" \
    --scenario-seed "$((TRAIN_SEED + 10))" \
    --output-dir "$OUTPUT_DIR" \
    --smoke-check 2>&1 | tee "$LOG_FILE"
  grep -q '"passed": true' "$OUTPUT_DIR/summary.json"
done

echo "PASS main training reproduction"
