#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"
DEVICE="${DEVICE:-cuda}"
GPU_LIST="${CUDA_VISIBLE_DEVICES:-0}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs}"
mkdir -p "$OUTPUT_ROOT/logs"
INSTANCES=80
OFFSET=20
if [[ "${SMOKE:-0}" == "1" ]]; then
  INSTANCES=4
  OFFSET=0
fi

for TRAIN_SEED in 20260805 20260807 20260808; do
  POLICY="checkpoints/static_drl/static_drl_seed_${TRAIN_SEED}.pt"
  [[ -f "$POLICY" ]] || { echo "missing static checkpoint: $POLICY" >&2; exit 1; }
  for EVAL_SEED in 20260825 20260826 20260827; do
    OUTPUT_DIR="$OUTPUT_ROOT/finalenv_static_drl_zero_shot_train${TRAIN_SEED}_eval_${EVAL_SEED}"
    if [[ ! -e "$OUTPUT_DIR" ]]; then
      CUDA_VISIBLE_DEVICES="$GPU_LIST" "$PYTHON_BIN" -u evaluate_static_dynamic_baseline.py \
        --device "$DEVICE" \
        --policy-checkpoint "$POLICY" \
        --reference-checkpoint checkpoints/base/reference_save_10_5.pt \
        --policy-seed "$TRAIN_SEED" \
        --data-dir data_dev/1005 --offset "$OFFSET" --instances "$INSTANCES" \
        --initial-jobs 6 --mean-interarrival 8 \
        --mean-time-between-failures 8 --mean-repair-time 2 \
        --events-per-machine 1 --stability-budget 1.7 \
        --seed "$EVAL_SEED" --output-dir "$OUTPUT_DIR" \
        2>&1 | tee "$OUTPUT_ROOT/logs/static_train${TRAIN_SEED}_eval_${EVAL_SEED}.log"
    else
      echo "skip existing baseline output: $OUTPUT_DIR"
    fi
  done
done

for EVAL_SEED in 20260825 20260826 20260827; do
  OUTPUT_DIR="$OUTPUT_ROOT/finalenv_dispatch_rules_eval_${EVAL_SEED}"
  if [[ ! -e "$OUTPUT_DIR" ]]; then
    CUDA_VISIBLE_DEVICES="$GPU_LIST" "$PYTHON_BIN" -u evaluate_dynamic_dispatching_rules.py \
      --device "$DEVICE" \
      --reference-checkpoint checkpoints/base/reference_save_10_5.pt \
      --data-dir data_dev/1005 --offset "$OFFSET" --instances "$INSTANCES" \
      --initial-jobs 6 --mean-interarrival 8 \
      --mean-time-between-failures 8 --mean-repair-time 2 \
      --events-per-machine 1 --stability-budget 1.7 \
      --seed "$EVAL_SEED" --output-dir "$OUTPUT_DIR" \
      2>&1 | tee "$OUTPUT_ROOT/logs/rules_eval_${EVAL_SEED}.log"
  else
    echo "skip existing rule output: $OUTPUT_DIR"
  fi
done

if [[ "${SMOKE:-0}" == "1" ]]; then
  echo "PASS fair-baseline smoke"
  exit 0
fi

SUMMARY_DIR="$OUTPUT_ROOT/fair_baselines_summary"
if [[ -e "$SUMMARY_DIR" ]]; then
  echo "summary already exists: $SUMMARY_DIR" >&2
  exit 1
fi
"$PYTHON_BIN" -u summarize_finalenv_fair_baselines.py \
  --save-dir "$OUTPUT_ROOT" --output-dir "$SUMMARY_DIR"
echo "PASS fair-baseline reproduction: $SUMMARY_DIR"
