#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"
DEVICE="${DEVICE:-cuda}"
GPU_LIST="${CUDA_VISIBLE_DEVICES:-0}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs}"
SAMPLE_REPEATS="${SAMPLE_REPEATS:-20}"
INSTANCES="${INSTANCES:-80}"
OFFSET="${OFFSET:-20}"
mkdir -p "$OUTPUT_ROOT/logs"

for TRAIN_SEED in 20260805 20260807 20260808; do
  SOURCE_DIR="$OUTPUT_ROOT/main_seed_${TRAIN_SEED}"
  [[ -f "$SOURCE_DIR/final_model.pt" ]] || { echo "missing source checkpoint: $SOURCE_DIR/final_model.pt" >&2; exit 1; }
  for EVAL_SEED in 20260825 20260826 20260827; do
    OUTPUT_DIR="$OUTPUT_ROOT/nominal_train${TRAIN_SEED}_eval_${EVAL_SEED}"
    LOG_FILE="$OUTPUT_ROOT/logs/nominal_train${TRAIN_SEED}_eval_${EVAL_SEED}.log"
    if [[ -f "$OUTPUT_DIR/summary.json" ]] && grep -q '"passed": true' "$OUTPUT_DIR/summary.json"; then
      echo "skip completed evaluation: $OUTPUT_DIR"
      continue
    fi
    if [[ -e "$OUTPUT_DIR" ]]; then
      echo "incomplete output exists; remove or rename it before rerunning: $OUTPUT_DIR" >&2
      exit 1
    fi
    local_instances="$INSTANCES"
    local_offset="$OFFSET"
    local_repeats="$SAMPLE_REPEATS"
    if [[ "${SMOKE:-0}" == "1" ]]; then
      local_instances=4
      local_offset=0
      local_repeats=2
    fi
    CUDA_VISIBLE_DEVICES="$GPU_LIST" "$PYTHON_BIN" -u evaluate_constrained_pilot.py \
      --device "$DEVICE" \
      --source-dir "$SOURCE_DIR" \
      --checkpoint checkpoints/base/reference_save_10_5.pt \
      --data-dir data_dev/1005 \
      --offset "$local_offset" \
      --instances "$local_instances" \
      --initial-jobs 6 \
      --mean-interarrival 8 \
      --mean-time-between-failures 8 \
      --mean-repair-time 2 \
      --events-per-machine 1 \
      --stability-budget 1.7 \
      --seed "$EVAL_SEED" \
      --sample-repeats "$local_repeats" \
      --stability-context-mode full \
      --selection-mode budget_first \
      --output-dir "$OUTPUT_DIR" 2>&1 | tee "$LOG_FILE"
  done
done

if [[ "${SMOKE:-0}" == "1" ]]; then
  echo "PASS nominal holdout smoke"
  exit 0
fi

SUMMARY_DIR="$OUTPUT_ROOT/nominal_summary"
if [[ -e "$SUMMARY_DIR" ]]; then
  echo "summary already exists: $SUMMARY_DIR" >&2
  exit 1
fi
"$PYTHON_BIN" -u summarize_finalenv_main_holdout.py \
  --save-dir "$OUTPUT_ROOT" \
  --directory-prefix nominal \
  --output-dir "$SUMMARY_DIR"
echo "PASS nominal holdout reproduction: $SUMMARY_DIR"
