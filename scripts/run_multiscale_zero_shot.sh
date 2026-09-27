#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"
DEVICE="${DEVICE:-cuda}"
GPU_LIST="${CUDA_VISIBLE_DEVICES:-0}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs}"
mkdir -p "$OUTPUT_ROOT/logs"

for TRAIN_SEED in 20260805 20260807 20260808; do
  [[ -f "$OUTPUT_ROOT/main_seed_${TRAIN_SEED}/final_model.pt" ]] || { echo "missing main checkpoint for seed $TRAIN_SEED" >&2; exit 1; }
done

for SCALE in 1510 2005 2010; do
  for TRAIN_SEED in 20260805 20260807 20260808; do
    for EVAL_SEED in 20260825 20260826 20260827; do
      OUTPUT_DIR="$OUTPUT_ROOT/multiscale_v2_${SCALE}_train${TRAIN_SEED}_eval_${EVAL_SEED}"
      LOG_FILE="$OUTPUT_ROOT/logs/multiscale_${SCALE}_train${TRAIN_SEED}_eval_${EVAL_SEED}.log"
      if [[ -f "$OUTPUT_DIR/summary.json" ]] && grep -q '"passed": true' "$OUTPUT_DIR/summary.json"; then
        echo "skip completed multiscale cell: $OUTPUT_DIR"
        continue
      fi
      if [[ -e "$OUTPUT_DIR" ]]; then
        echo "incomplete output exists; remove or rename it before rerunning: $OUTPUT_DIR" >&2
        exit 1
      fi
      local_instances=80
      local_offset=20
      local_repeats=20
      if [[ "${SMOKE:-0}" == "1" ]]; then
        local_instances=4
        local_offset=0
        local_repeats=2
      fi
      CUDA_VISIBLE_DEVICES="$GPU_LIST" "$PYTHON_BIN" -u evaluate_multiscale_zero_shot.py \
        --device "$DEVICE" \
        --source-dir "$OUTPUT_ROOT/main_seed_${TRAIN_SEED}" \
        --checkpoint checkpoints/base/reference_save_10_5.pt \
        --data-dir "data_dev/$SCALE" \
        --offset "$local_offset" --instances "$local_instances" \
        --initial-job-ratio 0.6 \
        --mean-interarrival 8 --mean-time-between-failures 8 \
        --mean-repair-time 2 --events-per-machine 1 \
        --stability-budget 1.7 --sample-repeats "$local_repeats" \
        --seed "$EVAL_SEED" --output-dir "$OUTPUT_DIR" \
        2>&1 | tee "$LOG_FILE"
    done
  done
done

if [[ "${SMOKE:-0}" == "1" ]]; then
  echo "PASS multiscale smoke"
  exit 0
fi

SUMMARY_DIR="$OUTPUT_ROOT/multiscale_summary"
if [[ -e "$SUMMARY_DIR" ]]; then
  echo "summary already exists: $SUMMARY_DIR" >&2
  exit 1
fi
"$PYTHON_BIN" -u summarize_p1_multiscale.py \
  --save-dir "$OUTPUT_ROOT" \
  --directory-prefix multiscale_v2 \
  --output-dir "$SUMMARY_DIR"
echo "PASS multiscale reproduction: $SUMMARY_DIR"
