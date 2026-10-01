#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"
DEVICE="${DEVICE:-cuda}"
GPU_LIST="${CUDA_VISIBLE_DEVICES:-0}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs}"
SAMPLE_REPEATS="${SAMPLE_REPEATS:-20}"
mkdir -p "$OUTPUT_ROOT/logs"
INSTANCES=80
OFFSET=20
if [[ "${SMOKE:-0}" == "1" ]]; then
  INSTANCES=4
  OFFSET=0
fi

for TRAIN_SEED in 20260805 20260807 20260808; do
  [[ -f "$OUTPUT_ROOT/main_seed_${TRAIN_SEED}/final_model.pt" ]] || { echo "missing main checkpoint for seed $TRAIN_SEED" >&2; exit 1; }
done
for TRAIN_SEED in 20260805 20260807 20260808; do
  for EVAL_SEED in 20260825 20260826 20260827; do
    [[ -f "$OUTPUT_ROOT/finalenv_nominal_holdout_train${TRAIN_SEED}_eval_${EVAL_SEED}/summary.json" ]] || {
      echo "run scripts/run_nominal_holdout.sh before robustness evaluation" >&2; exit 1;
    }
  done
done

run_condition() {
  local condition="$1" prefix="$2" iat="$3" mtbf="$4" repair="$5" events="$6"
  for TRAIN_SEED in 20260805 20260807 20260808; do
    for EVAL_SEED in 20260825 20260826 20260827; do
      local source_dir="$OUTPUT_ROOT/main_seed_${TRAIN_SEED}"
      local output_dir="$OUTPUT_ROOT/${prefix}_train${TRAIN_SEED}_eval_${EVAL_SEED}"
      local log_file="$OUTPUT_ROOT/logs/${condition}_train${TRAIN_SEED}_eval_${EVAL_SEED}.log"
      if [[ -f "$output_dir/summary.json" ]] && grep -q '"passed": true' "$output_dir/summary.json"; then
        echo "skip completed robustness cell: $output_dir"
        continue
      fi
      if [[ -e "$output_dir" ]]; then
        echo "incomplete output exists; remove or rename it before rerunning: $output_dir" >&2
        exit 1
      fi
      local repeats="$SAMPLE_REPEATS"
      if [[ "${SMOKE:-0}" == "1" ]]; then repeats=2; fi
      CUDA_VISIBLE_DEVICES="$GPU_LIST" "$PYTHON_BIN" -u evaluate_constrained_pilot.py \
        --device "$DEVICE" --source-dir "$source_dir" \
        --checkpoint checkpoints/base/reference_save_10_5.pt \
        --data-dir data_dev/1005 --offset "$OFFSET" --instances "$INSTANCES" \
        --initial-jobs 6 --mean-interarrival "$iat" \
        --mean-time-between-failures "$mtbf" --mean-repair-time "$repair" \
        --events-per-machine "$events" --stability-budget 1.7 \
        --seed "$EVAL_SEED" --sample-repeats "$repeats" \
        --stability-context-mode full --selection-mode budget_first \
        --output-dir "$output_dir" 2>&1 | tee "$log_file"
    done
  done
}

run_condition mild finalenv_robustness_mild 12 12 1 1
run_condition severe finalenv_robustness_severe 4 4 3 1
run_condition multi_failure_mtbf5 finalenv_robustness_multifailure_mtbf5 8 5 2 2

if [[ "${SMOKE:-0}" == "1" ]]; then
  echo "PASS robustness smoke"
  exit 0
fi

SUMMARY_DIR="$OUTPUT_ROOT/robustness_summary"
if [[ -e "$SUMMARY_DIR" ]]; then
  echo "summary already exists: $SUMMARY_DIR" >&2
  exit 1
fi
"$PYTHON_BIN" -u summarize_finalenv_robustness.py \
  --save-dir "$OUTPUT_ROOT" --output-dir "$SUMMARY_DIR"
echo "PASS robustness reproduction: $SUMMARY_DIR"
