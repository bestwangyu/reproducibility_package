#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
PYTHON_BIN="${PYTHON_BIN:-python}"
DEVICE="${DEVICE:-cuda}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs}"
GPU_LIST="${CUDA_VISIBLE_DEVICES:-0}"
INSTANCES=20
ITERATIONS=100
if [[ "${SMOKE:-0}" == "1" ]]; then
  OUTPUT_ROOT="$OUTPUT_ROOT/smoke"
  INSTANCES=4
  ITERATIONS=2
fi

for MODE in reward_only zero_context; do
  if [[ "$MODE" == "reward_only" ]]; then
    REWARD=1.0; LAMBDA=0.0; CONTEXT=full
  else
    REWARD=0.0; LAMBDA=1.0; CONTEXT=zero
  fi
  for SEED in 20260805 20260807 20260808; do
    DEST="$OUTPUT_ROOT/${MODE}_seed_${SEED}"
    if [[ -e "$DEST" ]]; then
      echo "training output exists; use a fresh OUTPUT_ROOT: $DEST" >&2
      exit 1
    fi
    CMD=("$PYTHON_BIN" -u train_constrained_dynamic.py
      --device "$DEVICE" --checkpoint checkpoints/base/reference_save_10_5.pt
      --data-dir data_dev/1005 --instances "$INSTANCES" --batch-size "$INSTANCES" --iterations "$ITERATIONS"
      --initial-jobs 6 --mean-interarrival 8 --mean-time-between-failures 8
      --mean-repair-time 2 --events-per-machine 1 --stability-budget 1.7 --lr 0.0001
      --reward-coeff "$REWARD" --cost-advantage-mode batch_centered --cost-signal raw
      --stability-context-mode "$CONTEXT" --lambda-init "$LAMBDA" --lambda-lr 0.0 --lambda-max "$LAMBDA"
      --train-seed "$SEED" --scenario-seed "$((SEED + 10))" --output-dir "$DEST" --smoke-check)
    if [[ "${DRY_RUN:-0}" == "1" ]]; then
      printf '%q ' "${CMD[@]}"
      printf '\n'
      continue
    fi
    mkdir -p "$OUTPUT_ROOT/logs"
    CUDA_VISIBLE_DEVICES="$GPU_LIST" "${CMD[@]}" 2>&1 | tee "$OUTPUT_ROOT/logs/${MODE}_train_${SEED}.log"
    if [[ "${SMOKE:-0}" != "1" ]]; then
      "$PYTHON_BIN" validate_finalenv_ablation_training.py --directory "$DEST" --ablation "$MODE" --train-seed "$SEED"
    fi
  done
done
if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo "DRY RUN: 6 training commands; no output written"
else
  echo "PASS ablation training"
fi
