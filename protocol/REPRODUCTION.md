# Reproduction of development, ablation and independent-test results

This guide belongs to the unpublished v1.0.1 reproduction-entry revision. The archived scientific assets are from v1.0.0, DOI https://doi.org/10.5281/zenodo.23040391. This is a reproduction workflow, not a new model-selection experiment. Run from the package root in the documented Python 3.8/PyTorch 1.13.1 environment. CPU smoke checks on newer environments do not establish GPU numerical equivalence.

## 1. Inputs and non-computing preflight

```bash
export PYTHON_BIN=python
export DEVICE=cuda
export CUDA_VISIBLE_DEVICES=0
python verify_reproduction_inputs.py
python -m unittest tests.test_reproduction_workflow tests.test_constrained_evaluation_modes -v
DRY_RUN=1 bash scripts/run_frozen_development.sh
DRY_RUN=1 bash scripts/run_ablation_evaluation.sh
DRY_RUN=1 bash scripts/run_independent_test.sh
```

The verifier checks all 800 instance and 19 checkpoint hashes against their manifests, checks non-overlap of the 100 development and 100 independent-test instances, and checks the main/static/reference model hashes against the original test lock. Expected ending: `PASS frozen input verification`. The dry runs plan 9, 18 and 21 evaluation cells, respectively, and create no output. The unchanged `independent_test_v1.json` contains historical paths; the verifier explicitly maps them to the released checkpoint names.

## 2. Small interface checks

```bash
export OUTPUT_ROOT=outputs/interface_check_v1_0_1
SMOKE=1 bash scripts/run_frozen_development.sh
SMOKE=1 bash scripts/run_ablation_evaluation.sh
SMOKE=1 bash scripts/run_independent_test.sh
```

Each command uses one training seed and one evaluation seed. Development has one cell; ablations have two; independent testing has three (main, static DRL, and four dispatch rules in one cell). Each cell uses four instances; learned-policy sampling uses two candidates. Results are placed in `$OUTPUT_ROOT/smoke/`. Expected endings start with `PASS development smoke`, `PASS ablations smoke`, and `PASS independent smoke`. These are interface checks, not the full experiments. No full-study summary or statistical conclusion is produced.

## 3. Archived-checkpoint evaluation: preferred route

No training is necessary for this route. Choose a fresh output root; do not combine smoke outputs, old `nominal_train...` directories, or different checkpoints in one analysis.

```bash
export OUTPUT_ROOT=outputs/frozen_reproduction_v1_0_1
bash scripts/run_frozen_development.sh
bash scripts/run_fair_baselines.sh
bash scripts/run_ablation_evaluation.sh
python analyze_finalenv_paired_statistics.py \
  --save-dir "$OUTPUT_ROOT" \
  --output-dir "$OUTPUT_ROOT/statistics_recomputed"
bash scripts/run_independent_test.sh
```

| Step | Dataset and protocol | Outputs relative to OUTPUT_ROOT |
| --- | --- | --- |
| Main development | `data_dev/1005`, offset 20, 80 instances, 3 training × 3 evaluation seeds, Best-of-20, budget 1.7 | 9 `finalenv_nominal_holdout_train{train}_eval_{eval}/` directories; `nominal_summary/` |
| Fair baselines | Same 80 development instances/scenarios; 3×3 static DRL greedy cells, 3 dispatch-rule cells | `finalenv_static_drl_zero_shot_train.../`, `finalenv_dispatch_rules_eval.../`, `fair_baselines_summary/` |
| Reward-only ablation | Same development split/scenarios; full stability context; budget-first Best-of-20 | 9 `finalenv_reward_only_holdout_train.../` directories; `finalenv_reward_only_holdout_3x3_summary/` |
| Zero-context ablation | Same development split/scenarios; zero context; budget-first Best-of-20 | 9 `finalenv_zero_context_holdout_train.../` directories; `finalenv_zero_context_holdout_3x3_summary/` |
| Paired statistics | The above main, ablation, static and rule evaluations; average 3 training runs within each instance × evaluation-seed unit | `statistics_recomputed/paired_unit_observations.csv`, `paired_comparisons.csv`, `summary.json` |
| Independent test | `data_test/1005`, offset 0, all 100 instances; archived models only | 9 `p0t_independent_main_train.../`, 9 `p0t_independent_static_drl_train.../`, 3 `p0t_independent_rules_eval.../`; `p0t_frozen_independent_test_summary/` |

The independent step is separate from the 80-instance development statistics. Its summary reports `output_directories=21`, `method_cells=39`, `scenario_hashes_match_across_all_methods=true`, `release_violations=0`, and `passed=True`. The 39 cells include 9 main and 9 initial-policy cells, 9 static cells, and 12 rule cells. Seven methods are summarized; the initial policy is evaluated alongside the main policy.

Archived independent-test main means for comparison are makespan 116.2381, stability cost 1.4892 and budget-hit rate 91.78%. These numbers describe the archived experiment, not a pass/fail tolerance for another hardware/software environment. Completion, feasibility, identical data/scenarios and statistics must be checked separately from numerical agreement. Never tune a model, budget or checkpoint using this test result.

The conditioned initial checkpoints supplied under `checkpoints/main/` are byte-identical to the original reward-only and zero-context conditioned initial checkpoints for the matching seeds. `--stability-context-mode zero` changes context use during inference, not the shared initial weight file. The archived ablation final checkpoints remain separate.

## 4. Optional training reproduction

Use a different output root from the archived-checkpoint route:

```bash
export OUTPUT_ROOT=outputs/retrained_reproduction_v1_0_1
bash scripts/run_main_training.sh
bash scripts/run_nominal_holdout.sh
bash scripts/run_ablation_training.sh
MODELS=retrained bash scripts/run_ablation_evaluation.sh
bash scripts/run_fair_baselines.sh
python analyze_finalenv_paired_statistics.py \
  --save-dir "$OUTPUT_ROOT" \
  --output-dir "$OUTPUT_ROOT/statistics_recomputed"
bash scripts/run_robustness.sh
bash scripts/run_multiscale_zero_shot.sh
```

The main and ablation trainers each use 20 training instances, batch size 20, 100 iterations, learning rate 0.0001, and training seeds 20260805/20260807/20260808. Scenario seeds equal training seed + 10. Nominal arrival/failure/repair settings are 8/8/2, with one failure per machine and budget 1.7. All use raw cost and batch-centered cost advantages.

| Training condition | reward coefficient | fixed multiplier | multiplier learning rate | context mode |
| --- | --- | --- | --- | --- |
| Main | 0.0 | 1.0 | 0.0 | full |
| Reward-only | 1.0 | 0.0 | 0.0 | full |
| Zero-context | 0.0 | 1.0 | 0.0 | zero |

The optional ablation trainer runs six training jobs and validates each completed formal run using `validate_finalenv_ablation_training.py`. `DRY_RUN=1 bash scripts/run_ablation_training.sh` prints all six commands. Training results must not replace the published independent-test checkpoint selection; use the frozen route for that test.

## 5. Paths, resumption and interpretation

The canonical development prefix is `finalenv_nominal_holdout`. It is shared by the original-training evaluation wrapper, the frozen-evaluation runner, the fair-baseline summarizer, the robustness prerequisite/summary and paired statistics. Do not change only one side of this contract.

The new runner records the exact command in each completed cell's `reproduction_request.json`. It resumes only complete cells with a matching request; partial or unmatched directories cause a clear error and are not overwritten. Existing summaries also stop execution. Use a fresh OUTPUT_ROOT for changed settings. Existing old-prefix runs are not deleted or silently renamed.

Independent-test summaries use the original validated aggregation script and original data/model lock. The final-environment paired-statistics script is for the 80-instance development/validation comparisons and must not be relabeled as the independent-test analysis. The archived `results/` files remain unchanged in both routes.
