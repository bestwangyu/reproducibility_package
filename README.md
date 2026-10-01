# Reproducibility package

## Manuscript metadata

**Article title:** Graph reinforcement learning for dynamic flexible job shop scheduling under composite disturbances with a schedule-stability budget

**Journal:** The International Journal of Advanced Manufacturing Technology

**Authors:** Yu Wang (1, corresponding author), Xiaoyao Ding (1)

**Affiliation 1:** School of Information Engineering and Artificial Intelligence, Henan Open University, Zhengzhou 450046, China

**Corresponding author:** Yu Wang, affiliation 1; wangyu@haou.edu.com

This package contains the source code, benchmark instances, frozen checkpoints, experiment configuration, selected result summaries, tests, and commands needed to reproduce the experiments reported in the manuscript:

**Graph reinforcement learning for dynamic flexible job shop scheduling under composite disturbances with a schedule-stability budget**

The package is prepared for supplementary submission and public release with the manuscript. It intentionally excludes temporary logs, editor files, local cache directories, Python bytecode, duplicate experiment runs, and machine-specific absolute paths.

## Public repository status

Repository: https://github.com/bestwangyu/reproducibility_package

Archived submission version: **v1.0.0**, commit `d4088f0133420dce1f23d18db9f1bf934aaaa6c2`, Zenodo DOI **https://doi.org/10.5281/zenodo.23040391** (record: https://zenodo.org/records/23040391).

This working copy is the **v1.0.1 reproduction-entry revision**, prepared locally and not yet published. The DOI above identifies v1.0.0; it does not identify this revision. v1.0.1 fixes output-directory matching, adds frozen-checkpoint evaluation and independent-test/ablation entry points, and adds the original independent-test lock. Experimental assets and scientific algorithms are unchanged. Do not replace the existing v1.0.0 release or Zenodo files with this revision under the same version label.

## Package scope

- project-root `*.py` files, `env/`, `dynamic/`, `graph/`, and `utils/`: executable source code and environment modules;
- `scripts/`: portable reproduction entry points and diagnostic helpers;
- `tests/`: unit and protocol tests;
- `data_dev/1005/`: 100 development-pool FJSP instances (20 training, 80 development/validation);
- `data_dev/1510/`, `data_dev/2005/`, and `data_dev/2010/`: multi-scale development instances used for zero-shot evaluation;
- `data_test/1005/`, `data_test/1510/`, `data_test/2005/`, and `data_test/2010/`: independent holdout instances;
- `checkpoints/`: the reference checkpoint and frozen policy checkpoints used by the reported comparisons;
- `results/`: selected CSV/JSON summaries supporting the manuscript tables;
- `results/multiscale/`: the frozen 27-cell zero-shot scale summary;
- `protocol/`: machine-readable protocol metadata and file manifests;
- `requirements-reproduction.txt`: portable core dependency specification.

## Environment used for the reported runs

The archived runs were executed with Python 3.8.18 (CPython), PyTorch 1.13.1, CUDA 11.7, cuDNN 8.5.0, NumPy 1.24.3, SciPy 1.10.1, Pandas 1.5.3, and OpenPyXL 3.1.2 on an NVIDIA GeForce GTX TITAN X. The package is compatible with a newer CPython environment only after the dependencies and GPU backend have been validated.

## Dataset and split

The development pool contains 100 instances with 10 jobs and 5 machines. The first 20 instances are used for training; the remaining 80 are used for model development and validation. The independent holdout contains 100 non-overlapping instances with the same nominal scale. Zero-shot evaluation uses 15 jobs × 10 machines, 20 jobs × 5 machines, and 20 jobs × 10 machines.

All 800 supplied FJS files match the upstream `songwenas12/fjsp-drl` instances byte for byte. They are reused synthetic instances, not newly authored datasets. The base checkpoint matches upstream `results/save_10_5.pt`. See `THIRD_PARTY_NOTICES.md` and `protocol/upstream_file_provenance.csv`.

## Random seeds

- Training seeds: `20260805`, `20260807`, `20260808`.
- Evaluation seeds: `20260825`, `20260826`, `20260827`.
- The exact instance manifests and file hashes are stored under `protocol/`.

## Nominal and stress conditions

- Nominal: mean job inter-arrival time 8, mean time between failures 8, mean repair time 2, and one failure per machine.
- Stress: mean time between failures 5 and two failures per machine.
- Formal failure-boundary regression: MTBF 4 and MTBF 5 with two failures per machine.
- Simultaneous-event tolerance: `epsilon_event = 1e-5`.

## Reproduction order

Run commands from the package root in the documented CPython/PyTorch environment. The tests and rollouts require PyTorch; the dry-run planner and input verifier do not.

```bash
python verify_reproduction_inputs.py
python -m unittest discover -s tests -v
DRY_RUN=1 PYTHON_BIN=python bash scripts/run_independent_test.sh
```

The following sequence evaluates the archived checkpoints without retraining. Use one fresh output root for the development, baseline, ablation and paired-statistics steps:

```bash
export PYTHON_BIN=python DEVICE=cuda CUDA_VISIBLE_DEVICES=0
export OUTPUT_ROOT=outputs/frozen_reproduction_v1_0_1
bash scripts/run_frozen_development.sh
bash scripts/run_fair_baselines.sh
bash scripts/run_ablation_evaluation.sh
python analyze_finalenv_paired_statistics.py --save-dir "$OUTPUT_ROOT" --output-dir "$OUTPUT_ROOT/statistics_recomputed"
bash scripts/run_independent_test.sh
```

The development step evaluates 80 instances (offset 20); the independent-test step evaluates 100 distinct `data_test/1005/` instances (offset 0), with 9 main-policy cells, 9 static-policy cells and 3 dispatch-rule cells. It reports 39 method cells because each main-policy evaluation also records the initial policy and each rule evaluation contains four rules. The independent-test summary is generated at `$OUTPUT_ROOT/p0t_frozen_independent_test_summary/`.

For training reproduction and the existing robustness/multiscale routes, see **[protocol/REPRODUCTION.md](protocol/REPRODUCTION.md)**. It specifies the full commands, ablation hyperparameters, output names, expected checks, and treatment of the historical directory prefix. `run_nominal_holdout.sh` retains its filename but now emits `finalenv_nominal_holdout_train...`, matching all downstream readers.

For a small interface check, `SMOKE=1` on the new frozen-evaluation wrappers uses one training seed, one evaluation seed, four instances and two candidates under `$OUTPUT_ROOT/smoke/`. No formal summary or inferential statistics are generated for smoke outputs. `DRY_RUN=1` prints the complete planned commands without running rollouts or creating output directories. Archived results under `results/` are never overwritten.

The supplied `protocol/independent_test_v1.json` is the unchanged original test lock. It records historical development paths; the input verifier maps their model hashes to the portable `checkpoints/` paths. Reproduction uses the fixed settings and checkpoints, not new model selection on the test set. The protocol's original code hashes describe the historical experiment; the current source is covered separately by `protocol/file_manifest.sha256`.

## Data and code availability

The fixed submission release includes the supplied instances, checkpoints, selected result files, instance/checkpoint manifests, provenance, and the archived independent-test summary. Original experimental results under results/ are CC BY 4.0; original code and author-owned checkpoints are Apache-2.0. Inherited files retain their applicable terms. See LICENSE_SCOPE.md.

## License and attribution

Copyright (c) 2026 Yu Wang and Xiaoyao Ding for their original contributions.

The author team selected Apache-2.0 for its original code and checkpoints, and CC BY 4.0 for its original experimental results and protocol documentation. This selection permits uses allowed by the license texts, including commercial uses; the repository's reproduction purpose imposes no extra review-only or non-commercial restriction. It does not relicense third-party material or claim ownership of upstream instances and the base checkpoint.

Read LICENSE, LICENSES/CC-BY-4.0.txt, LICENSE_SCOPE.md, NOTICE, DATA_AND_CHECKPOINTS_NOTICE.md, and THIRD_PARTY_NOTICES.md. Citation metadata are in CITATION.cff. No continuing development or technical-support commitment is made.
