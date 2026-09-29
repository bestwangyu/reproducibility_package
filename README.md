# Reproducibility package

## Manuscript metadata

**Article title:** Graph reinforcement learning for dynamic flexible job shop scheduling under composite disturbances with a schedule-stability budget

**Journal:** The International Journal of Advanced Manufacturing Technology

**Authors:** Yu Wang (1, corresponding author), Wenqiang Zhang (2), Xiaoyao Ding (1)

**Affiliation 1:** School of Information Engineering and Artificial Intelligence, Henan Open University, Zhengzhou 450046, China

**Affiliation 2:** School of Information Science and Engineering, Henan University of Technology, Zhengzhou 450001, China

**Corresponding author:** Yu Wang, affiliation 1; wangyu@haou.edu.com

This package contains the source code, benchmark instances, frozen checkpoints, experiment configuration, selected result summaries, tests, and commands needed to reproduce the experiments reported in the manuscript:

**Graph reinforcement learning for dynamic flexible job shop scheduling under composite disturbances with a schedule-stability budget**

The package is prepared for supplementary submission and public release with the manuscript. It intentionally excludes temporary logs, editor files, local cache directories, Python bytecode, duplicate experiment runs, and machine-specific absolute paths.

## Public repository status

Repository: https://github.com/bestwangyu/reproducibility_package

Submission version: **v1.0.0**. The exact published commit is available from the tag and the release record. Use this fixed version rather than the moving main branch. The earlier computational snapshot was d6a6f72300647470c240ced9e92443c3591ee1d6; v1.0.0 adds publication metadata, explicit author-approved licenses, attribution, manifests, and the archived independent-test summary without rerunning experiments.

Zenodo metadata are prepared in .zenodo.json and CITATION.cff. No Zenodo DOI has yet been assigned. A future DOI must be copied from the actual published Zenodo record, not inferred or fabricated. See protocol/ZENODO_ARCHIVING.md.

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

Run commands from the package root.

```bash
python -m unittest discover -s tests -v
```

The following commands reproduce the main analysis stages. They can require a GPU and substantial compute time. The training and evaluation wrappers write only to `outputs/`; they do not overwrite the frozen files under `results/`.

The unit suite imports PyTorch-backed environment modules. Run it inside the archived CPython/PyTorch environment; a system Python without PyTorch will execute only the dependency-free tests and report import errors for the remaining tests.

```bash
bash scripts/run_main_training.sh
bash scripts/run_nominal_holdout.sh
bash scripts/run_fair_baselines.sh
bash scripts/run_robustness.sh
bash scripts/run_multiscale_zero_shot.sh
python analyze_finalenv_paired_statistics.py --save-dir outputs --output-dir outputs/statistics_recomputed
```

The wrappers accept `PYTHON_BIN`, `DEVICE`, and `CUDA_VISIBLE_DEVICES` overrides. For a low-cost smoke check, set `SMOKE=1`; this runs a small validation subset and is not a replacement for the reported full experiment.

The scripts stop when an expected output is missing or inconsistent. Output directories are written under `outputs/` in the clean copy. Existing manuscript result summaries are under `results/` and are not overwritten by the reproduction commands.

Despite its legacy name, `scripts/run_nominal_holdout.sh` evaluates 80 development/validation instances, not the separate 100-instance independent test. The former summaries remain in `results/main_holdout/`; the newly included, unchanged independent-test summaries are in `results/independent_test/`. This metadata update does not change executable experiment logic.

## Data and code availability

The fixed submission release includes the supplied instances, checkpoints, selected result files, instance/checkpoint manifests, provenance, and the archived independent-test summary. Original experimental results under results/ are CC BY 4.0; original code and author-owned checkpoints are Apache-2.0. Inherited files retain their applicable terms. See LICENSE_SCOPE.md.

## License and attribution

Copyright (c) 2026 Yu Wang, Wenqiang Zhang, and Xiaoyao Ding for their original contributions.

The author team selected Apache-2.0 for its original code and checkpoints, and CC BY 4.0 for its original experimental results and protocol documentation. This selection permits uses allowed by the license texts, including commercial uses; the repository's reproduction purpose imposes no extra review-only or non-commercial restriction. It does not relicense third-party material or claim ownership of upstream instances and the base checkpoint.

Read LICENSE, LICENSES/CC-BY-4.0.txt, LICENSE_SCOPE.md, NOTICE, DATA_AND_CHECKPOINTS_NOTICE.md, and THIRD_PARTY_NOTICES.md. Citation metadata are in CITATION.cff. No continuing development or technical-support commitment is made.
