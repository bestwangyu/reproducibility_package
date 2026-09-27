# Reproducibility package

This package contains the source code, benchmark instances, frozen checkpoints, experiment configuration, selected result summaries, tests, and commands needed to reproduce the experiments reported in the manuscript:

**Graph reinforcement learning for dynamic flexible job shop scheduling under composite disturbances with a schedule-stability budget**

The package is prepared for supplementary submission and for a later GitHub or archival-repository deposit. It intentionally excludes temporary logs, editor files, local cache directories, Python bytecode, duplicate experiment runs, and machine-specific absolute paths.

## Public repository status

The working repository is [https://github.com/bestwangyu/reproducibility_package](https://github.com/bestwangyu/reproducibility_package). No DOI has been assigned yet. The repository should remain private until the authors confirm ownership and redistribution permission for the generated instances and frozen checkpoints. A public release should add a project-specific software license and a clear data/checkpoint redistribution statement.

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

## Data and code availability

At the time this package was prepared, no public repository DOI or GitHub URL had been assigned. The final manuscript should replace the corresponding availability statements with the permanent repository URL/DOI after deposit. Until then, this package can be uploaded as a supplementary file through the journal submission system.

## License and reuse

No redistribution license is asserted in this supplementary package. Before public release on GitHub or an archival repository, the authors must choose and add an appropriate license, confirm third-party dependency notices, and confirm that the generated instances and checkpoints may be redistributed. Until then, repository visibility should be set to private.
