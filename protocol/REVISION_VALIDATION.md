# Reproduction-entry revision validation

Date: 29 September 2026. Revision: v1.0.1, local/unpublished. Archived scientific version: v1.0.0, https://doi.org/10.5281/zenodo.23040391.

## Checks performed

- All 800 instances and 19 checkpoint files match the v1.0.0 manifests. The original independent-test dataset and model hashes match the preserved test lock; development/test content overlap is zero.
- Python files parse using Python 3.8 grammar; shell entry points pass `bash -n`.
- Thirteen targeted unit tests passed, covering output-directory agreement with downstream readers, the independent 21-cell plan, ablation context/checkpoint choices, smoke isolation, archive-output protection, no-write dry runs, and both checkpoint-loading interfaces.
- Real CPU interface checks completed for development (one cell), reward-only and zero-context ablations (two cells), and independent main/static/rule evaluation (three cells). Each cell used four instances; learned-policy sampling used two candidates. All reported feasibility checks passed with zero release-time violations. These outputs are temporary validation artifacts, not manuscript results.
- The original source-directory interface and new explicit-checkpoint interface produced identical output files on the same CPU small-instance evaluation.
- The independent-test, fair-baseline, reward-only and zero-context summarizers passed when reading the existing archived full-run outputs. The 39 independent method-cell aggregates and baseline differences exactly match the archived independent-test summary.
- The development paired-statistics analysis passed on the existing archived outputs. All 24 paired comparisons agree with the archived CSV within 1e-14; the largest observed floating-point difference was 4.45e-16 in paired effect size. Archived CSVs were not modified.

CPU interface checks used local Python 3.12 and PyTorch 2.2.2; statistical verification used SciPy 1.16.1. The reported experiments used CPython 3.8.18, PyTorch 1.13.1 and SciPy 1.10.1. This revision has not undergone a new full GPU reproduction, and the local checks do not replace the published experimental results.

No formal experiment, trained weight, instance, archived result value, dynamic event semantics, reward definition, budget or candidate-selection algorithm was changed for this revision.

## Metadata and manuscript consistency check — 1 October 2026

- The submitting author confirmed that supplementary authorship and copyright for author-owned contributions list Yu Wang and Xiaoyao Ding. Upstream attributions and license texts were retained.
- Frozen-input verification again passed for 800 instances and 19 checkpoints, with zero development/test overlap and matching original test models.
- Sixteen existing targeted tests passed: reproduction workflow, constrained evaluation modes, and constrained PPO. These include the upper-clipped cost surrogate used in the corrected manuscript equation.
- The Word methods description was aligned with the existing implementation: batch-centered discounted cost returns, an upper clipped cost surrogate, and the factor 1/2 when the reward coefficient is zero and the fixed multiplier is one. No learning algorithm was changed to fit the manuscript.
- The revised supplementary PDF and v1.0.1 ZIP have new SHA-256 values. The v1.0.0 supplementary assets and DOI remain unchanged. This check does not establish a new full GPU reproduction.
