# v1.0.1 — reproduction-entry revision (unpublished)

Prepared locally after the v1.0.0 Zenodo archive, DOI https://doi.org/10.5281/zenodo.23040391. That DOI identifies v1.0.0, not these changes. No new GitHub tag, Release or Zenodo version has been published for v1.0.1.

- Unify development output directories as `finalenv_nominal_holdout_train...`, including the robustness prerequisite.
- Allow explicit frozen initial/final checkpoints without fabricating a training-output directory.
- Add development, reward-only/zero-context ablation, and 100-instance independent-test evaluation entry points, plus optional six-run ablation training.
- Preserve and verify the original independent-test data/model lock; add existing independent-test and ablation summarizers.
- Document complete evaluation/statistics commands and isolated smoke/dry-run modes.
- Synchronize supplementary authorship and author-owned copyright attribution to Yu Wang and Xiaoyao Ding, as confirmed by the submitting author on 1 October 2026; retain all upstream notices.

All 800 instances, 19 checkpoint files and existing result summaries remain byte-identical to v1.0.0. No formal experiment is rerun to create this revision. Local checks and small interface smoke runs are not new manuscript results.

## v1.0.0 — archived submission package

This fixed release accompanies the manuscript **Graph reinforcement learning for dynamic flexible job shop scheduling under composite disturbances with a schedule-stability budget**. It does not imply journal acceptance.

## Contents

Source code, 800 reused synthetic FJSP instances, 19 checkpoints, frozen protocols, selected result summaries including the independent-test summary, source-provenance records, and SHA-256 manifests.

## Changes from the initial public snapshot

- Record author-approved Apache-2.0 licensing for author-owned software/checkpoints and CC BY 4.0 for original experimental results and protocol documentation.
- Preserve upstream licenses, attributions, and scope exclusions.
- Add citation and Zenodo metadata.
- Include author/correspondence metadata and the previously archived independent-test summary.
- Add provenance and integrity manifests. No experiment is rerun for this release; no result number or checkpoint weight is changed.

The legacy nominal_holdout wrapper evaluates the 80-instance development/validation split, not the separate 100-instance independent test. The latter archived results are in results/independent_test/.

## Supplementary assets

- Supplementary_File_1_Experimental_Protocols.pdf
- Supplementary_File_2_Reproducibility_Package.zip
- SHA256SUMS.txt

Read LICENSE_SCOPE.md, NOTICE, and THIRD_PARTY_NOTICES.md. v1.0.0 is now archived at https://doi.org/10.5281/zenodo.23040391. Reproduction support is provided as a frozen research artifact without a continuing maintenance commitment.
