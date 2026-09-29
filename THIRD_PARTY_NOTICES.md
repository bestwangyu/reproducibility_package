# Third-party sources and attribution

This package extends fjsp-drl, the implementation accompanying:

Song W, Chen X, Li Q, Cao Z (2023) Flexible job shop scheduling via graph neural network and deep reinforcement learning. IEEE Transactions on Industrial Informatics 19(2):1600–1610. https://doi.org/10.1109/TII.2022.3189725

Source: https://github.com/songwenas12/fjsp-drl

Audit snapshot: d7637b70029be6cb8fe78e450dcb07767876daea, checked 29 September 2026. This is not a claim about the historical download revision. Upstream declares Apache-2.0; LICENSE is preserved verbatim. No root NOTICE file was present in the audited tree.

## Inherited files and modifications

The package includes upstream source modules, configuration, synthetic instances, and a base checkpoint. protocol/upstream_file_provenance.csv records byte-identical matches. A nonmatching hash can reflect edits, another source revision, or another origin and does not establish original authorship.

Study extensions add dynamic events, stability evaluation/training, candidate selection, wrappers, tests, and result summaries. Original copyright and attribution must remain. Modified upstream files require appropriate modification notices and a history check before distribution; this metadata update does not certify all file histories.

## Transitive-source provenance

- mlp.py names https://github.com/zcaicaros/L2D as a reference. The GitHub license endpoint returned no repository-level license on 29 September 2026. This is a provenance warning, not a determination that permission is absent. Confirm applicable file-level or separate permission before declaring the package fully rights-cleared.
- utils/gpu_mem_track.py names https://github.com/Oldpan/Pytorch-Memory-Utils. Preserve that attribution and confirm applicable upstream terms if distributing the helper.
- Installed dependencies retain their own licenses and are not relicensed here.

## Author-created additions

The author team confirmed its authority and selected Apache-2.0 for author-owned code and checkpoints and CC BY 4.0 for original experimental results and protocol documentation on 29 September 2026. These choices do not override third-party rights or resolve missing source-specific information. LICENSE_SCOPE.md and DATA_AND_CHECKPOINTS_NOTICE.md define scope and exclusions.
