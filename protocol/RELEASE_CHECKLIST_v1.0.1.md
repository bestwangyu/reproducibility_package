# v1.0.1 发布与归档操作

本文件对应 2026-10-01 准备的本地修订。v1.0.0 及 DOI https://doi.org/10.5281/zenodo.23040391 保留为历史归档，不覆盖、不移动标签。本次只修复复现入口及元信息，未改变调度算法、800 个实例、19 个检查点或归档结果。

## 1. 服务器验证

解压新的 v1.0.1 ZIP，进入其中的 reproducibility_package 目录，激活已有实验环境：

```bash
conda activate dyfjsp-gpu-cpython
export PYTHON_BIN=python
export DEVICE=cuda
export CUDA_VISIBLE_DEVICES=0
python verify_reproduction_inputs.py
python -m unittest tests.test_reproduction_workflow tests.test_constrained_evaluation_modes tests.test_constrained_ppo -v
DRY_RUN=1 bash scripts/run_frozen_development.sh
DRY_RUN=1 bash scripts/run_ablation_evaluation.sh
DRY_RUN=1 bash scripts/run_independent_test.sh
```

预期：输入校验 PASS，16 项测试 OK；三个 dry run 分别规划 9、18、21 个评估单元，不产生正式输出。

小规模接口验证（输出目录必须尚未存在）：

```bash
export OUTPUT_ROOT=outputs/release_1_0_1_interface
SMOKE=1 bash scripts/run_frozen_development.sh
SMOKE=1 bash scripts/run_ablation_evaluation.sh
SMOKE=1 bash scripts/run_independent_test.sh
```

预期依次出现 `PASS development smoke`、`PASS ablations smoke`、`PASS independent smoke`。这不是正式论文结果，不能混入正式汇总。

建议在发布前完成归档检查点的完整复现，无需重新训练：

```bash
export OUTPUT_ROOT=outputs/release_1_0_1_full
bash scripts/run_frozen_development.sh
bash scripts/run_fair_baselines.sh
bash scripts/run_ablation_evaluation.sh
python analyze_finalenv_paired_statistics.py --save-dir "$OUTPUT_ROOT" \
  --output-dir "$OUTPUT_ROOT/statistics_recomputed"
bash scripts/run_independent_test.sh
```

独立测试汇总应报告：21 个输出目录、39 个 method cells、跨方法场景哈希一致、release_violations=0、passed=true。归档主方法参考值为 Makespan 116.2381、稳定性成本 1.4892、预算命中率 91.78%；这些值用于核对，不能用作调参目标，也不能代替完整性与可行性检查。不同环境的数值差异需先诊断。

## 2. 发布前固定材料

1. 保留上述验证输出，核查所有正式单元完成，并确认不改变正文结果。
2. 核实 `.zenodo.json` 和 `CITATION.cff` 作者仅为 Yu Wang、Xiaoyao Ding，单位为 Henan Open University；第三方 NOTICE 保留。
3. 本轮文件中的 `unpublished` / `not yet published` 是当前状态。确定发布后，统一改为发布说明；补充 PDF 的版本状态同步更新。实际尚未分配的新 DOI 不得提前填写。更新后重新生成 `protocol/file_manifest.sha256`、ZIP 和外层 `SHA256SUMS_v1.0.1.txt`。
4. 在 GitHub 新提交上创建 v1.0.1；不要把 v1.0.0 标签移动到新提交。

## 3. 上传 GitHub

为避免混入其他本地改动，可新建干净克隆。以下命令在 Mac 工作目录执行；目标文件夹已存在时请使用新名字，不要删除旧目录。

```bash
cd /Users/wlen/Documents/CodexWork
git -c http.version=HTTP/1.1 clone \
  https://github.com/bestwangyu/reproducibility_package.git \
  reproducibility_package_release_1_0_1
rsync -a --exclude '.git/' --exclude '__pycache__/' --exclude 'outputs/' \
  fjsp-drl/docs/reproducibility_package/ \
  reproducibility_package_release_1_0_1/
cd reproducibility_package_release_1_0_1
git status --short
git diff --check
git add -A
git commit -m "Fix reproduction entry points and synchronize submission metadata"
git -c http.version=HTTP/1.1 push origin main
```

在 GitHub 的 Releases → Draft a new release 中填写：

- Tag：`v1.0.1`（新建标签，指向刚上传的提交；若已经存在，先核实，不能强制覆盖）。
- Release title：`v1.0.1 — Reproduction workflow and metadata corrections`。
- Description：

```text
This release corrects the reproduction workflow for the accompanying IJAMT manuscript.

Changes:
- Align development-evaluation output directories with downstream statistics readers.
- Add explicit frozen-checkpoint evaluation and independent-test and ablation entry points.
- Synchronize supplementary authorship and author-owned copyright attribution to Yu Wang and Xiaoyao Ding while retaining upstream notices.

The scheduling algorithm, event semantics, 800 instances, 19 checkpoint files, and archived experimental results are unchanged from v1.0.0.

The original v1.0.0 remains archived at https://doi.org/10.5281/zenodo.23040391. That DOI identifies the original version, not this release.
See protocol/REPRODUCTION.md and protocol/REVISION_VALIDATION.md for commands and the scope of validation.
```

把验证结果按实际完成情况补到描述中；不得把本机接口测试写成完整 GPU 复现。附加新 PDF、ZIP 和 SHA256SUMS_v1.0.1.txt。

## 4. Zenodo 与文稿同步

- 若使用 GitHub–Zenodo 自动归档：发布前确认仓库连接已启用；发布后核查生成的记录是否关联 v1.0.0，作者、版本、许可是否正确。
- 若旧记录是手动归档：在原记录选择 New version，上传本次固定材料，填写 1.0.1、两位作者、实际 GitHub release/commit，再发布。不要编辑旧版本文件冒充新版本。
- 不要同时建立两份独立的新归档；新版本应与旧记录关联。
- 取得实际 v1.0.1 DOI 后，同步 Word 的 Supplementary Information、Data availability、Code availability，以及补充 PDF 和仓库中的版本引用；历史 v1.0.0 DOI 保留用于来源说明。
- 最终核对文件名、版本、commit、DOI、作者、单位、校验值；退出登录后检查公开可访问性。
