# Zenodo 归档状态与后续修订

v1.0.0 已公开： https://zenodo.org/records/23040391 ，版本 DOI 为 https://doi.org/10.5281/zenodo.23040391 。该 DOI 对应已归档文件，不对应当前尚未发布的 v1.0.1 复现入口修订。

以下保留首次归档时的操作说明。后续如发布 v1.0.1，应创建关联的新版本，核对新文件与新校验值，并使用该版本实际分配的 DOI；不移动 v1.0.0 标签、不覆盖历史归档。

## 当前边界

首次归档准备阶段仅准备元数据；当前 v1.0.0 的公开记录如上。v1.0.1 仍为本地修订，不填写未经分配的新 DOI。

## 已准备的元数据

- 仓库根目录 CITATION.cff：GitHub 引用按钮使用的作者、单位、题名、版本及许可。
- 仓库根目录 .zenodo.json：Zenodo GitHub 集成优先读取的元数据；同时存在两者时，Zenodo 优先使用 .zenodo.json。
- 主资源类型：Software；本次待发布版本：1.0.1；历史归档版本：1.0.0。
- 主要软件许可：Apache-2.0；原始实验结果/协议说明另为 CC BY 4.0；继承文件按原许可。不要用一个许可覆盖全部第三方资产。
- 不填写虚构的论文 DOI、Zenodo DOI、ORCID 或录用日期。

## 推荐的后续操作

1. 登录或注册 Zenodo。确认作者排列为 Yu Wang、Xiaoyao Ding，单位与投稿稿一致。v1.0.0 的历史记录保持不变；上述作者信息用于新版本。
2. 新版本的补充文件为 Supplementary_File_2_Reproducibility_Package_v1.0.1.zip、Supplementary_File_1_Experimental_Protocols_v1.0.1.pdf 和 SHA256SUMS_v1.0.1.txt。GitHub 自动归档与 Zenodo 手动新版本二选一作为本次归档入口，避免为同一版本重复创建记录；不要拿之后变动的 main 分支代替固定标签。
3. 使用 .zenodo.json 中的题名、描述、作者、关键词和版本填写表单；这份文件是元数据来源，不表示网页版必然支持直接导入。
4. 记录与实际 GitHub Release 的关联，核对上传文件的 SHA-256。权属确认仅适用于作者自己的贡献；第三方来源和许可须保留。
5. 发布前可按平台实际功能预留 DOI；预留不等于公开归档。只有真实生成的 DOI 才能回填文稿。
6. 作者核对后发布 Zenodo 记录，检查退出登录时可访问，再同步主稿 Data availability、Code availability、补充协议和仓库 README。
7. 若改用 GitHub 自动归档，应先在 Zenodo 中连接账户并启用仓库，再创建需要归档的新 Release。不要反复重建或移动 v1.0.0 标签来触发归档。

## 最终一致性检查

作者顺序、单位、版本、commit、主稿补充文件名、数据/软件许可和文件校验和必须对应同一份发布内容。本文尚未发表时，不把软件记录描述为已发表的期刊论文。

官方说明：
- https://help.zenodo.org/docs/github/describe-software/citation-file/
- https://help.zenodo.org/docs/github/describe-software/zenodo-json/
- https://help.zenodo.org/docs/github/archive-software/github-upload/
