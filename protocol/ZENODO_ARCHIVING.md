# Zenodo 归档准备：v1.0.0

## 当前边界

本轮准备固定 GitHub 版本与元数据。此文件不表示已经在 Zenodo 创建记录、预留 DOI 或发布归档。不要把计划中的 DOI 写入论文。

## 已准备的元数据

- 仓库根目录 CITATION.cff：GitHub 引用按钮使用的作者、单位、题名、版本及许可。
- 仓库根目录 .zenodo.json：Zenodo GitHub 集成优先读取的元数据；同时存在两者时，Zenodo 优先使用 .zenodo.json。
- 主资源类型：Software；版本：1.0.0。
- 主要软件许可：Apache-2.0；原始实验结果/协议说明另为 CC BY 4.0；继承文件按原许可。不要用一个许可覆盖全部第三方资产。
- 不填写虚构的论文 DOI、Zenodo DOI、ORCID 或录用日期。

## 推荐的后续操作

1. 登录或注册 Zenodo。确认作者排列为 Yu Wang、Wenqiang Zhang、Xiaoyao Ding，单位与投稿稿一致。
2. GitHub Release 先发布后，可手动上传该版本的 Supplementary_File_2_Reproducibility_Package.zip 和 Supplementary_File_1_Experimental_Protocols.pdf。不要拿之后变动的 main 分支代替这个固定版本。
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
