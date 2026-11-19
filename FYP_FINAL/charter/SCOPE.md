# 终稿范围（按官方 Index 1 / PROJECT_BRIEF A–F）

快照日期：2026-11-17（Asia/Shanghai）

本目录按照导师课题书 **Index 1** 与 `charter/PROJECT_BRIEF.md` 的 **A–F** 收束，用于撰写最终毕业论文。不是继续实验的工作区，也不复制原项目的训练数据、权重或代码环境。

## 已经齐、可以据此写正文的

- 零样本 SAM（ViT-B / ViT-L）、零样本 MedSAM
- 轻量适配：SAMed LoRA、MedSAM+LoRA、息肉域 MSA-B / MSA-L
- 数据与协议：Kvasir-SEG（train 880 / val 120）与 CVC-ClinicDB（全 612）；评估为 GT 紧框固定外扩（主表默认 30 像素），不是 random jitter
- 复现与评估代码仍在原项目 `E:\Project\FYP\code\`（本目录不复制代码与环境）
- PP-SAM 的 1/5/10/full-880 是已有方法的全参 / shot 曲线对照，**不是已定的 Ours，也不是本课题必须主张的原创模型**

## 补充对照（可写附录，不必塞进章程主表）

HQ-SAM、SAM3、SAM2.1、MobileSAM、SAM-Med2D、SAMUS、SAM ViT-H、MedSAM+MSA，以及无 box 的 Chen SAM-Adapter、ASPS、OP-SAM。Chen / ASPS / OP-SAM **不得与 box-prompt 主表混排排名**。

## 正文必须如实限制、不能写成已系统完成的

- object-scale（按面积分档）与 weak-boundary（弱边界）是课题描述中的分析目标，但**尚未系统量化**。正文可以讨论、也可以用已有预测做探索性分箱，不能声称已完成系统分析。
- 多种子与显著性检验未做。表中 ± 是测试图像间标准差。
- 额外的零样本 SAM-L / MedSAM 框消融、A0/A1/B1 的 5-shot 之后的正式训练，**不是开写门槛**，也未放进本目录的正式主表。
- 不得仅因优于零样本就断言达到 SOTA。课题书的 “meet the SOTA” 不能只用这条来宣称。

## 现在真正要做的

更新论文正文、正式表格与新图、核验引用。不必默认再追加训练或扩展实验。
