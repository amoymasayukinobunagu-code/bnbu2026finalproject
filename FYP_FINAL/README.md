# FYP 终稿写作材料

快照日期：**2026-11-17**（Asia/Shanghai）  
位置：`E:\\Project\\FYP_FINAL`  
原项目：`E:\\Project\\FYP`（完整保留，未搬家、未删除）  
`E:\\Project\\fyp_github` 未改动。

本目录只放写最终毕业论文真正需要的材料：官方课题要求、划分文件、正式数字证据、可复用格式，以及标明未审核的历史稿。**不是原项目全拷贝**，也不是新的训练环境。

先读这三份：

1. `README.md`（本文件）
2. `charter/SCOPE.md`（按 Index 1 的范围与限制）
3. `numbers/NUMBERS_INDEX.md`（数字从哪抄）

追溯清单：`SOURCE_MANIFEST.md` / `SOURCE_MANIFEST.csv`。

## 目录图

```
FYP_FINAL/
  README.md                          入口
  SOURCE_MANIFEST.md / .csv          相对路径、原路径、SHA256
  charter/                           官方课题 PDF、PROJECT_BRIEF、范围说明
  splits/                            仅 Kvasir train/val 文件名列表（无图像）
  numbers/                           正式数字与说明（入口在 NUMBERS_INDEX.md）
    official_core/                   章程主线：SAM / MedSAM / SAMed / LoRA / MSA / PP-SAM
    appendix_evidence/               补充对照，勿塞进主表
    ablation_box_expand/             固定外扩 0/10/30/50 原始 json
    EXPERIMENT_SUMMARY.md            可读台账副本
  manuscript/
    tables/                          由正式 json 导出的新表（无 6 月旧表）
    figures/                         空目录，留给新图
    ref.bib                          引用库，使用前需核验
    template/                        格式样例与 Overleaf 说明
    legacy_reference/                6 月章节/旧表/旧图，未按正式结果审核
```

## 主线与范围

按 `charter/PROJECT_BRIEF.md` 的 A–F 与 `charter/Donglong_Chen_2024.pdf` Index 1：这是对 **零样本 SAM、零样本 MedSAM、轻量 Adapter/LoRA** 的系统对比与复现适配研究，数据为 Kvasir-SEG 与 CVC-ClinicDB。

主体实验已经齐，可以收束写终稿。多种子、额外零样本框消融、A1/B1 全量训练**不是**开写门槛。

**PP-SAM 是已有方法 / 全参对照，不要默认写成 Ours 或本课题原创模型。** 不得仅凭优于零样本就断言达到 SOTA。object-scale 与弱边界尚未系统量化，正文必须如实限制。

## 正式结果 vs 补充对照

- **主线（可进正文主表）**：`numbers/official_core/`
- **补充零样本与额外 PEFT（附录）**：`numbers/appendix_evidence/zeroshot_extra/`、`peft_extra/`
- **无 box / 协议不同（附录，不混排）**：`numbers/appendix_evidence/prompt_free/`（含 Chen SAM-Adapter）
- **框消融**：`numbers/ablation_box_expand/` 与 `numbers/prompt_ablation_box_expand_SUMMARY.md`

## 历史参考（不要当正式数字）

`manuscript/legacy_reference/`：6 月章节、过时 LaTeX 表、旧定性图、Partial Thesis。仅供结构参考。

## 现在要做的

更新论文正文、用 `manuscript/tables/` 或正式 json 制表、补新图、核验 `manuscript/ref.bib`。  
不必默认再开训练或扩展实验。代码与大数据集仍在原项目中，需要复现实验时再回到 `E:\\Project\\FYP`。
