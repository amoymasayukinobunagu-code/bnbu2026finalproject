# 终稿数字入口（2026-11-17，Asia/Shanghai）

写论文只从本目录已复制的正式 `summary.json` 和已核对的消融汇总抄数字。若与 json 冲突，**以 json 为准**。

原项目仍在 `E:\\Project\\FYP`，本目录是写作快照，不依赖打开原目录也能核对已复制的数字文件。下面同时给出**本目录相对路径**和**原路径**。

## 协议与限制（抄表必读）

- 评估框：GT 紧框 + **固定外扩**（主表默认 30 像素），**不是** random jitter。PP-SAM **训练**仍用随机 0–50 px，不要和评估混写。
- ±：测试图像间标准差，**不是**多种子。
- FPS：口径不统一，论文若要报速度需在空闲 GPU 重测。本目录保留了 PP-SAM full 880 两次评估的 FPS 差异（8.30/8.83 与 8.62/8.70）。
- **PP-SAM 不是已定的 Ours**，按已有全参对照 / shot 曲线来写。
- object-scale、弱边界系统分析、多种子：**未做**，不能虚写完成。
- Chen SAM-Adapter、ASPS、OP-SAM 为无 box / dense / prompt-free，只放附录，不与主表混排。
- MedSAM+LoRA 的 **CVC 0/10/50 为用户提供、本机未重跑**（见 `ablation_box_expand/USER_PROVIDED_MEDSAM_CVC.md`）。
- SAMed Kvasir box 30：本机消融 **0.8951**，正式云端评估 **0.8955**（见 `ablation_box_expand/SAMED_BOX30_SOURCE.md`）。

可读台账：`numbers/EXPERIMENT_SUMMARY.md`。由 json 导出的终稿用表：`manuscript/tables/`。

## 主线正式结果（章程对比）

| 方法 | 本目录 | 原路径（E:\\Project\\FYP\\results\\...） |
|---|---|---|
| SAM ViT-B zero-shot | `numbers/official_core/sam_v3/summary.json` | `results/sam_v3/summary.json` |
| SAM ViT-L zero-shot | `numbers/official_core/sam_v3/vit_l/summary.json` | `results/sam_v3/vit_l/summary.json` |
| MedSAM zero-shot | `numbers/official_core/medsam_v3/summary.json` | `results/medsam_v3/summary.json` |
| SAMed LoRA r4 1/5/10/full | `numbers/official_core/samed_baseline_cloud/samed_baseline/r4_*_best/summary.json` | `results/samed_baseline_cloud/samed_baseline/r4_*_best/summary.json` |
| MedSAM+LoRA r4 full 880 | `numbers/official_core/medsam_peft/r4_full_best/summary.json` | `results/medsam_peft/r4_full_best/summary.json` |
| MSA-B | `numbers/official_core/msa_polyp_baseline_v3/summary.json` | `results/msa_polyp_baseline_v3/summary.json` |
| MSA-L | `numbers/official_core/msa_vitl_polyp_baseline_v3/summary.json` | `results/msa_vitl_polyp_baseline_v3/summary.json` |
| PP-SAM 1/5/10/full 880 | `numbers/official_core/pp_sam_baseline_v3/eval_best_{1,5,10,880}shot/summary.json` | `results/pp_sam_baseline_v3/eval_best_*shot/summary.json` |

PP-SAM full 880 的第二次评估（Dice 相同、FPS 不同）在 `numbers/official_core/pp_sam_baseline/eval_full880/`。

每份目录一般同时有 `summary.json` 与较小的 `per_image.json`（逐图指标，不是预测掩膜）。

## 补充对照（附录，不必进章程主表）

见 `numbers/appendix_evidence/`：

- `zeroshot_extra/`：HQ-SAM、SAM3、SAM2.1、MobileSAM、SAM-Med2D、SAMUS、SAM ViT-H
- `peft_extra/`：MedSAM+MSA（负向对照）
- `prompt_free/`：Chen SAM-Adapter（无 box）、ASPS-v3、发布版 ASPS 的 PraNet 测试集、OP-SAM
- `negative_probes/`：MSA 官方 zoo 非息肉探针（Kvasir 仅 5 张 sanity）

不要使用原项目里发布版 ASPS 在 Kvasir val 120 / CVC 612 上的结果（test-on-train，本目录未复制）。不要使用 `results/phase5_sam2_v3` 的 CVC（与正式 `sam2_baseline_v3` 冲突，本目录未复制）。

## 框消融

汇总：`numbers/prompt_ablation_box_expand_SUMMARY.md`  
原始 json：`numbers/ablation_box_expand/`  
用户提供的 MedSAM CVC、SAMed 两个来源说明见该目录下的标记文件。

## 不要抄

- `manuscript/legacy_reference/stale_tables/` 的 6 月 table1 / table2
- 6 月章节与 Partial Thesis 里的表
- A0/A1/B1 5-shot 烟测（本目录未收录）
- 原项目巨型权重、训练集、预测掩膜（本目录未复制）

更完整的原项目说明副本：`numbers/NUMBERS_INDEX_from_repo.md`（路径仍写原仓库；以本文件的本目录路径为准）。
