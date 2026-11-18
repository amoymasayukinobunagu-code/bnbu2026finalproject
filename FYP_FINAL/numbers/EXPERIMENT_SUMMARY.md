> **FYP_FINAL 写作快照（2026-11-17，Asia/Shanghai）**  
> 本文件是从 `E:\\Project\\FYP` 复制到 `E:\\Project\\FYP_FINAL` 的写作材料，原项目未移动、未删除。  
> 请以本目录 `numbers/NUMBERS_INDEX.md` 为入口。文中若仍出现 `E:\\Project\\FYP\\results\\...`，那是**原路径**，对应副本见本目录相对路径（见 SOURCE_MANIFEST）。  
> 可读台账副本；抄数字请同时核对本目录 json。

# FYP 实验总结（截至 2026-11-17）

本文件是整理后的可读台账，汇总已跑的 segmentation 实验并标注协议是否公平。  
**数字若与正式 `summary.json` 冲突，以 json 为准。** 抄表前先看 `docs/NUMBERS_INDEX.md`。

> **2026-11-17 台账更新（重要）**
> - 按本机正式 `summary.json` 校正主表：补入 HQ-SAM、SAM3、MSA-B、MSA-L、MedSAM+LoRA、Chen SAM-Adapter、MedSAM+MSA；SAM3 从 pending 并入主表。
> - 评估框改为写明：**GT 紧框 + 固定外扩**（主表默认 30 px），**不是** random jitter。
> - **± 是测试图像间标准差，不是多种子。** 多种子与显著性检验尚未做。
> - **FPS 口径不统一**，不同 json / 复跑不要假装可直接比；论文若要报 FPS 需空闲 GPU 重测。
> - **PP-SAM 是否写作 Ours 未定**，主表不再标 ours。
> - 纠正旧稿冲突：B6 SAMUS 叙述段曾写 0.6778 / 0.5124，与 `phase6_samus_v3/summary.json` 的 0.6024 / 0.3516 不符，**以 json 为准**（主表原本已是 json 数）。
> - 纠正主表笔误：B1 CVC Dice 标准差应为 0.363（旧表误写 0.191）；B2 CVC 应为 0.211（旧表误写 0.175）；B3 CVC 应为 0.021（旧表误写 0.011，疑与 IoU std 混淆）；B7 Hiera-S Kvasir 标准差应为 0.218（旧表误写 0.213）。
> - B1 / B2 等 FPS 改回对应正式 json（例如 SAM ViT-B 为 6.92 / 7.14，不再用来源不明的 8.13 / 8.37）。
> - 息肉域 MSA-B / MSA-L 训练与正式评估已完成，不再写成 future work。官方 zoo 的非息肉探针仍只作负向背景。
> - Prompt 消融（固定外扩 0/10/30/50）已写入 §1.5。A0/A1/B1 5-shot 仍是烟测，不进正式表。object-scale / 弱边界系统分析 / 多种子：**未做**。
>
> **2026-06-16 v3 大改写 (重要)**: 全 baseline 协议统一 + 4 个 baseline 重新跑。
> - Kvasir 评估统一用官方 **val.txt (120 张 held-out)**（旧 v2 用 `sorted[:120]`，跟 val.txt 仅 23.3% 重叠）
> - 评估 box 统一为 GT 紧框外扩（当时文档常写成「±30 px 扰动 / PP-SAM 论文协议」；2026-11-17 起统一写 **固定外扩**，避免被抄成 random）
> - B4 PP-SAM **重训**：训练样本从 train.txt (880 张) 抽，不再 leak val.txt
> - 全 baseline 报 **5 指标**：Dice / IoU / Precision / Recall / HD95
>
> **2026-06-10 v2 / v1 更新**：见历史备份；v2 数字已废弃。2026-10-08 曾补 PP-SAM full 880；2026-10-12 起的 HQ-SAM / SAM3 / MSA / LoRA / Chen 当时未并入本文件主表。

---

## 0. 评估协议统一（重要）

| 协议项 | v2 (旧, 已被覆盖) | **v3 (当前正式评估)** |
|---|---|---|
| 数据选择 (Kvasir) | `sorted[:N]` (旧) | **`Data-split/val.txt` 120 张 (held-out, 官方)** |
| 数据选择 (CVC) | 全 612 | 全 612 (CVC 无官方 split) |
| 训练抽样 | 部分实验曾从全 1000 张 sample | **`Data-split/train.txt` 880 张**（严格不 leak val.txt） |
| Box prompt（评估） | 文档曾写「无扰动 / ±30 px 扰动」 | **GT 紧框 + 固定外扩**（主表默认 30 像素；**不是** random jitter） |
| Box prompt（PP-SAM 训练） | — | 训练仍用随机扰动 0–50 px，与评估的固定外扩不是同一件事 |
| Multimask | False（多数）/ True + argmax IoU（MSA） | 同左；Chen / ASPS / OP-SAM 不走 box-prompt |
| 评估指标 | 3 项 | **5 项 (Dice / IoU / Precision / Recall / HD95)**；FPS 另计，口径不统一 |

**± 的含义**：表中 `Dice ±` 来自该次评估 `summary.json` 的 `dice_std`，是 **测试图像间** 波动，不是多种子重复训练的标准差，也不能当作显著性检验。

**FPS**：必须视为载荷/实现相关测值。同一权重的不同 json 可能 Dice 一致但 FPS 不同（例如 PP-SAM full 880 的两次评估、MSA-B 的两份 Kvasir FPS、SAM / MedSAM 的 phase 目录与 `*_v3` 目录）。论文若要报速度，需在空闲 GPU 上按统一脚本重测。

**不要直接排名的行**：ASPS（dense / 无 box）、OP-SAM（prompt-free）、Chen SAM-Adapter（automatic，无 box）、发布版 ASPS 在 val120/CVC612 上的 test-on-train 结果。

---

## 1. 公平对比表（v3 主表，固定外扩 30 px）

抄表请同时打开 `docs/NUMBERS_INDEX.md`。下表 Dice / IoU / P / R / HD95 均来自正式 json；FPS 取自同一份 json，**仅供参考**。未把 PP-SAM 标为 Ours。主表不加粗「最佳」，避免在协议不完全相同或「Ours」未定时被误读。

### 1.1 Box-prompt 方法

| 模型 | 数据集 | N | Dice ↑ | IoU ↑ | Precision | Recall | HD95 ↓ | FPS* |
|---|---|---|---|---|---|---|---|---|
| SAM ViT-B (zero-shot) | Kvasir-SEG val | 120 | 0.7497 ± 0.295 | 0.6680 | 0.7290 | 0.8012 | 95.63 | 6.92 |
| SAM ViT-B (zero-shot) | CVC-ClinicDB | 612 | 0.5182 ± 0.363 | 0.4300 | 0.4767 | 0.6377 | 63.76 | 7.14 |
| SAM ViT-L (zero-shot, 308M) | Kvasir-SEG val | 120 | 0.8140 ± 0.239 | 0.7367 | 0.7873 | 0.8862 | 91.53 | 3.75 |
| SAM ViT-L (zero-shot, 308M) | CVC-ClinicDB | 612 | 0.6588 ± 0.323 | 0.5676 | 0.5994 | 0.8181 | 51.90 | 3.87 |
| SAM ViT-H (zero-shot, 636M) | Kvasir-SEG val | 120 | 0.7574 ± 0.299 | 0.6808 | 0.7187 | 0.8360 | 102.04 | 2.29 |
| SAM ViT-H (zero-shot, 636M) | CVC-ClinicDB | 612 | 0.6323 ± 0.329 | 0.5378 | 0.5597 | 0.7932 | 55.93 | 2.30 |
| MedSAM (zero-shot) | Kvasir-SEG val | 120 | 0.7034 ± 0.235 | 0.5846 | 0.6217 | 0.8394 | 96.35 | 6.91 |
| MedSAM (zero-shot) | CVC-ClinicDB | 612 | 0.5172 ± 0.211 | 0.3755 | 0.4096 | 0.7672 | 51.65 | 7.15 |
| HQ-SAM ViT-B (zero-shot) | Kvasir-SEG val | 120 | 0.8007 ± 0.219 | 0.7104 | 0.7633 | 0.8891 | 91.09 | 7.71 |
| HQ-SAM ViT-B (zero-shot) | CVC-ClinicDB | 612 | 0.6525 ± 0.294 | 0.5477 | 0.6339 | 0.7996 | 51.72 | 7.93 |
| SAM3 (zero-shot, box XYXY) | Kvasir-SEG val | 120 | 0.7661 ± 0.247 | 0.6715 | 0.7057 | 0.8857 | 96.75 | 7.26 |
| SAM3 (zero-shot, box XYXY) | CVC-ClinicDB | 612 | 0.6287 ± 0.275 | 0.5104 | 0.5268 | 0.8521 | 56.81 | 7.60 |
| SAM-Med2D (zero-shot, Adapter, 256²) | Kvasir-SEG val | 120 | 0.6577 ± 0.203 | 0.5230 | 0.7909 | 0.6086 | 97.77 | 29.55 |
| SAM-Med2D (zero-shot, Adapter, 256²) | CVC-ClinicDB | 612 | 0.5922 ± 0.239 | 0.4611 | 0.6538 | 0.5929 | 51.25 | 30.44 |
| MobileSAM (zero-shot, TinyViT) | Kvasir-SEG val | 120 | 0.7848 ± 0.204 | 0.6838 | 0.7100 | 0.9393 | 102.09 | 24.54 |
| MobileSAM (zero-shot, TinyViT) | CVC-ClinicDB | 612 | 0.5974 ± 0.283 | 0.4810 | 0.4962 | 0.8644 | 59.46 | 33.59 |
| SAMUS (zero-shot, ultrasound, 256²) | Kvasir-SEG val | 120 | 0.6024 ± 0.199 | 0.4571 | 0.5323 | 0.7456 | 138.40 | 20.53 |
| SAMUS (zero-shot, ultrasound, 256²) | CVC-ClinicDB | 612 | 0.3516 ± 0.215 | 0.2348 | 0.3021 | 0.4863 | 87.19 | 20.08 |
| SAM2.1 Hiera-S (zero-shot) | Kvasir-SEG val | 120 | 0.7738 ± 0.218 | 0.6727 | 0.6986 | 0.9174 | 104.11 | 13.96 |
| SAM2.1 Hiera-S (zero-shot) | CVC-ClinicDB | 612 | 0.5378 ± 0.314 | 0.4291 | 0.4512 | 0.7844 | 65.36 | 16.15 |
| SAM2.1 Hiera-B+ (zero-shot) | Kvasir-SEG val | 120 | 0.7318 ± 0.236 | 0.6227 | 0.6455 | 0.9058 | 121.48 | 10.24 |
| SAM2.1 Hiera-B+ (zero-shot) | CVC-ClinicDB | 612 | 0.4475 ± 0.288 | 0.3345 | 0.3655 | 0.7043 | 82.14 | 13.74 |
| SAM2.1 Hiera-L (zero-shot) | Kvasir-SEG val | 120 | 0.7121 ± 0.296 | 0.6203 | 0.6438 | 0.8544 | 110.46 | 6.31 |
| SAM2.1 Hiera-L (zero-shot) | CVC-ClinicDB | 612 | 0.5754 ± 0.299 | 0.4629 | 0.4815 | 0.8182 | 62.73 | 6.52 |
| PP-SAM 1-shot | Kvasir-SEG val | 120 | 0.7992 ± 0.222 | 0.7096 | 0.7764 | 0.8650 | 87.71 | 6.54 |
| PP-SAM 1-shot | CVC-ClinicDB | 612 | 0.7516 ± 0.200 | 0.6362 | 0.6605 | 0.9222 | 44.55 | 7.34 |
| PP-SAM 5-shot | Kvasir-SEG val | 120 | 0.8512 ± 0.162 | 0.7683 | 0.8223 | 0.9159 | 80.25 | 6.73 |
| PP-SAM 5-shot | CVC-ClinicDB | 612 | 0.8040 ± 0.155 | 0.6962 | 0.7212 | 0.9545 | 39.47 | 7.68 |
| PP-SAM 10-shot | Kvasir-SEG val | 120 | 0.8550 ± 0.140 | 0.7682 | 0.8444 | 0.8959 | 83.23 | 8.03 |
| PP-SAM 10-shot | CVC-ClinicDB | 612 | 0.7709 ± 0.165 | 0.6533 | 0.6932 | 0.9361 | 40.56 | 8.23 |
| PP-SAM full 880 | Kvasir-SEG val | 120 | 0.9226 ± 0.0705 | 0.8630 | 0.8962 | 0.9594 | 66.25 | 8.30 / 8.62 |
| PP-SAM full 880 | CVC-ClinicDB | 612 | 0.8576 ± 0.130 | 0.7699 | 0.8067 | 0.9511 | 27.72 | 8.83 / 8.70 |
| SAMed LoRA r4 1-shot | Kvasir-SEG val | 120 | 0.7456 ± 0.204 | 0.6287 | 0.6367 | 0.9418 | 108.20 | 12.45 |
| SAMed LoRA r4 1-shot | CVC-ClinicDB | 612 | 0.5759 ± 0.196 | 0.4310 | 0.4348 | 0.9660 | 64.98 | 13.19 |
| SAMed LoRA r4 5-shot | Kvasir-SEG val | 120 | 0.7571 ± 0.212 | 0.6469 | 0.6601 | 0.9360 | 112.13 | 12.54 |
| SAMed LoRA r4 5-shot | CVC-ClinicDB | 612 | 0.5747 ± 0.244 | 0.4418 | 0.4510 | 0.8933 | 63.71 | 13.25 |
| SAMed LoRA r4 10-shot | Kvasir-SEG val | 120 | 0.7230 ± 0.166 | 0.5909 | 0.5972 | 0.9805 | 123.19 | 12.35 |
| SAMed LoRA r4 10-shot | CVC-ClinicDB | 612 | 0.5435 ± 0.180 | 0.3950 | 0.3972 | 0.9892 | 70.79 | 13.18 |
| SAMed LoRA r4 full 880 | Kvasir-SEG val | 120 | 0.8955 ± 0.121 | 0.8280 | 0.8866 | 0.9334 | 88.29 | 12.59 |
| SAMed LoRA r4 full 880 | CVC-ClinicDB | 612 | 0.8608 ± 0.159 | 0.7814 | 0.8513 | 0.9125 | 32.75 | 13.15 |
| MedSAM+LoRA r4 full 880 | Kvasir-SEG val | 120 | 0.9049 ± 0.0978 | 0.8375 | 0.8692 | 0.9624 | 81.71 | 5.76 |
| MedSAM+LoRA r4 full 880 | CVC-ClinicDB | 612 | 0.8476 ± 0.153 | 0.7596 | 0.8246 | 0.9098 | 35.41 | 6.03 |
| MSA-B（息肉 Adapter, ViT-B, 880） | Kvasir-SEG val | 120 | 0.8524 ± 0.202 | 0.7826 | 0.8864 | 0.8670 | 104.97 | 19.68 |
| MSA-B（息肉 Adapter, ViT-B, 880） | CVC-ClinicDB | 612 | 0.7685 ± 0.281 | 0.6896 | 0.7998 | 0.8342 | 61.59 | 22.71 |
| MSA-L（息肉 Adapter, ViT-L, 880） | Kvasir-SEG val | 120 | 0.8555 ± 0.182 | 0.7820 | 0.9118 | 0.8567 | 103.25 | 12.38 |
| MSA-L（息肉 Adapter, ViT-L, 880） | CVC-ClinicDB | 612 | 0.8234 ± 0.217 | 0.7432 | 0.8904 | 0.8170 | 41.18 | 12.78 |
| MedSAM+MSA（880） | Kvasir-SEG val | 120 | 0.7820 ± 0.183 | 0.6738 | 0.7725 | 0.8637 | 200.80 | 21.32 |
| MedSAM+MSA（880） | CVC-ClinicDB | 612 | 0.7143 ± 0.257 | 0.6091 | 0.7113 | 0.8103 | 101.44 | 24.01 |
| MSA zoo 非息肉探针（N/A） | Kvasir-SEG val | 5 (sanity) | 0.0097 ± 0.019 | 0.0050 | 0.0984 | 0.0051 | 434.37 | 6.76 |
| MSA zoo 非息肉探针（N/A） | CVC-ClinicDB | 612 | 0.0040 ± 0.021 | 0.0021 | 0.0121 | 0.0025 | 222.59 | 8.68 |

\*FPS 仅录自对应 json，不能当作统一测速。PP-SAM full 880 给出两次评估的 FPS（`eval_best_880shot` 8.30/8.83 与 `eval_full880` 8.62/8.70）；指标本身逐位一致。MSA-B 的 `eval_v3` Kvasir FPS 为 22.98，与上表 19.68 不同。

**出处（路径相对 `results/`）**：SAM `sam_v3/`（及 `vit_l/`、`vit_h/`）；MedSAM `medsam_v3/`；HQ-SAM `hq_sam_baseline_v3/`；SAM3 `sam3_baseline_v3/`；SAM-Med2D `sammed2d_baseline_v3/`；MobileSAM `mobile_sam_v3/`；SAMUS `phase6_samus_v3/`；SAM2.1 `sam2_baseline_v3/`（含 `hiera_base_plus/`、`hiera_large/`）；PP-SAM `pp_sam_baseline_v3/eval_best_{1,5,10,880}shot/`；SAMed `samed_baseline_cloud/samed_baseline/r4_*_best/`；MedSAM+LoRA `medsam_peft/r4_full_best/`；MSA-B `msa_polyp_baseline_v3/`；MSA-L `msa_vitl_polyp_baseline_v3/`；MedSAM+MSA `msa_medsam_polyp_v3/`；MSA zoo `msa_v3/`。

### 1.2 非 box-prompt / 协议不同（不可与 §1.1 直接排名）

| 模型 | 数据集 | N | Dice ↑ | IoU ↑ | Precision | Recall | HD95 ↓ | FPS* | 协议 |
|---|---|---|---|---|---|---|---|---|---|
| Chen SAM-Adapter（880） | Kvasir-SEG val | 120 | 0.8378 ± 0.211 | 0.7629 | 0.8615 | 0.8659 | 106.34 | 8.61 | 无 box（automatic） |
| Chen SAM-Adapter（880） | CVC-ClinicDB | 612 | 0.7576 ± 0.278 | 0.6744 | 0.7688 | 0.8389 | 66.50 | 9.28 | 无 box（automatic） |
| ASPS-v3（Kvasir 880 重训） | Kvasir-SEG val | 120 | 0.9006 ± 0.142 | 0.8412 | 0.9082 | 0.9240 | 82.25 | 7.27 | dense / 无 box |
| ASPS-v3（Kvasir 880 重训） | CVC-ClinicDB | 612 | 0.8640 ± 0.174 | 0.7898 | 0.8887 | 0.8840 | 32.68 | 7.32 | dense / 无 box |
| 发布版 ASPS（PraNet 测试集，可用） | PraNet-Kvasir | 100 | 0.9093 ± 0.149 | 0.8572 | 0.9370 | 0.9028 | 80.46 | 7.06 | 非 v3 划分 |
| 发布版 ASPS（PraNet 测试集，可用） | PraNet-CVC | 62 | 0.9129 ± 0.156 | 0.8632 | 0.9269 | 0.9293 | 14.21 | 6.96 | 非 v3 划分 |
| 发布版 ASPS（禁用） | Kvasir val 120 | 120 | 0.9629 | 0.9303 | 0.9626 | 0.9653 | 56.44 | 7.22 | **test-on-train** |
| 发布版 ASPS（禁用） | CVC 612 | 612 | 0.9564 | 0.9211 | 0.9598 | 0.9581 | 8.64 | 7.36 | **test-on-train** |
| OP-SAM CPG（1-shot, training-free） | Kvasir-SEG val | 120 | 0.4373 ± 0.302 | 0.3327 | 0.4583 | 0.6015 | 212.17 | 2.48 | 无 box |
| OP-SAM CPG（1-shot, training-free） | CVC-ClinicDB | 612 | 0.3719 ± 0.323 | 0.2847 | 0.3573 | 0.5610 | 110.75 | 2.63 | 无 box |

出处：Chen `sam_adapter_polyp_v3/`；ASPS-v3 `asps_baseline_v3/eval_best/`；发布版 ASPS `asps_baseline/`；OP-SAM `opsam_baseline/`。ASPS-v3 的 `val.txt` 兼作选 best 与报告，存在轻度乐观偏差。四配置 OP-SAM 消融见后文 P1 回填与 `opsam_ablation/`。

### 1.3 Shot 曲线与同数据量对比（已有正式结果，不是新跑）

**PP-SAM 1-shot vs 零样本**（Kvasir-val 120 + CVC 612，固定外扩 30）：
- Kvasir: vs SAM ViT-B +5.0 pp（0.7497→0.7992），vs MedSAM +9.6 pp（0.7034→0.7992）
- CVC：vs SAM / MedSAM 均约 +23.4 pp（0.5182 / 0.5172 → 0.7516）
- 跨域提升大于 in-domain。这是对照结果，不因此把 PP-SAM 写成已定的 Ours。

**PP-SAM 5-shot vs 1-shot**：Kvasir 0.7992→0.8512（+5.2 pp），CVC 0.7516→0.8040（+5.2 pp）。

**PP-SAM 10-shot vs 5-shot**：Kvasir 0.8512→0.8550（+0.38 pp），CVC 0.8040→0.7709（−3.31 pp）。当时在 1/5/10-shot 区间曾称 5-shot 为 sweet spot；**该结论只适用于这一局部阶梯**，已被 full 880 修正。

**PP-SAM full 880**（2026-10-08）：best @ epoch 61/100；`checkpoints/pp_sam/pp_sam_vit_b_880shot_best.pth`。Kvasir 0.9226 / CVC 0.8576。相对 5-shot：in-domain +7.1 pp，跨域 +5.4 pp。10-shot 的 CVC 退化只是局部异常，不是「更多数据必然跨域变差」。

同为 880 张标注的 box-prompt 对照：

| 方法（同 880 张，box 固定外扩 30） | Kvasir Dice | CVC Dice |
|---|---|---|
| SAMed LoRA rank-4 full 880 | 0.8955 | 0.8608 |
| MedSAM+LoRA r4 full 880 | 0.9049 | 0.8476 |
| MSA-B | 0.8524 | 0.7685 |
| MSA-L | 0.8555 | 0.8234 |
| MedSAM+MSA | 0.7820 | 0.7143 |
| PP-SAM full 880（全 encoder + prompt 微调，对照） | 0.9226 | 0.8576 |

- in-domain：PP-SAM full 880 高于 SAMed（+2.7 pp）和 MedSAM+LoRA（+1.8 pp）。
- 跨域：SAMed 0.8608、PP-SAM 0.8576、MedSAM+LoRA 0.8476，前两者接近（差 0.3 pp，且没有多种子检验）。
- MedSAM+MSA 明显低于 MSA-B 与 MedSAM+LoRA，可作为负向 / 结构对照，不要写成「接上 MedSAM 再加 Adapter 就一定更好」。
- 不要再用「880 张 LoRA vs 5 张 PP-SAM」这种不对称比较下结论。
- ASPS-v3（无 box）Kvasir 0.9006 / CVC 0.8640 只能作协议不同的参考，不能直接排进 §1.1。

PP-SAM 训练时间（历史记录，未重测）：1-shot 83.2 min，5-shot 51.9 min，10-shot 54.3 min，full 880 约 8.5 h。

**SAMed LoRA r4**：路径见上表。K-shot 阶段 CVC 明显弱于同 shot 的 PP-SAM，且高 Recall、低 Precision；full 880 后 Precision / HD95 回升。K-shot 曲线在 10-shot 下降，不能把 10-shot 当成单调增益。

### 1.4 2026-10-12 起补入主表的方法（摘要）

**HQ-SAM ViT-B zero-shot**（2026-10-13）：`results/hq_sam_zeroshot_v3.md`。Kvasir 0.8007 / CVC 0.6525，高于 SAM ViT-B 与 SAM3，低于 SAM ViT-L 的 Kvasir 0.8140 / CVC 0.6588。

**SAM3 zero-shot**（2026-10-12，已并入主表，不再 pending）：`results/sam3_zeroshot_v3.md`。Kvasir 0.7661 / CVC 0.6287。Prompt 为 SAM-1 式 `predict_inst` box XYXY，不是 text/PCS。

**MSA-B / MSA-L**（2026-10-13/14）：息肉域 Adapter 已在 Kvasir 880 上训练并按 v3 评估，详见 `msa_polyp_zeroshot_or_trained_v3.md`、`msa_vitl_polyp_trained_v3.md`。这不再是 future work。官方 zoo 的非息肉权重探针仍接近 0，只说明「现成跨域权重不可用」，不能代替息肉 Adapter 主结果。

**MedSAM+LoRA r4 full 880**（2026-10-15）：`medsam_peft/EVAL_R4_FULL_BEST.md`。相对零样本 MedSAM：Kvasir Dice +0.2015，CVC +0.3304。相对 SAMed full 880：Kvasir +0.0094，CVC −0.0132。

**Chen SAM-Adapter**（2026-10-16）：`sam_adapter_polyp_v3/`。这是 tianrun-chen 的 SAM-Adapter，**不是** Medical SAM Adapter。评估为方法原生 automatic prompt，**无 GT box**，故放在 §1.2。

**MedSAM+MSA**（2026-10-16）：在 MSA 流水线中以 MedSAM 权重初始化。Kvasir 0.7820 / CVC 0.7143，低于 MSA-B 与 MedSAM+LoRA。

**不要用的旧 SAM2 叙述**：有的历史段落仍写 B7 Kvasir 0.6509 或 CVC 0.3625。0.6509 已被协议对齐后的 0.7738 覆盖。`phase5_sam2_v3` 的 CVC 0.3625 与正式 `sam2_baseline_v3` 的 0.5378 冲突，**主表只用后者**。

**不要用的旧 SAMUS 叙述**：同一文件旧版曾在后文写 Kvasir 0.6778 / CVC 0.5124，并据此说「跨域比 SAM 更稳」。正式 json 是 0.6024 / 0.3516，跨域差 25.1 pp，并不比 SAM ViT-B 的 23.2 pp 更稳。以 json 为准。

### 1.5 Prompt 消融：固定外扩 0 / 10 / 30 / 50（2026-11-17）

完整汇总与路径：`results/prompt_ablation_box_expand_SUMMARY.md`。协议是 GT 紧框 + 固定外扩，**不是** random。未改正式 `summary.json`。

| 模型 | 最优点（本表） | box 30（与正式主表对齐情况） | 备注 |
|---|---|---|---|
| MedSAM+LoRA r4 | Kvasir 紧框 0.9269 / 0.8712；CVC box 10 为 0.8720 / 0.7959（仅比紧框高 0.18 个点） | Kvasir 本机重跑与 `r4_full_best` 一致（0.9049 / 0.8375）；CVC 用用户数，且 box 30 与正式 0.8476 / 0.7596 一致 | CVC 0/10/50 用户提供，**本机未重跑**；无对应本机 json |
| PP-SAM full 880 | 两端均为 box 10：0.9280 / 0.8720 与 0.8948 / 0.8221 | 与正式 880-shot 完全一致 | 本机重跑 |
| SAMed r4 full | Kvasir 紧框 0.9039 / 0.8419；CVC box 10 为 0.8729 / 0.8012 | CVC 与正式几乎逐位一致；**Kvasir 本机 0.8951 vs 正式 0.8955**（差约 0.0003） | 正式主表仍用云端 0.8955，未改 json |
| MSA-B | 变化相对最小；Kvasir box 10 为 0.8573 / 0.7871，CVC 紧框 0.7894 / 0.7082 | 与正式 MSA-B 一致 | 本机重跑 |

阅读要点：现行 v3 默认 ±30 **不是**这四个模型的峰值；紧框或 box 10 更好，CVC 对松框更敏感。不要写成单一最优点（MedSAM 在 Kvasir 上随框变松单调下降，与 CVC 上 box 10 略高并不完全一样）。

**尚未做的消融**：零样本 SAM-L、零样本 MedSAM 的 0/10/30/50。也还没有折线图。

---

<!-- P1 回填保留；ASPS-v3 / OP-SAM 详细表。数字已与当时 json 对齐。 -->

<!-- BEGIN P1 BACKFILL (generated by code/asps_baseline/report_p1.py) -->

### B8 ASPS — v3-aligned retraining

- Checkpoint: `results/asps_baseline_v3/asps_v3_vitb/checkpoints/asps_v3_best.pth` — trained on Kvasir `train.txt` 880 only; `val.txt` 120 was used for best-epoch selection and is never a training sample.
- ⚠️ **Protocol**: ASPS is **dense / self-prompting** — it does **not** consume the v3 ±30 px GT-derived box prompt (`Prompt Type = none`). These numbers are therefore **not directly rankable** against the box-prompt baselines; report the row the same way as B10 OP-SAM.
- ⚠️ **Comparability vs the authors' number**: the paper trains on the PraNet split (Kvasir 900 + **CVC 550**). We train on Kvasir 880 only, so CVC-612 is a genuinely unseen cross-domain set and is expected to be **lower** than the paper's 0.9281. Quote the paper number only as "reported by the authors".

| ASPS-v3 entry | N | Dice | IoU | Precision | Recall | HD95 | FPS | Prompt Type |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| ASPS-v3 — Kvasir val.txt 120 | 120 | 0.9006 | 0.8412 | 0.9082 | 0.9240 | 82.25 | 7.27 | none (dense) |
| ASPS-v3 — CVC-ClinicDB 612 | 612 | 0.8640 | 0.7898 | 0.8887 | 0.8840 | 32.68 | 7.32 | none (dense) |

**External reference (NOT v3-aligned; kept for continuity)** — the released `B-L-0015200.pth`, trained on the PraNet split:

| released ASPS ckpt | N | Dice | IoU | Precision | Recall | HD95 | FPS | Prompt Type |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| released — PraNet-Kvasir held-out 100 | 100 | 0.9093 | 0.8572 | 0.9370 | 0.9028 | 80.46 | 7.06 | none (dense) |
| released — PraNet-CVC held-out 62 | 62 | 0.9129 | 0.8632 | 0.9269 | 0.9293 | 14.21 | 6.96 | none (dense) |
| released — Kvasir val 120 (**test-on-train, inflated — do not use**) | 120 | 0.9629 | 0.9303 | 0.9626 | 0.9653 | 56.44 | 7.22 | none (dense) |
| released — CVC 612 (**test-on-train, inflated — do not use**) | 612 | 0.9564 | 0.9211 | 0.9598 | 0.9581 | 8.64 | 7.36 | none (dense) |

### B10 OP-SAM — full-pipeline ablation (CPG / EPE / SPF / SPF+EPE)

- ⚠️ **Protocol**: one-shot, training-free and **prompt-free** — it conditions on ONE support image (`Kvasir-SEG/Data-split/train.txt[0]`, the same single label as PP-SAM 1-shot) and generates its own SAM2 prompts. It does **not** consume the v3 ±30 px box prompt (`Prompt Type = none`).
- Per-image paired comparison (which images each component wins/loses on) is in `results/opsam_ablation/ABLATION.md`.

| dataset | mode | N | Dice | IoU | Precision | Recall | HD95 | FPS |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Kvasir-SEG-val | cpg | 120 | 0.4373 | 0.3327 | 0.4583 | 0.6015 | 212.17 | 2.47 |
| Kvasir-SEG-val | epe | 120 | 0.3712 | 0.2846 | 0.3045 | 0.9363 | 310.49 | 2.10 |
| Kvasir-SEG-val | spf | 120 | 0.4440 | 0.3404 | 0.4629 | 0.6112 | 211.00 | 0.84 |
| Kvasir-SEG-val | spf+epe | 120 | 0.3823 | 0.2952 | 0.3080 | 0.9529 | 308.67 | 0.79 |
| CVC-ClinicDB | cpg | 612 | 0.3719 | 0.2847 | 0.3573 | 0.5610 | 110.75 | 2.57 |
| CVC-ClinicDB | epe | 612 | 0.2838 | 0.2077 | 0.2146 | 0.9105 | 172.89 | 2.47 |
| CVC-ClinicDB | spf | 612 | 0.3762 | 0.2879 | 0.3634 | 0.5639 | 110.28 | 0.86 |
| CVC-ClinicDB | spf+epe | 612 | 0.2889 | 0.2121 | 0.2192 | 0.9130 | 172.23 | 0.84 |
| Kvasir-SEG-val | cpg (v1, 2026-06-23) | 120 | 0.4373 | 0.3327 | 0.4583 | 0.6015 | 212.17 | 2.48 |
| CVC-ClinicDB | cpg (v1, 2026-06-23) | 612 | 0.3719 | 0.2847 | 0.3573 | 0.5610 | 110.75 | 2.63 |

**Cost (measured 2026-09-30, so that nothing has to be dropped)**: OP-SAM `cpg` takes 2.6 min wall on Kvasir-120 and 5.1 min on CVC-612 including model load (0.38-0.41 s/img); SPF is ≈3.3× per image. The whole 4-mode × 2-dataset sweep is ≈1.2 h, so **all four configurations are reported on both datasets** — a MISSING row above means the run has not landed yet, not that it was skipped for cost.

<!-- END P1 BACKFILL -->

> 2026-11-17：§2 保留历史实验流水账（含已废弃的 v2 / cherry-pick / 烟测）。**抄正式数字请用 §1 与 `docs/NUMBERS_INDEX.md`，不要从本节约旧叙述里摘冲突数字。** 2026-10 月以后的 HQ-SAM / SAM3 / MSA-B·L / MedSAM+LoRA / Chen / MedSAM+MSA 以及 2026-11-17 的 prompt 消融，以 §1 为准。

## 2. 全部已跑实验 (raw)

### Phase 0 — Vanilla SAM ViT-B (zero-shot baseline)

| 实验 | 数据 | N | Dice | 备注 |
|---|---|---|---|---|
| `phase0_sam_smoke` | CVC | 5 | 0.9465 | 烟雾测试 |
| `phase0_sam_50` | CVC | 50 | 0.9526 | **cherry-pick, 0.86~0.99** |
| `phase0_sam_kvasir_50` | Kvasir | 50 | 0.7881 | 部分 train 泄漏 |
| **`phase0_sam_v3` (新协议)** | Kvasir-val | 120 | **0.7497** | **val.txt, ±30px, 当前** |
| **`phase0_sam_v3` (新协议)** | CVC | 612 | **0.5182** | **val.txt, ±30px, 当前** |

### Phase 1 — MedSAM (medical-fine-tuned ViT-B)

| 实验 | 数据 | N | Dice | 备注 |
|---|---|---|---|---|
| `phase1_medsam_smoke` | CVC | 5 | 0.6570 | 烟雾测试 |
| `phase1_medsam_cvc_50_v2` | CVC | 50 | 0.7302 | v2 (split OK) |
| `phase1_medsam_kvasir_50_v2` | Kvasir | 50 | 0.7798 | v2 |
| `medsam_new_protocol` | Kvasir-val | 120 | 0.7034 | 旧协议, 等价 v3 (字节级一致) |
| **`phase1_medsam_v3` (新协议)** | Kvasir-val | 120 | **0.7034** | **val.txt, ±30px, 当前** |
| **`phase1_medsam_v3` (新协议)** | CVC | 612 | **0.5172** | **val.txt, ±30px, 当前** |

### Phase 2 — MSA 官方 zoo 非息肉探针（历史记录；息肉 MSA-B/L 见 §1.1，不再是未做）

| 实验 | 数据 | N | Dice | 备注 |
|---|---|---|---|---|
| `phase2_msa_smoke` | CVC | 5 | 0.9465 | 等同 vanilla SAM (adapter 未加载?) |
| `phase2_msa_cvc_50_v2` | CVC | 50 | **0.0064** | **N/A — 领域权重不匹配** |
| `phase2_msa_kvasir_50_v2` | Kvasir | 50 | **0.0245** | **N/A — 领域权重不匹配** |
| **`phase2_msa_v3` (sanity 5 张)** | Kvasir-val | 5 | 0.0097 | 验证 v3 脚本工作正常, 仍 N/A |

**MSA 官方 zoo 探针状态**：仓库发布权重只有 4 个领域 (Lung / Melanoma / Retina / Surgical-Instrument)，**没有**现成的息肉权重。在 Kvasir/CVC 上 Dice 接近 0 是 **expected behavior**，不是实现 bug。这一行仍只作负向背景。

**2026-10 更新（重要）**：息肉域 MSA-B / MSA-L 已在 Kvasir train 880 上训练，并按 v3 评估（见 §1.1 与 `results/msa_polyp_zeroshot_or_trained_v3.md`、`results/msa_vitl_polyp_trained_v3.md`）。**不能再把息肉 Adapter 写成未做或 future work。** 论文里应区分「官方 zoo 非息肉探针（N/A）」与「本机训练的 MSA-B/L（正式结果）」。

### Phase 4 — PP-SAM (Prompt-perturbation SAM, 1-shot / 5-shot / 10-shot / full-880 fine-tune)

| 实验 | 数据 | N | Dice | 备注 |
|---|---|---|---|---|
| **`pp_sam_v3/train_1shot` (新协议, 重训)** | Kvasir-val (val.txt 120) | 120 | **0.7992** | epoch 3 / 100 best (过拟合 onset ep 7)，train sample 从 train.txt 880 |
| `train_ppsam_100ep` (旧 v2, 已废弃) | Kvasir (1000 张 full) | 1000 | 0.7827 → 0.8432 | **in-distribution, 含 val leak** — 不再使用 |
| **`pp_sam_v3/eval_best_1shot`** | Kvasir-val | 120 | **0.7992** | best.pth 重测 (held-out, 干净) |
| **`pp_sam_v3/eval_best_1shot`** | CVC (跨域) | 612 | **0.7516** | best.pth 重测 (跨域, 干净) |
| **`pp_sam_v3/train_5shot` (新协议, 重训)** | Kvasir-val (val.txt 120) | 120 | **0.8512** | epoch 48 / 100 best, train sample 从 train.txt 880 random |
| **`pp_sam_v3/eval_best_5shot`** | Kvasir-val | 120 | **0.8512** | best.pth 重测 (held-out, 干净) |
| **`pp_sam_v3/eval_best_5shot`** | CVC (跨域) | 612 | **0.8040** | best.pth 重测 (跨域, 干净) |
| **`pp_sam_v3/train_10shot` (新协议, 2026-06-21)** | Kvasir-val (val.txt 120) | 120 | **0.8550** | epoch 22 / 100 best (过拟合 onset ep 23), train sample 从 train.txt 880 random |
| **`pp_sam_v3/eval_best_10shot`** | Kvasir-val | 120 | **0.8550** | best.pth 重测 (held-out, 干净) |
| **`pp_sam_v3/eval_best_10shot`** | CVC (跨域) | 612 | **0.7709** | best.pth 重测 (跨域, 干净) — 比 5-shot 退化 -3.3 pp |
| **`pp_sam_baseline/train_880shot` (新协议, 2026-10-07 21:56 → 10-08 06:27)** | Kvasir-val (val.txt 120) | 120 | **0.9226** | epoch 61 / 100 best, train sample = train.txt **全部 880 张**（不再是 random K 张）; 墙钟 511.6 min |
| **`pp_sam_baseline/eval_full880`** | Kvasir-val | 120 | **0.9226** | best.pth 重测 (held-out, 干净) — 与训练期 best val 相同 |
| **`pp_sam_baseline/eval_full880`** | CVC (跨域) | 612 | **0.8576** | best.pth 重测 (跨域, 干净) — 逆转 10-shot 的 CVC 退化 (+8.7 pp); HD95 27.72 为 **CVC-612 各行最低**（ASPS-v3 32.68 / SAMed full 880 32.75 / PP-SAM 5-shot 39.47）。§1 里更低的 14.21（ASPS, PraNet-CVC 62 张）与 8.64（test-on-train 虚高）口径不同，不可比 |

**5-shot 训练时间**: 51.9 min (vs 1-shot 83.2 min — 5-shot 因为每 epoch 训 5 张耗时多但收敛快反而短)
**10-shot 训练时间**: 54.3 min (best @ epoch 22, 之后严重过拟合)
**880-shot / full-data 训练时间**: 511.6 min ≈ 8.5 h (100 epoch × 880 张/epoch, batch=1) — 本项目最长的单次 PP-SAM 训练
**880-shot 代码改动 (2026-10-07)**: `code/pp_sam_baseline/train_ppsam.py:387-393` 把预载图像改为 CPU uint8 保存。原实现将预处理后的 float32 张量常驻 GPU（≈12 MB/张），1/5/10-shot 无碍，但 880 张 ≈ 10.3 GB → 16 GB 卡 OOM。改动后显存占用与 shot 数无关；dtype 语义不变（`preprocess` 本来就接收 uint8 张量）。除该处外，超参与 1/5/10-shot 完全一致（同一脚本、同一 box 扰动 0–50 px、同一 AdamW lr=1e-4）。


### B-SAMed — SAMed LoRA rank-4 PEFT (cloud, 2026-06-25)

| 实验 | 数据 | N | Dice | 备注 |
|---|---|---|---|---|
| **`samed_cloud/r4_1shot_best`** | Kvasir-val | 120 | **0.7456** | 1-shot LoRA, high-recall / low-precision |
| **`samed_cloud/r4_1shot_best`** | CVC | 612 | **0.5759** | 跨域明显弱于 PP-SAM 1-shot |
| **`samed_cloud/r4_5shot_best`** | Kvasir-val | 120 | **0.7571** | 比 1-shot 小涨 |
| **`samed_cloud/r4_5shot_best`** | CVC | 612 | **0.5747** | 基本不涨 |
| **`samed_cloud/r4_10shot_best`** | Kvasir-val | 120 | **0.7230** | K-shot LoRA 退化 |
| **`samed_cloud/r4_10shot_best`** | CVC | 612 | **0.5435** | K-shot LoRA 退化 |
| **`samed_cloud/r4_full_best`** | Kvasir-val | 120 | **0.8955** | full 880 LoRA, v3-aligned |
| **`samed_cloud/r4_full_best`** | CVC | 612 | **0.8608** | full 880 LoRA, cross-domain strong |

**状态**: 云端 RTX 4090D 训练完成，正式结果已复制到 `results/samed_baseline_cloud/`。full 初始 `lr=0.005` 曾发散，修正为 `lr=0.0005` 后稳定得到 r4_full_best。

### B6 SAMUS (zero-shot)
- 仓库: `材料/repos/SAMUS/`；权重: `checkpoints/samus/SAMUS.pth`（US30K ultrasound 预训练）
- 正式结果：**只以** `results/phase6_samus_v3/summary.json` 为准。
  - Kvasir-val 120: Dice **0.6024** ± 0.199，IoU 0.4571，P 0.5323，R 0.7456，HD95 138.40，FPS 20.53
  - CVC 612: Dice **0.3516** ± 0.215，IoU 0.2348，P 0.3021，R 0.4863，HD95 87.19，FPS 20.08
- 本段旧稿曾写 Kvasir 0.6778 / CVC 0.5124（Recall 0.95 / 0.93，跨域差 -16.5 pp，「比 SAM 更稳」）。这些数 **与 json 冲突，已废弃**。按 json，跨域差为 25.1 pp，并不比 SAM ViT-B 的 23.2 pp 更稳。
- FPS 与 SAM 的倍数不要按旧文「20.2 vs 8.05 = 2.5x」直接写进论文；SAM 正式 json 为 6.92 / 7.14，且 FPS 口径不统一，需重测。input 分辨率 256 vs 1024 可以作为方法差异说明。
- 结果目录: `results/phase6_samus_v3/`

### B3 MSA 官方 zoo 非息肉探针（负向背景，不是息肉主结果）
- 仓库: `材料/repos/Medical-SAM-Adapter/`，探针权重如 `Surgical_Instrument_Video_SAM_1024.pth`
- 官方 zoo 仍无息肉权重，这一探针接近 0 是 expected behavior
- **不要**再写「论文用 MedSAM 或 SAMUS 替代 MSA 主结果」。息肉域 MSA-B/L 已有正式结果（见 §1.1）

## 3. 关键发现（2026-11-17 按正式 json 重写，去掉已过时表述）

1. **协议已经统一，但旧稿和旧表不能用。** 正式评估是 Kvasir val 120 + CVC 612 + GT 紧框固定外扩（默认 30）。6 月 `manuscript/tables/` 与 Partial Thesis 仍是旧数，且把扰动写成 random、把 PP-SAM 写成 ours，不能沿用。
2. **零样本主线可以写。** 同一协议下，SAM ViT-B 为 0.7497 / 0.5182，MedSAM 为 0.7034 / 0.5172（in-domain 低于 SAM，跨域几乎打平，但 MedSAM 更高 Recall、更低 Precision）。HQ-SAM ViT-B 为 0.8007 / 0.6525；SAM3 为 0.7661 / 0.6287；SAM ViT-L 零样本仍是最强零样本之一（0.8140 / 0.6588）。
3. **Adapter / LoRA 主线可以写。** 同为 880 张、同一 box 协议：MedSAM+LoRA 0.9049 / 0.8476，SAMed 0.8955 / 0.8608，MSA-B 0.8524 / 0.7685，MSA-L 0.8555 / 0.8234。息肉域 MSA 已完成，不再是 future work。官方 zoo 非息肉探针仍接近 0，只作负向背景。
4. **PP-SAM 可作为全参 / shot 曲线对照，但不是已定的 Ours。** 1/5/10/full 都是正式结果。5-shot 只在 1/5/10 阶梯里像甜区；加到 880 后 in-domain 与跨域都继续升。同数据量下 in-domain 高于 LoRA，跨域与 SAMed 接近。
5. **MedSAM+MSA 是负向对照**：0.7820 / 0.7143，低于 MSA-B 和 MedSAM+LoRA，不能写成「MedSAM 初始化加 Adapter 必然更好」。
6. **SAMUS 以 json 为准**：0.6024 / 0.3516，跨域差 25.1 pp。不要再用旧叙述 0.6778 / 0.5124，也不要据此写「比 SAM 跨域更稳」。
7. **SAM2.1 正式数字是 `sam2_baseline_v3`**：Hiera-S 为 0.7738 / 0.5378。不要用已覆盖的 0.6509，也不要用 `phase5_sam2_v3` 的 CVC 0.3625。
8. **Prompt 敏感性已有部分证据，但不是完整章程分析。** 四个已训模型上，默认 ±30 都不是峰值；尚未做零样本 SAM-L / MedSAM 消融，也没有折线图。object-scale、弱边界系统分析、多种子：**未做**。A0/A1/B1 只有 5-shot 烟测，不能进正式表。
9. **Chen SAM-Adapter、ASPS、OP-SAM 协议不同**，可写进论文，但必须标明无 box / dense / prompt-free，不能与 §1.1 直接排名。
10. **FPS 不能当已经统一的效率结论来用。** 旧文常见「SAM ≈ 8 FPS、SAMUS 快 2.5×」混用了不同来源；现正式 json 中 SAM ViT-B 为 6.92 / 7.14。要写效率，先重测。

历史细节（PP-SAM 过拟合 onset、OP-SAM 失败模式、ASPS-v3 训练曲线等）仍见 §2 与 P1 回填；引用时核对是否与上表冲突。

---

## 4. 论文图表对应

| 论文位置 | 现在应引用 | 不要用 |
|---|---|---|
| 方法 / 协议 | 本文件 §0，`docs/NUMBERS_INDEX.md` | 6 月稿里把评估写成 random perturbation 的段落 |
| Table 1（零样本 + Adapter/LoRA + 跨域） | 本文件 §1.1，数字回溯到对应 `summary.json` | `manuscript/tables/table1_sota.tex` |
| Table 2（shot 曲线） | 本文件 §1.3 与 PP-SAM / SAMed 的 `eval_best_*` json | `manuscript/tables/table2_ablation.tex`（缺 full-880，且是旧 1/5-shot） |
| Prompt 消融 | `results/prompt_ablation_box_expand_SUMMARY.md` | 把固定外扩写成 random；把未做的零样本消融写成已完成 |
| 定性图 | 现有 `manuscript/figures/` 与 `results/figure3_qualitative/` 仍是 6 月图，**不是**正式权重下的新图 | 不要把旧图写成当前正式权重的定性结果 |
| object-scale / 弱边界 / 多种子 | — | 任何「已完成」表述 |

`manuscript/README_NUMBERS.md` 也有同样的旧表警告。本章正文尚未重写。

---

## 5. 后续工作（TODO）

### 已完成（实验主线，到 2026-11-17）

- [x] v3 协议下的零样本 SAM / MedSAM，以及 ViT-L/H、MobileSAM、SAM-Med2D、SAMUS、SAM2.1、HQ-SAM、SAM3
- [x] PP-SAM 1/5/10/full 880（正式评估）
- [x] SAMed LoRA r4 1/5/10/full 880
- [x] MedSAM+LoRA r4 full 880
- [x] 息肉域 MSA-B、MSA-L
- [x] MedSAM+MSA（负向对照）
- [x] Chen SAM-Adapter（无 box，正式 json 已有）
- [x] ASPS-v3 重训与 OP-SAM 四配置消融
- [x] 四个已训模型的 box 0/10/30/50 固定外扩消融（2026-11-17）
- [x] 本机台账与「数字从哪抄」索引（2026-11-17；GitHub / Notion 尚未同步）

### 仍未完成（不要写成已经做完）

- [ ] **object-scale**（按面积分档）系统分析
- [ ] **弱边界**系统定量 / 定性（A1 5-shot 烟测不能代替）
- [ ] **多种子与显著性检验**
- [ ] Prompt 消融补零样本 SAM-L、零样本 MedSAM，并做折线图
- [ ] 正式权重下的新定性图（替换 6 月旧图）
- [ ] **FPS 空闲 GPU 统一重测**
- [ ] 点 prompt 正式表
- [ ] 是否把 PP-SAM 写作 Ours：待定
- [ ] 重写论文章节，并用正式 json 重做 LaTeX 表（旧 `table1` / `table2` 仅作过时警告保留）
- [ ] Prompt generation toolkit 产品化
- [ ] 扩展项：伪标签半监督；PolypGen / ETIS / ColonDB（本机无这些数据）

### 阻塞 / 外部依赖

- [ ] PolypGen / ETIS-Larib / ColonDB 跨域 — **本机无数据**，需另行下载；建议在主人在场时手动获取。旧链接不要想当然可用。

---

## 附录 A. 2026-11-17 数字校正备忘

| 项目 | 旧台账 | 现以 json 为准 |
|---|---|---|
| B1 CVC Dice ± | 0.5182 ± 0.191 | 0.5182 ± 0.363（`sam_v3`） |
| B2 CVC Dice ± | 0.5172 ± 0.175 | 0.5172 ± 0.211（`medsam_v3`） |
| B3 CVC Dice ± | 0.0040 ± 0.011 | 0.0040 ± 0.021（`msa_v3`） |
| SAM2.1 Hiera-S Kvasir ± | 0.7738 ± 0.213 | 0.7738 ± 0.218 |
| B1 FPS | 8.13 / 8.37（来源不明） | 6.92 / 7.14（`sam_v3`） |
| B2 FPS | 8.03 / 8.38 | 6.91 / 7.15（`medsam_v3`） |
| PP-SAM 1-shot FPS | 8.05 / 8.32 | 6.54 / 7.34 |
| SAMUS 叙述 | 曾写 0.6778 / 0.5124 | 0.6024 / 0.3516 |
| SAM3 | pending，未进主表 | 已并入 §1.1 |
| PP-SAM 标签 | 多处写 ours | 已去掉；是否 Ours 未定 |

备份：`backup/ledger_update_20261117/EXPERIMENT_SUMMARY.md`。正式 `summary.json` 未被改动。
