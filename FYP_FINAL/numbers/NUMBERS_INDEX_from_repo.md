# 数字从哪抄（2026-11-17，Asia/Shanghai）

写论文或做表时，**只从本机正式 `summary.json` 和已核对的消融汇总抄数字**。  
`results/EXPERIMENT_SUMMARY.md` 是整理后的可读台账，但若与 json 冲突，**以 json 为准**。

本文件只做索引，不替代正式结果文件。不要开训练、不要改 `summary.json`。

---

## 1. 应该抄哪里

| 用途 | 权威来源 |
|---|---|
| 主表 / shot 曲线 / 负结果 / 跨域 | 下表所列正式 `results/**/summary.json` |
| Prompt 框大小消融（固定外扩 0/10/30/50） | `results/prompt_ablation_box_expand_SUMMARY.md`，细节仍以各档 `summary.json` 为准 |
| 协议与限制（何为正式、何为烟测、何为旧稿） | 本文件 + `results/EXPERIMENT_SUMMARY.md` 文首 |

可读表：`E:\Project\FYP\results\EXPERIMENT_SUMMARY.md`（2026-11-17 已按 json 校正）。

---

## 2. 不要抄这些

| 来源 | 原因 |
|---|---|
| `manuscript/tables/table1_sota.tex` | 2026-06-10 旧表。数字与正式 v3 不符（含旧 1-shot），把扰动写成 random，并把 PP-SAM 标成 ours |
| `manuscript/tables/table2_ablation.tex` | 同上；shot 表仍是旧 1/5-shot，缺 full-880，且 5-shot 口径已过时 |
| `manuscript/Partial Thesis/` 与 6 月章节正文里的表 | 停在当时的零样本和部分 k-shot，不能当正式数字 |
| A0 / A1 / B1 的 5-shot 目录 | **烟测**，不能升成正式表 |
| 发布版 ASPS 在 Kvasir val 120 / CVC 612 上的 0.9629 / 0.9564 | **test-on-train，虚高，禁用** |
| `results/phase5_sam2_v3/summary.json` 的 CVC | 与正式 `sam2_baseline_v3` 的 CVC **冲突**（0.3625 vs 0.5378）。Kvasir 与正式文件字节级一致，但 CVC 以 `sam2_baseline_v3` 为准 |
| 旧版 `EXPERIMENT_SUMMARY.md`（截至 2026-10-08）里 B6 SAMUS 叙述段的 0.6778 / 0.5124 | 与 `phase6_samus_v3/summary.json`（0.6024 / 0.3516）冲突；已按 json 改正 |
| 旧 v2 协议数字（`sorted[:120]` 等） | 已废弃 |

---

## 3. 抄数字时必须带上的限制

- **评估框**：v3 默认是 GT 紧框再**固定外扩**（主表默认 30 像素），**不是** random jitter。PP-SAM **训练**仍使用随机扰动 0–50 px，不要和评估协议混写。
- **±**：测试图像间标准差（`dice_std`），**不是**多种子、也不是显著性检验。多种子尚未做。
- **FPS**：不同 json、不同复跑之间口径不一致，且部分在 GPU 非空闲时测得。论文若要报 FPS，需在空闲 GPU 上重测。Dice / IoU / P / R / HD95 不受此影响。
- **PP-SAM 是否写作 Ours：未定。** 先按 Index 1 的「零样本 SAM / 零样本 MedSAM / Adapter·LoRA」三角来写；PP-SAM 先作全参对照。
- **object-scale、弱边界系统分析、多种子：未做。** 不能写成已完成。A1 5-shot 烟测不能代替弱边界分析。
- **Prompt 消融未做**：零样本 SAM-L、零样本 MedSAM 的 0/10/30/50。已做的是 MedSAM+LoRA、PP-SAM full880、SAMed r4 full、MSA-B。
- **MedSAM+LoRA 的 CVC 四档消融**为用户 2026-11-17 提供，本机未重跑；其中 box 30 与正式 `r4_full_best` 一致（0.8476 / 0.7596）。
- **SAMed Kvasir box 30**：本机消融 0.8951，正式云端评估 0.8955（差约 0.0003）。正式主表仍用云端 `r4_full_best`，未改正式 json。

---

## 4. 正式 summary.json 路径（主表与 shot 曲线）

路径均相对 `E:\Project\FYP\results\`。

### 零样本（box-prompt，v3）

| 方法 | 正式 json |
|---|---|
| SAM ViT-B | `sam_v3/summary.json`（同指标亦见于 `phase0_sam_v3/`，FPS 略不同，主表用 `sam_v3`） |
| SAM ViT-L | `sam_v3/vit_l/summary.json` |
| SAM ViT-H | `sam_v3/vit_h/summary.json` |
| MedSAM | `medsam_v3/summary.json`（`phase1_medsam_v3/` 指标相同，FPS 略不同） |
| HQ-SAM ViT-B | `hq_sam_baseline_v3/summary.json` |
| SAM3 | `sam3_baseline_v3/summary.json` |
| SAM-Med2D | `sammed2d_baseline_v3/summary.json` |
| MobileSAM | `mobile_sam_v3/summary.json` |
| SAMUS | `phase6_samus_v3/summary.json` |
| SAM2.1 Hiera-S | `sam2_baseline_v3/summary.json`（不要用 `phase5_sam2_v3` 的 CVC） |
| SAM2.1 Hiera-B+ | `sam2_baseline_v3/hiera_base_plus/summary.json` |
| SAM2.1 Hiera-L | `sam2_baseline_v3/hiera_large/summary.json` |

### Adapter / LoRA / 全参微调（在 Kvasir train 880 上训练，除非另行注明 shot 数）

| 方法 | 正式 json | 备注 |
|---|---|---|
| PP-SAM 1 / 5 / 10-shot | `pp_sam_baseline_v3/eval_best_{1,5,10}shot/summary.json` | 不是已定的 Ours |
| PP-SAM full 880 | `pp_sam_baseline_v3/eval_best_880shot/summary.json` 与 `pp_sam_baseline/eval_full880/summary.json` | Dice 等指标逐位一致；FPS 分别为 8.30/8.83 与 8.62/8.70 |
| SAMed LoRA r4（1/5/10/full） | `samed_baseline_cloud/samed_baseline/r4_{1shot,5shot,10shot,full}_best/summary.json` | 正式主表用云端评估 |
| MedSAM+LoRA r4 full 880 | `medsam_peft/r4_full_best/summary.json` | |
| MSA-B（息肉 Adapter，ViT-B） | `msa_polyp_baseline_v3/summary.json` | `msa_polyp_train/eval_v3` 指标相同，Kvasir FPS 不同（19.68 vs 22.98） |
| MSA-L（息肉 Adapter，ViT-L） | `msa_vitl_polyp_baseline_v3/summary.json` | 与 `msa_vitl_polyp_train/eval_v3` 相同 |
| MedSAM+MSA | `msa_medsam_polyp_v3/summary.json` | 负向对照：低于 MSA-B 与 MedSAM+LoRA |

### 协议不同、不可与 box-prompt 主表直接排名

| 方法 | 正式 json | 协议 |
|---|---|---|
| Chen SAM-Adapter（tianrun-chen） | `sam_adapter_polyp_v3/summary.json` | **无 box**（方法原生 automatic visual adapter） |
| ASPS-v3（本机按 Kvasir 880 重训） | `asps_baseline_v3/eval_best/Kvasir-SEG-val/summary.json` 与 `.../CVC-ClinicDB/summary.json` | dense / self-prompting，无 box |
| 发布版 ASPS（PraNet 测试集，可用） | `asps_baseline/PraNet-Kvasir-test/summary.json`、`asps_baseline/PraNet-CVC-test/summary.json` | 非 v3 划分；val120/CVC612 是 test-on-train，禁用 |
| OP-SAM CPG | `opsam_baseline/summary.json` | one-shot、training-free、**无 box**；四配置消融见 `opsam_ablation/` |
| MSA 官方 zoo 非息肉探针 | `msa_v3/summary.json` | Kvasir 仅 sanity 5 张；不是息肉主结果 |

---

## 5. Prompt 消融 json（2026-11-17）

汇总：`results/prompt_ablation_box_expand_SUMMARY.md`

| 模型 | 目录 |
|---|---|
| MedSAM+LoRA（仅本机重跑 Kvasir） | `medsam_peft/prompt_ablation_box_expand/kvasir_box{0,10,30,50}/summary.json` |
| PP-SAM full 880 | `pp_sam_baseline/prompt_ablation_box_expand/box{0,10,30,50}/summary.json` |
| SAMed r4 full | `samed_baseline/prompt_ablation_box_expand/box{0,10,30,50}/summary.json` |
| MSA-B | `msa_polyp_train/prompt_ablation_box_expand/box{0,10,30,50}/summary.json` |

MedSAM+LoRA 的 CVC 0/10/50 **没有**本机 json（用户提供）；CVC box 30 以正式 `medsam_peft/r4_full_best` 为准。

本机 FYP 没有 `docs/experiments/prompt_ablation_v3/SUMMARY.md`（该模板在 GitHub 克隆里）。稍后若同步远程，应改那份模板，而不是继续留空表。

---

## 6. 烟测（不能进正式表）

包括但不限于：

- `checkpoints/` 与 `results/` 下的 `a0_structure_5shot`、`a1_edge_5shot`、`b1_cnn_5shot`、`r4_smoke`、`polyp_smoke`
- `opsam_baseline_smoke*`（小 N 预实验）
- `mobile_sam_v3/sanity5`、`msa_v3` 的 Kvasir 5 张、以及更早的 phase smoke / 50 张 cherry-pick

这些目录可以在附录说明「只做了烟测」，但数字不得写入正式对比表。

---

*索引日期：2026-11-17。若之后产生新的正式评估，先改对应 `summary.json`，再改本索引与 `EXPERIMENT_SUMMARY.md`。*
