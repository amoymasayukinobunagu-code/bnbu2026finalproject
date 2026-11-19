# MedSAM+LoRA CVC box 0/10/30/50 — 用户提供，本机未重跑

**醒目标记：CVC 四档中的 0/10/50 没有本机 `summary.json`。**  
数字由用户于 2026-11-17 提供。本机只重跑了同一权重的 **Kvasir-120** 四档。

| box | CVC-612 Dice / IoU | 来源 |
|---|---|---|
| 0 | 0.8702 / 0.7980 | 用户提供，本机未重跑，无本地 json |
| 10 | 0.8720 / 0.7959 | 用户提供，本机未重跑，无本地 json |
| 30 | 0.8476 / 0.7596 | 用户提供，且与正式 `medsam_peft/r4_full_best`（本目录 `numbers/official_core/medsam_peft/r4_full_best/summary.json`）一致 |
| 50 | 0.8134 / 0.7151 | 用户提供，本机未重跑，无本地 json |

Kvasir 四档是本机新跑，原始 json 在 `numbers/ablation_box_expand/medsam_peft/prompt_ablation_box_expand/kvasir_box{0,10,30,50}/`。

写论文时必须保留这一来源区别，不能把 CVC 0/10/50 写成「本机正式重跑」。
