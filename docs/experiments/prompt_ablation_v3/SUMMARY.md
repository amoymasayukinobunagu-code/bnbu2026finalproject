# prompt_ablation_v3 SUMMARY（模板）

> 协议 v3：Kvasir `val.txt` 120 + CVC-ClinicDB 612，GT box ±N px。不训新权重。
> 目标路径：`E:\Project\FYP\results\prompt_ablation_v3\SUMMARY.md`
> ±30 列已从 Notion 总览 / 1016 阶段说明抄入；0 / 10 / 50 跑完后填空。`—` = 待测。

## Kvasir-120 mean Dice

| 模型 | ±0 | ±10 | ±30（锚点） | ±50 |
|---|---|---|---|---|
| SAM ViT-L（0-shot） | — | — | 0.8140 | — |
| MedSAM（0-shot） | — | — | 0.7034 | — |
| SAMed LoRA full-880 | — | — | 0.8955 | — |
| MedSAM+LoRA r4 | — | — | 0.9049 | — |
| MSA polyp ViT-B | — | — | 0.8524 | — |
| MSA polyp ViT-L（选跑） | — | — | 0.8555 | — |
| PP-SAM full-880（选跑，full-FT 对照） | — | — | 0.9226 | — |

## CVC-612 mean Dice

| 模型 | ±0 | ±10 | ±30（锚点） | ±50 |
|---|---|---|---|---|
| SAM ViT-L（0-shot） | — | — | 0.6588 | — |
| MedSAM（0-shot） | — | — | 0.5172 | — |
| SAMed LoRA full-880 | — | — | 0.8608 | — |
| MedSAM+LoRA r4 | — | — | 0.8476 | — |
| MSA polyp ViT-B | — | — | 0.7685 | — |
| MSA polyp ViT-L（选跑） | — | — | 0.8234 | — |
| PP-SAM full-880（选跑，full-FT 对照） | — | — | 0.8576 | — |

## 建议开跑顺序（省 GPU）

1. 先确认 eval 是否支持 `--box_jitter` / `box_offset`（或共用 GT-box 生成）。
2. 第一个必跑：`MedSAM+LoRA r4`，只跑 `0` 与 `50`（ckpt `checkpoints/medsam_peft/r4_full/medsam_lora_r4_best.pth`）。
3. 再：MedSAM 0-shot → SAMed LoRA full-880 → MSA-B → SAM ViT-L。
4. 每个模型写完 `summary.json` 后立刻回填本表；全部完成再勾 Notion Goals #3 并写一页阶段说明。

## 跳过

Chen SAM-Adapter / MedSAM+MSA：已定负结果，不进消融主表。

## 来源

- SAMed ±30：本机 `results/samed_baseline_cloud/.../r4_full_best/summary.json`（Dice 0.8955 / 0.8608）

- Notion 总览结果快照（2026-10-16）
- [Prompt 消融执行清单](https://app.notion.com/p/3fe3092ac6c481769949e3a5298a8415)
- 模板生成：2026-10-21（Asia/Shanghai）
