# MedSAM+LoRA r4_full — v3 eval (best ckpt)

**Date:** 2026-10-15 14:23–14:27 CST (Asia/Shanghai)  
**Machine:** DESKTOP-FARUAJU (RTX 4070 Ti SUPER)  
**Ckpt:** `E:/Project/FYP/checkpoints/medsam_peft/r4_full/medsam_lora_r4_best.pth`  
(best KvasirVal Dice **0.9049** @ epoch 50; train finished 12:29 CST)  
**Out tag:** `r4_full_best`  
**Protocol:** GT box ±30; Kvasir-SEG val 120 + CVC-ClinicDB 612; Dice / IoU / P / R / HD95  
**Data root:** `E:/Project/FYP/datasets_link`  
**Log:** `E:\Project\FYP\results\medsam_peft\logs\eval_r4_full_best_20261015_142340.log`  
**Summary:** `E:\Project\FYP\results\medsam_peft\r4_full_best\summary.json`

## Primary numbers (best ckpt)

| Split | N | Dice (mean±std) | IoU | Precision | Recall | HD95 |
|-------|---|-----------------|-----|-----------|--------|------|
| Kvasir-SEG-val | 120 | **0.9049±0.0978** | **0.8375** | 0.8692 | 0.9624 | 81.71 |
| CVC-ClinicDB | 612 | **0.8476±0.1531** | **0.7596** | 0.8246 | 0.9098 | 35.41 |

Exact from `summary.json`: Kvasir dice_mean=0.904865… iou_mean=0.837510…; CVC dice_mean=0.847605… iou_mean=0.759581….

## Comparison (Dice / IoU unless noted)

| Method | Kvasir-120 Dice/IoU | CVC-612 Dice/IoU |
|--------|---------------------|------------------|
| MedSAM zeroshot | 0.7034 / 0.5846 | 0.5172 / 0.3755 |
| **MedSAM+LoRA r4 best (this run)** | **0.9049 / 0.8375** | **0.8476 / 0.7596** |
| SAMed full880 | 0.8955 / — | 0.8608 / — |
| MSA-B | 0.8524 / — | 0.7685 / — |

Notes on baselines: zeroshot Dice/IoU from `results/phase1_medsam_v3`. SAMed full880 and MSA-B figures are **Dice only** (Kvasir / CVC) as cited in train notes — IoU not restated here.

### Deltas vs MedSAM zeroshot
- Kvasir Dice: +0.2015 (0.7034 → 0.9049); IoU: +0.2529
- CVC Dice: +0.3304 (0.5172 → 0.8476); IoU: +0.3841

### vs PEFT peers (Dice)
- vs SAMed full880: Kvasir **+0.0094** (0.9049 vs 0.8955); CVC **−0.0132** (0.8476 vs 0.8608)
- vs MSA-B: Kvasir **+0.0525**; CVC **+0.0791**

## Artifacts
- Per-image: `E:\Project\FYP\results\medsam_peft\r4_full_best\per_image.json`
- Split dirs: `...\r4_full_best\Kvasir-SEG-val\`, `...\r4_full_best\CVC-ClinicDB\`
- Eval helper bat: `E:\Project\FYP\results\medsam_peft\run_eval_r4_full.bat`
- Last ckpt **not** eval'd this run (optional; best required and done)

## Status
Eval exit code **0**. No failures. GPU was free (train PID 52688 gone) before start; wall ~4.1 min for both splits.
