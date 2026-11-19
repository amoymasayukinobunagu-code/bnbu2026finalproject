# MSA polyp adapter — TRAINED evaluation (v3 protocol)

**Date**: 2026-10-13 (Asia/Shanghai)  
**Status**: **TRAINED polyp adapter** (not zero-shot)  
**Model**: Medical SAM Adapter (`sam_adpt`), ViT-B + Adapter PEFT  
**Checkpoint**: `checkpoints/medical_sam_adapter/polyp/best_dice_checkpoint.pth`  
  (alias: `polyp_kvasir_880_best.pth`; best epoch **26**)  
**Code**: `code/medical_sam_adapter/eval_polyp_v3.py`  
**Results**: `results/msa_polyp_train/eval_v3/` (mirror: `results/msa_polyp_baseline_v3/`)  
**Hyperparams**: `results/msa_polyp_train/HYPERPARAMS.md`

## Training (summary)

| Item | Value |
|------|-------|
| Data | Kvasir-SEG train 880 / val 120 |
| image_size | **512** (revised from 1024) |
| batch size | **4** |
| epochs | **40** (best Dice @ **ep26**) |
| out_size | 256 |
| lr | 1e-4 Adam |
| base | `sam_vit_b.pth`, mod=`sam_adpt` |
| GPU | RTX 4070 Ti SUPER 16GB, `sam_env` |

This run finetunes MSA adapters on Kvasir polyps. It is **not** a zero-shot / off-the-shelf MSA evaluation.

## Eval protocol (v3)

- Kvasir-SEG: official `Data-split/val.txt` (**120**)
- CVC-ClinicDB: full **612**
- Box prompt: GT bbox **±30 px**; multimask argmax
- Image resize to 512 for MSA encode/decode, mask resized back to native
- Metrics: Dice / IoU / Precision / Recall / HD95 / FPS

## Main results (means)

| Dataset | n | Dice ± std | IoU ± std | Precision | Recall | HD95 | FPS |
|---------|---|------------|-----------|-----------|--------|------|-----|
| Kvasir-SEG-val | 120 | **0.8524** ± 0.2017 | **0.7826** ± 0.2293 | 0.8864 | 0.8670 | 104.97 | 19.68 |
| CVC-ClinicDB | 612 | **0.7685** ± 0.2808 | **0.6896** ± 0.2915 | 0.7998 | 0.8342 | 61.59 | 22.71 |

## Comparison (same v3 box±30 protocol)

| Method | Setting | Kvasir Dice | Kvasir IoU | CVC Dice | CVC IoU |
|--------|---------|-------------|------------|----------|---------|
| HQ-SAM ViT-B | zero-shot | 0.8007 | 0.7104 | 0.6525 | 0.5477 |
| **MSA polyp adapter** | **trained (Kvasir 880, ep26)** | **0.8524** | **0.7826** | **0.7685** | **0.6896** |
| SAMed LoRA r4 full 880 | trained | **0.8955** | 0.8280 | **0.8608** | 0.7814 |

Notes:
- vs HQ-SAM zero-shot: **+5.2 pp** Kvasir Dice, **+11.6 pp** CVC Dice.
- vs SAMed full-880: **−4.3 pp** Kvasir Dice, **−9.2 pp** CVC Dice (SAMed still stronger under this protocol).
- MSA FPS (~20–23) is higher than SAMed (~12–13) and HQ-SAM (~8) on this machine.

## Artifacts

| Path | Role |
|------|------|
| `checkpoints/medical_sam_adapter/polyp/best_dice_checkpoint.pth` | Best trained ckpt (ep26) |
| `code/medical_sam_adapter/eval_polyp_v3.py` | Eval script |
| `results/msa_polyp_train/HYPERPARAMS.md` | Training hyperparams |
| `results/msa_polyp_train/eval_v3/summary.json` | Combined means (Kvasir+CVC) |
| `results/msa_polyp_train/eval_v3/per_image.json` | Per-image metrics |
| `results/msa_polyp_train/eval_v3/{Kvasir-SEG-val,CVC-ClinicDB}/preds/` | Prediction masks |
| `results/msa_polyp_baseline_v3/summary.json` | Clean combined copy |
| `results/msa_polyp_train/eval_val120.log` | Kvasir val120 stdout |
| `results/msa_polyp_train/eval_cvc612.log` | CVC 612 stdout |