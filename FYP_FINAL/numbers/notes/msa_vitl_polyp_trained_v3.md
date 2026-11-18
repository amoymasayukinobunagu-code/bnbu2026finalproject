# MSA polyp — SAM ViT-L + Adapter (trained, not zero-shot)

**Date:** 2026-10-14  
**Method:** Medical SAM Adapter (`sam_adpt`) on **SAM ViT-L**, train.txt 880 only  
**Protocol v3:** Kvasir val 120 + CVC-ClinicDB 612, GT box ±30  
**Hyperparams:** image_size 512, batch 2 (bs4 OOM mid ep1), 40 epochs, Adam 1e-4, best @ epoch 21  
**Ckpt:** `checkpoints/medical_sam_adapter/polyp_vitl/best_dice_checkpoint.pth`  
**Eval:** `results/msa_vitl_polyp_train/eval_v3/summary.json`

| Split | Dice | IoU | HD95 | FPS |
|-------|------|-----|------|-----|
| Kvasir-120 | **0.8555** ±0.1823 | 0.7820 | 103.25 | 12.38 |
| CVC-612 | **0.8234** ±0.2165 | 0.7432 | 41.18 | 12.78 |

## vs peers (same v3)

| Method | Setting | Kvasir Dice | CVC Dice |
|--------|---------|-------------|----------|
| SAM ViT-L | zero-shot | 0.8140 | 0.6588 |
| HQ-SAM ViT-B | zero-shot | 0.8007 | 0.6525 |
| MSA polyp ViT-B | trained 880 | 0.8524 | 0.7685 |
| **MSA polyp ViT-L** | **trained 880** | **0.8555** | **0.8234** |
| SAMed LoRA full 880 | trained | 0.8955 | 0.8608 |
| PP-SAM full 880 | trained | 0.9226 | 0.8576 |

**Takeaway:** ViT-L + Adapter lifts CVC a lot vs MSA-B (+5.5pp) and vs ViT-L zero-shot (+16.5pp); Kvasir only +0.3pp vs MSA-B. Still below SAMed / PP-SAM full-880.
