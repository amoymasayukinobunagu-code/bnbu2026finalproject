# MSA Polyp Adapter Training — FYP Kvasir 880/120

## Goal
Train a polyp-specific Medical SAM Adapter (ImprintLab/Medical-SAM-Adapter) from scratch.
Do NOT use third-party Kvasir-pretrained adapters. Official zoo has no polyp weights.

## Data
- Root: E:/Project/FYP/材料/datasets/Kvasir-SEG
- Train: Data-split/train.txt (880)
- Val (during train): Data-split/val.txt (120)
- Prompt: random positive click (MSA default)
- Mode: 2D image (colonoscopy polyps)

## Base weights
- SAM ViT-B: E:/Project/FYP/checkpoints/sam_vit_b.pth
- Mod: sam_adpt (Adapter PEFT; freeze non-Adapter image_encoder params)

## Smoke (verified OK)
- image_size=256, out_size=256, batch=2, epoch=1
- max_train_samples=16, max_val_samples=8
- Pre-adapt val Dice≈0.33 / IoU≈0.23 on tiny subset (random)
- Train loss≈1.31 in ~0.85s for 8 steps
- Log: results/msa_polyp_train/smoke_train.log

## Full run hyperparams
- exp_name: polyp_kvasir_880
- image_size: 1024
- out_size: 256  (MSA default; matches SAM low-res mask head)
- batch size: 2
- workers: 0 (Windows-safe)
- lr: 1e-4 (Adam)
- epochs: 40
- val_freq: 5
- seed: 24
- GPU: RTX 4070 Ti SUPER 16GB, sam_env (torch 2.11+cu126)
- Env: PYTHONPATH unset; KMP_DUPLICATE_LIB_OK=TRUE

## Outputs
- MSA run logs/ckpts: 材料/repos/Medical-SAM-Adapter/logs/polyp_kvasir_880_<timestamp>/
- Consolidated: results/msa_polyp_train/
- Best adapter copy target: checkpoints/medical_sam_adapter/polyp/

## Code changes (local)
- dataset/kvasir.py + registration in dataset/__init__.py
- cfg: -epoch, -max_train_samples, -max_val_samples
- train.py: honor args.epoch; update best_dice
- function.py: guard args.vis None
- optional dataset imports made non-fatal (missing SimpleITK etc.)

## Eval protocol (after train)
- Val 120 + CVC-ClinicDB 612 with v3 box ±30 via code/medical_sam_adapter/infer.py (point to new polyp ckpt)
## Full run (ACTIVE) — revised for overnight
- **Switched from image_size=1024 (≈6s/batch, ~45min/epoch, ~30h/40ep) to 512**
- image_size: 512
- out_size: 256
- batch size: 4
- epochs: 40
- val_freq: 5
- train set: full 880 / val 120
- PID file: results/msa_polyp_train/full_train.pid
- Logs: results/msa_polyp_train/full_train.err.log (tqdm+logger)
- Run prefix under: 材料/repos/Medical-SAM-Adapter/logs/polyp_kvasir_880_<timestamp>/
- Prior aborted 1024 partial log: full_train.err.log.1024_partial
