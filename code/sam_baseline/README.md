# Phase 0: Base SAM Zero-Shot Baseline

Reproduce **B1 (SAM zero-shot)** for the FYP thesis — the floor that P-SAM
(PEFT) is expected to improve upon.

## Layout

```
sam_baseline/
├── infer.py          # Main zero-shot inference (this directory)
├── README.md         # This file
└── TODO.md           # Optional follow-up (e.g. predicted-box variant)
```

## One-time setup

```bash
conda activate sam_env

# 1) Download SAM ViT-B weights (~375 MB)
mkdir -p E:/Project/FYP/checkpoints
cd E:/Project/FYP/checkpoints
curl -L -o sam_vit_b.pth \
  https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
```

## Run

```bash
export FYP_DATA_ROOT="E:/Project/FYP/材料/datasets"
export FYP_SAM_CKPT="E:/Project/FYP/checkpoints/sam_vit_b.pth"
export FYP_OUT_DIR="E:/Project/FYP/results/phase0_sam"

python E:/Project/FYP/code/sam_baseline/infer.py
```

## Expected output

- `E:/Project/FYP/results/phase0_sam/summary.json`  →  per-dataset metrics
- `E:/Project/FYP/results/phase0_sam/{Kvasir-SEG,CVC-ClinicDB}/preds/*.png`  →  binary predictions

## Expected numbers (sanity check)

| Dataset        | Dice      | IoU       | HD95   | FPS (RTX 3090) |
|----------------|-----------|-----------|--------|----------------|
| Kvasir-SEG     | ~75–80 %  | ~65–70 %  | ~25 px | 5–8            |
| CVC-ClinicDB   | ~70–78 %  | ~60–68 %  | ~30 px | 5–8            |

If Kvasir Dice < 70 %, double-check the box prompt, mask loading, and SAM
checkpoint integrity (`md5sum` should match `01ec64...`).

## Box prompt

This script uses **ground-truth-derived boxes** (zero-shot upper bound).
For a realistic evaluation (e.g. to study Q1: prompt quality), swap to a
**predicted box** (YOLO / GroundingDINO). See `TODO.md` for the swap.
