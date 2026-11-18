# HQ-SAM zero-shot evaluation (v3 protocol)

**Date**: 2026-10-13 (Asia/Shanghai)  
**Model**: HQ-SAM ViT-B (`sam_hq_vit_b.pth`, HuggingFace `lkeab/hq-sam`)  
**Code**: `code/hq_sam_baseline/infer.py`  
**Results**: `results/hq_sam_baseline_v3/`

## Protocol

- Kvasir-SEG: official `Data-split/val.txt` (120 held-out)
- CVC-ClinicDB: full 612
- Box prompt: GT bbox ±30 px; `multimask_output=False`
- Metrics: Dice / IoU / Precision / Recall / HD95 / FPS
- Env: `sam_env` + `pip install segment-anything-hq`; `PYTHONPATH` cleared

## Main results (means)

| Method | Kvasir Dice | Kvasir IoU | Kvasir HD95 | Kvasir FPS | CVC Dice | CVC IoU | CVC HD95 | CVC FPS |
|--------|-------------|------------|-------------|------------|----------|---------|----------|---------|
| SAM ViT-B (B1) | 0.7497 | 0.6680 | 95.63 | ~6.9 | 0.5182 | 0.4300 | 63.76 | ~7.1 |
| SAM3 | 0.7661 | 0.6715 | 96.75 | ~7.3 | 0.6287 | 0.5104 | 56.81 | ~7.6 |
| **HQ-SAM ViT-B** | **0.8007** | **0.7104** | **91.09** | **7.71** | **0.6525** | **0.5477** | **51.72** | **7.93** |

Full HQ-SAM split detail (from `summary.json`):

| Dataset | n | Dice ± std | IoU ± std | Precision | Recall | HD95 | FPS |
|---------|---|------------|-----------|-----------|--------|------|-----|
| Kvasir-SEG-val | 120 | 0.8007 ± 0.2186 | 0.7104 ± 0.2389 | 0.7633 | 0.8891 | 91.09 | 7.71 |
| CVC-ClinicDB | 612 | 0.6525 ± 0.2943 | 0.5477 ± 0.2951 | 0.6339 | 0.7996 | 51.72 | 7.93 |

## Comparison notes

- vs SAM ViT-B: **+5.1 pp** Kvasir Dice, **+13.4 pp** CVC Dice; lower HD95 on both sets.
- vs SAM3: **+3.5 pp** Kvasir Dice, **+2.4 pp** CVC Dice; similar FPS (~8).
- HQ-SAM ViT-B is the strongest of these three zero-shot box-prompt baselines under v3 on both datasets.
- Prompt mode logged as `hq_sam_box_xyxy_pm30` (parity with SAM / SAM3 box protocol).

## Artifacts

| Path | Role |
|------|------|
| `code/hq_sam_baseline/infer.py` | Reproducible eval script |
| `code/hq_sam_baseline/README.md` | Run instructions |
| `checkpoints/hq_sam/sam_hq_vit_b.pth` | Weights (~379 MB, HF download OK) |
| `材料/repos/sam-hq` | Official SysCV clone |
| `results/hq_sam_baseline_v3/summary.json` | Compact means |
| `results/hq_sam_baseline_v3/per_image.json` | Per-image metrics |
| `results/hq_sam_full_eval.log` | Full run stdout |

## Setup notes

- Checkpoint download: **HuggingFace worked** (`huggingface_hub` → `lkeab/hq-sam` / `sam_hq_vit_b.pth`). No ModelScope fallback needed.
- Package: `segment-anything-hq` in `sam_env` (does not replace vanilla `segment_anything`).
- Smoke (limit=3) passed before full run; full eval ~2.5 min on RTX 4070 Ti SUPER.
