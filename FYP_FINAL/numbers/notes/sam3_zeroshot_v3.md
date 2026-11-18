**Status update (2026-11-17):** 已并入 `results/EXPERIMENT_SUMMARY.md` §1.1 主表。数字仍以 `results/sam3_baseline_v3/summary.json` 为准。

﻿# SAM3 Zero-Shot Results (v3 protocol) — DRAFT

**Date:** 2026-10-12 (Asia/Shanghai)  
**Status:** full eval completed & verified on machine DESKTOP-FARUAJU  
**Do not treat as final table entry until user reviews / EXPERIMENT_SUMMARY updated.**

## Protocol

| Item | Value |
|------|-------|
| Model | Meta SAM 3 (`sam3.pt`, ~848M) |
| Prompt | **SAM-1 interactive** `predict_inst` box XYXY (NOT text/PCS) |
| Box | GT bbox **±30 px** (v3 iron rule) |
| Datasets | Kvasir-SEG `val.txt` (120) + CVC-ClinicDB (612) |
| Metrics | Dice / IoU / Precision / Recall / HD95 / FPS |
| Env | `sam3_env` (Py3.12, torch 2.14+cu126), RTX 4070 Ti SUPER 16GB |

## Results (mean)

| Dataset | N | Dice | IoU | Precision | Recall | HD95 | FPS |
|---------|---|------|-----|-----------|--------|------|-----|
| Kvasir-SEG-val | 120 | **0.7661 ± 0.2466** | 0.6715 | 0.7057 | 0.8857 | 96.75 | 7.26 |
| CVC-ClinicDB | 612 | **0.6287 ± 0.2751** | 0.5104 | 0.5268 | 0.8521 | 56.81 | 7.60 |

## Paths

- Script: `E:\Project\FYP\code\sam3_baseline\infer.py`
- README: `E:\Project\FYP\code\sam3_baseline\README.md`
- Results dir: `E:\Project\FYP\results\sam3_baseline_v3\`
  - `summary.json`, `per_image.json`, `*/preds/`
- Full log: `E:\Project\FYP\results\sam3_full_eval.log`
- Smoke log: `E:\Project\FYP\results\sam3_smoke.log`
- Checkpoint: `E:\Project\FYP\checkpoints\sam3\sam3.pt` (ModelScope mirror)
- Repo: `E:\Project\FYP\材料\repos\sam3` (edt.py Windows patch)

## Notes / caveats

1. **HF gate:** `huggingface.co/facebook/sam3` returns 403 without token; used **ModelScope** `facebook/sam3` weights instead (same `sam3.pt`).
2. **Windows:** Triton unavailable → EDT/post-process falls back; Meta notes this OK for most results.
3. High recall, lower precision on both sets (over-segmentation tendency under loose ±30 box).
4. CVC Dice notably lower than Kvasir (cross-domain), consistent with other zero-shot SAMs.

## Suggested next steps for user

- Spot-check a few triptychs (re-run without `--no-triptych` if needed).
- Compare side-by-side with phase0 SAM / SAM2 / MedSAM v3 numbers in EXPERIMENT_SUMMARY.
- If numbers look good, append formal row to `results/EXPERIMENT_SUMMARY.md`.
- Optional: request HF access + set `HF_TOKEN` for official hub downloads later.

## Context vs existing v3 zero-shot baselines (same ±30 box)

| Method | Kvasir-120 Dice | CVC-612 Dice |
|--------|-----------------|--------------|
| SAM ViT-B (phase0) | 0.7497 | 0.5182 |
| MedSAM (phase1) | 0.7034 | 0.5172 |
| SAM2.1 Hiera-S | 0.7738 | 0.5378 |
| **SAM3 (this run)** | **0.7661** | **0.6287** |

Numbers above taken from existing `summary.json` files; SAM3 from this verified run.
