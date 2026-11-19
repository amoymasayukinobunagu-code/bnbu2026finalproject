# OP-SAM (B10) ablation — aggregated results

Pipeline variants of the ICCV-2025 OP-SAM cascade (CPG → optional SPF → optional EPE).

| run dir | mode | support_stem | feature_layer | n (per dataset) |
|---|---|---|---|---|
| `E:\Project\FYP\results\opsam_ablation\cpg` | cpg | `cju0qkwl35piu0993l0dewei2` | 7 | Kvasir-SEG-val:120, CVC-ClinicDB:612 |
| `E:\Project\FYP\results\opsam_ablation\epe` | epe | `cju0qkwl35piu0993l0dewei2` | 7 | Kvasir-SEG-val:120, CVC-ClinicDB:612 |
| `E:\Project\FYP\results\opsam_ablation\spf` | spf | `cju0qkwl35piu0993l0dewei2` | 7 | Kvasir-SEG-val:120, CVC-ClinicDB:612 |
| `E:\Project\FYP\results\opsam_ablation\spf_epe` | spf+epe | `cju0qkwl35piu0993l0dewei2` | 7 | Kvasir-SEG-val:120, CVC-ClinicDB:612 |

## Metrics per mode

| dataset | mode | n | Dice | IoU | Precision | Recall | HD95 | FPS |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Kvasir-SEG-val | cpg | 120 | 0.4373 | 0.3327 | 0.4583 | 0.6015 | 212.17 | 2.47 |
| Kvasir-SEG-val | epe | 120 | 0.3712 | 0.2846 | 0.3045 | 0.9363 | 310.49 | 2.10 |
| Kvasir-SEG-val | spf | 120 | 0.4440 | 0.3404 | 0.4629 | 0.6112 | 211.00 | 0.84 |
| Kvasir-SEG-val | spf+epe | 120 | 0.3823 | 0.2952 | 0.3080 | 0.9529 | 308.67 | 0.79 |
| CVC-ClinicDB | cpg | 612 | 0.3719 | 0.2847 | 0.3573 | 0.5610 | 110.75 | 2.57 |
| CVC-ClinicDB | epe | 612 | 0.2838 | 0.2077 | 0.2146 | 0.9105 | 172.89 | 2.47 |
| CVC-ClinicDB | spf | 612 | 0.3762 | 0.2879 | 0.3634 | 0.5639 | 110.28 | 0.86 |
| CVC-ClinicDB | spf+epe | 612 | 0.2889 | 0.2121 | 0.2192 | 0.9130 | 172.23 | 0.84 |

## Paired per-image comparison vs `cpg`

All modes see the identical image list and the identical fixed support image, so per-image deltas are paired and much more sensitive than comparing means (OP-SAM's Dice distribution is U-shaped — near-perfect and complete-fail are both common).

| dataset | mode | n paired | mean Δ Dice | wins | losses | ties | images with |Δ|>0.1 |
|---|---|---:|---:|---:|---:|---:|---:|
| Kvasir-SEG-val | epe | 120 | -0.0661 | 48 | 72 | 0 | 84 |
| Kvasir-SEG-val | spf | 120 | +0.0067 | 67 | 48 | 5 | 5 |
| Kvasir-SEG-val | spf+epe | 120 | -0.0550 | 51 | 69 | 0 | 85 |
| CVC-ClinicDB | epe | 612 | -0.0881 | 247 | 363 | 2 | 407 |
| CVC-ClinicDB | spf | 612 | +0.0043 | 284 | 264 | 64 | 15 |
| CVC-ClinicDB | spf+epe | 612 | -0.0830 | 248 | 362 | 2 | 401 |

## Protocol caveats (must travel with these numbers)

- OP-SAM is **one-shot, training-free and prompt-free**: it does NOT consume the v3 ±30 px GT-derived box prompt, so it is **not directly rankable** against the box-prompt baselines on raw Dice. Report it as its own row with `Prompt Type = none` (like ASPS, which is dense).
- Training cost = **1 labelled support image** (`Kvasir-SEG/Data-split/train.txt[0]`), the same single label as PP-SAM 1-shot, so the annotation budget is byte-for-byte identical.
- Kvasir-SEG `val.txt` 120 held-out + full CVC-ClinicDB 612; 5 metrics + FPS; ACL-safe reads; SEED=42.
- Cost, measured 2026-09-30: cpg = 2.6 min (Kvasir-120) / 5.1 min (CVC-612) wall including model load; SPF ≈3.3× per image. The full 4-mode × 2-dataset sweep is ≈1.2 h, so every configuration is reported on both datasets and nothing is skipped for cost.
