"""
MedSAM ViT-B zero-shot evaluation on Kvasir-SEG (val split) + CVC-ClinicDB.

Protocol (matches eval_ppsam.py / eval_sam_zeroshot.py for fair comparison):
- Load MedSAM ViT-B (E:/Project/FYP/checkpoints/medsam_vit_b.pth)
- GT box + 30 px inference perturbation (PP-SAM paper's protocol)
- 5 metrics: Dice, IoU, Precision, Recall, HD95
- ResizeLongestSide 1024, multimask_output=False
- Kvasir-SEG val.txt (120) + CVC-ClinicDB full 612
- Per-image JSON + triptych PNGs + summary JSON

Why this script exists
----------------------
The old `phase1_medsam_*` runs used `--limit 50` and no 30-px box perturbation,
producing cherry-picked numbers (CVC 0.73 on first 50) that aren't comparable
to the new protocol used for B1/B4 baselines. This script emits apples-to-apples
MedSAM numbers for Table 1.

Usage
-----
    conda activate sam_env
    python E:/Project/FYP/code/pp_sam_baseline/eval_medsam_zeroshot.py
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch
from PIL import Image
from scipy.spatial.distance import directed_hausdorff
from segment_anything import sam_model_registry, SamPredictor

# Re-use helpers from eval_ppsam
sys.path.insert(0, str(Path(__file__).parent))
from eval_ppsam import (  # noqa: E402
    DATA_ROOT,
    load_image, load_mask, gt_box_from_mask,
    dice_score, iou_score, precision_recall, hd95_score,
    discover_kvasir_val, discover_clinicdb,
    save_triptych, DATASETS,
)

# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
MEDSAM_CKPT = Path(os.environ.get(
    "FYP_MEDSAM_CKPT", "E:/Project/FYP/checkpoints/medsam_vit_b.pth"))
OUT_DIR = Path(os.environ.get(
    "FYP_OUT_DIR", "E:/Project/FYP/results/medsam_new_protocol"))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE = 1024


def load_medsam(ckpt_path: Path) -> "SamPredictor":
    """Load MedSAM weights into the standard SAM ViT-B architecture.

    MedSAM (medsam_vit_b.pth) uses the same state_dict key layout as the
    original SAM ViT-B (image_encoder.*, prompt_encoder.*, mask_decoder.*).
    We attempt strict=True first; if that mismatches we fall back to
    strict=False with a printed diagnostic.
    """
    print(f"[Config] device={DEVICE}  ckpt={ckpt_path.name}")
    if not ckpt_path.exists():
        raise FileNotFoundError(ckpt_path)

    # Build empty SAM ViT-B architecture, then load MedSAM weights.
    sam = sam_model_registry["vit_b"]()
    state = torch.load(str(ckpt_path), map_location="cpu", weights_only=True)
    try:
        sam.load_state_dict(state, strict=True)
        print(f"[OK] MedSAM weights loaded with strict=True ({len(state)} tensors).")
    except RuntimeError as e:
        # Diagnostic: print first few mismatched keys, then fall back.
        sam_keys = set(sam.state_dict().keys())
        ckpt_keys = set(state.keys())
        missing = list(sam_keys - ckpt_keys)
        unexpected = list(ckpt_keys - sam_keys)
        print(f"[WARN] strict=True failed: {e}")
        print(f"  Missing keys (first 5):   {missing[:5]}")
        print(f"  Unexpected keys (first 5): {unexpected[:5]}")
        print(f"  Falling back to strict=False (loading {len(state)} tensors).")
        sam.load_state_dict(state, strict=False)
    sam.to(DEVICE)
    sam.eval()
    return SamPredictor(sam)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--limit", type=int, default=None,
                   help="cap images per dataset (DO NOT use for final eval)")
    p.add_argument("--dataset", type=str, default=None,
                   choices=list(DATASETS.keys()),
                   help="only evaluate on this dataset (default: all)")
    p.add_argument("--no-triptych", action="store_true",
                   help="skip 3-panel visualisations")
    p.add_argument("--box-perturb", type=int, default=30,
                   help="GT box outward perturbation in pixels (v3 default 30)")
    args = p.parse_args()

    print(f"[Config] infer_perturb={args.box_perturb}  out_dir={OUT_DIR}")
    predictor = load_medsam(MEDSAM_CKPT)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary = {}
    datasets = [args.dataset] if args.dataset else list(DATASETS.keys())

    for ds_name in datasets:
        pairs = DATASETS[ds_name]()
        if not pairs:
            print(f"[SKIP] {ds_name}: no images found")
            continue
        if args.limit:
            print(f"[WARN] limiting to first {args.limit} images (only for sanity)")
            pairs = pairs[: args.limit]
        print(f"=== {ds_name}: {len(pairs)} images ===")

        out_ds = OUT_DIR / ds_name
        pred_dir = out_ds / "preds"
        trip_dir = out_ds / "triptychs"
        pred_dir.mkdir(parents=True, exist_ok=True)
        if not args.no_triptych:
            trip_dir.mkdir(parents=True, exist_ok=True)

        dices, ious, precs, recs, hd95s, fps_list = [], [], [], [], [], []
        per_image = []

        for i, (img_path, msk_path) in enumerate(pairs):
            image = load_image(img_path)
            gt = load_mask(msk_path)
            if gt.sum() == 0:
                continue
            box = gt_box_from_mask(gt, perturb=args.box_perturb)
            if box is None:
                continue

            t0 = time.time()
            predictor.set_image(image)
            masks, scores, _ = predictor.predict(
                point_coords=None, point_labels=None,
                box=box[None, :], multimask_output=False,
            )
            elapsed = time.time() - t0
            pred = masks[0].astype(np.uint8)
            fps_list.append(1.0 / (elapsed + 1e-8))

            d = dice_score(pred, gt)
            j = iou_score(pred, gt)
            pr, rc = precision_recall(pred, gt)
            h = hd95_score(pred, gt)
            dices.append(d); ious.append(j); precs.append(pr); recs.append(rc); hd95s.append(h)
            per_image.append({
                "name": img_path.name,
                "dice": d, "iou": j, "precision": pr, "recall": rc,
                "hd95": h, "fps": 1.0 / (elapsed + 1e-8),
            })

            Image.fromarray((pred * 255).astype(np.uint8)).save(pred_dir / img_path.name)
            if not args.no_triptych:
                save_triptych(image, gt, pred, trip_dir / f"{img_path.stem}.jpg", d)

            if (i + 1) % 25 == 0 or i + 1 == len(pairs):
                print(f"  [{i+1:3d}/{len(pairs)}] running Dice={np.mean(dices):.4f} "
                      f"IoU={np.mean(ious):.4f} HD95={np.nanmean(hd95s):.2f}")

        if dices:
            summary[ds_name] = {
                "n": len(dices),
                "dice_mean": float(np.mean(dices)),
                "dice_std": float(np.std(dices)),
                "iou_mean": float(np.mean(ious)),
                "iou_std": float(np.std(ious)),
                "precision_mean": float(np.mean(precs)),
                "recall_mean": float(np.mean(recs)),
                "hd95_mean": float(np.nanmean(hd95s)),
                "fps_mean": float(np.nanmean(fps_list)),
                "per_image": per_image,
            }
            print(f"\n  >>> {ds_name}: "
                  f"Dice={np.mean(dices):.4f}±{np.std(dices):.4f}  "
                  f"IoU={np.mean(ious):.4f}  "
                  f"Prec={np.mean(precs):.4f}  Rec={np.mean(recs):.4f}  "
                  f"HD95={np.nanmean(hd95s):.2f}  FPS={np.nanmean(fps_list):.2f}\n")

    compact = {k: {kk: vv for kk, vv in v.items() if kk != "per_image"}
               for k, v in summary.items()}
    with open(OUT_DIR / "summary.json", "w") as f:
        json.dump(compact, f, indent=2)
    with open(OUT_DIR / "per_image.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n[DONE] summary -> {OUT_DIR / 'summary.json'}")
    print(json.dumps(compact, indent=2))


if __name__ == "__main__":
    main()
