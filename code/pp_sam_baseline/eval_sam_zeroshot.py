"""
Same eval as eval_ppsam.py but loads the FROZEN vanilla SAM ViT-B (no fine-tune).
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from segment_anything import sam_model_registry, SamPredictor

# Re-use helpers from eval_ppsam
sys.path.insert(0, str(Path(__file__).parent))
from eval_ppsam import (
    DATA_ROOT, SAM_CKPT, OUT_DIR as _UNUSED,
    load_image, load_mask, gt_box_from_mask,
    dice_score, iou_score, precision_recall, hd95_score,
    discover_kvasir_val, discover_clinicdb,
    save_triptych, DATASETS,
)

OUT_DIR = Path("E:/Project/FYP/results/pp_sam_baseline/eval_sam_zeroshot")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def main(limit=None, dataset=None, triptych=True, box_perturb=30):
    print(f"[Config] device={DEVICE}  ckpt={SAM_CKPT.name}  perturb={box_perturb}")
    sam = sam_model_registry["vit_b"](checkpoint=str(SAM_CKPT))
    sam.to(DEVICE)
    sam.eval()
    predictor = SamPredictor(sam)
    print("[OK] Vanilla SAM ViT-B loaded (no fine-tune).\n")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary = {}
    datasets = [dataset] if dataset else list(DATASETS.keys())

    for ds_name in datasets:
        pairs = DATASETS[ds_name]()
        if limit:
            pairs = pairs[:limit]
        print(f"=== {ds_name}: {len(pairs)} images ===")
        out_ds = OUT_DIR / ds_name
        pred_dir = out_ds / "preds"
        trip_dir = out_ds / "triptychs"
        pred_dir.mkdir(parents=True, exist_ok=True)
        if triptych:
            trip_dir.mkdir(parents=True, exist_ok=True)

        dices, ious, precs, recs, hd95s, fps_list = [], [], [], [], [], []
        per_image = []
        for i, (img_path, msk_path) in enumerate(pairs):
            image = load_image(img_path)
            gt = load_mask(msk_path)
            if gt.sum() == 0:
                continue
            box = gt_box_from_mask(gt, perturb=box_perturb)
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
            d = dice_score(pred, gt); j = iou_score(pred, gt)
            pr, rc = precision_recall(pred, gt); h = hd95_score(pred, gt)
            dices.append(d); ious.append(j); precs.append(pr); recs.append(rc); hd95s.append(h)
            per_image.append({"name": img_path.name, "dice": d, "iou": j,
                              "precision": pr, "recall": rc, "hd95": h,
                              "fps": 1.0 / (elapsed + 1e-8)})
            Image.fromarray((pred * 255).astype(np.uint8)).save(pred_dir / img_path.name)
            if triptych:
                save_triptych(image, gt, pred, trip_dir / f"{img_path.stem}.jpg", d)
            if (i + 1) % 50 == 0 or i + 1 == len(pairs):
                print(f"  [{i+1}/{len(pairs)}] running Dice={np.mean(dices):.4f} IoU={np.mean(ious):.4f}")

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
            print(f"  >>> {ds_name}: Dice={np.mean(dices):.4f}±{np.std(dices):.4f}  "
                  f"IoU={np.mean(ious):.4f}  HD95={np.nanmean(hd95s):.2f}\n")

    compact = {k: {kk: vv for kk, vv in v.items() if kk != "per_image"} for k, v in summary.items()}
    with open(OUT_DIR / "summary.json", "w") as f:
        json.dump(compact, f, indent=2)
    with open(OUT_DIR / "per_image.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[DONE] -> {OUT_DIR / 'summary.json'}")
    print(json.dumps(compact, indent=2))


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--dataset", type=str, default=None)
    p.add_argument("--no-triptych", action="store_true")
    p.add_argument("--box-perturb", type=int, default=30,
                   help="GT box outward perturbation in pixels (v3 default 30)")
    a = p.parse_args()
    main(limit=a.limit, dataset=a.dataset, triptych=not a.no_triptych,
         box_perturb=a.box_perturb)
