"""
MedSAM + LoRA evaluation — FYP v3 protocol (Kvasir val.txt 120 + CVC 612).

Loads a MedSAM+LoRA checkpoint trained by train_medsam_lora.py and evaluates with
+/-30 px GT box prompt (Dice / IoU / Precision / Recall / HD95 + FPS).

Backbone must be the same official MedSAM weights used at train time
(checkpoints/medsam_vit_b.pth); LoRA A/B + prompt_encoder + mask_decoder come from --ckpt.

Usage:
    python eval_medsam_lora.py --ckpt E:/Project/FYP/checkpoints/medsam_peft/r4_full/medsam_lora_r4_best.pth \
                               --rank 4 --out-tag r4_full_best
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import torch
from PIL import Image
from scipy.spatial.distance import directed_hausdorff

from segment_anything import sam_model_registry, SamPredictor

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lora_sam import LoRA_Sam

# --------------------------------------------------------------------------- #
# Config (env overrides honored)
# --------------------------------------------------------------------------- #
PROJECT_ROOT = Path(r"E:/Project/FYP")
DATA_ROOT = Path(os.environ.get("FYP_DATA_ROOT", PROJECT_ROOT / "datasets_link"))  # junction -> 材料/datasets
SAM_CKPT = Path(os.environ.get(
    "FYP_SAM_CKPT",
    os.environ.get("FYP_MEDSAM_CKPT", PROJECT_ROOT / "checkpoints" / "medsam_vit_b.pth"),
))
OUT_ROOT = Path(os.environ.get("FYP_OUT_DIR", PROJECT_ROOT / "results" / "medsam_peft"))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42

np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# --------------------------------------------------------------------------- #
# I/O (Linux: plain open()/cv2.imread, no NTFS ACL workaround)
# --------------------------------------------------------------------------- #
def load_image(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise IOError(f"cv2.imread failed for {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def load_mask(path: Path) -> np.ndarray:
    m = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if m is None:
        raise IOError(f"cv2.imread (gray) failed for {path}")
    return (m > 127).astype(np.uint8)


def gt_box_from_mask(mask: np.ndarray, perturb: int = 30) -> np.ndarray:
    """v3 protocol: tight box + ±30 px outward perturbation."""
    ys, xs = np.where(mask > 0)
    H, W = mask.shape
    return np.array([
        max(0, int(xs.min()) - perturb),
        max(0, int(ys.min()) - perturb),
        min(W, int(xs.max()) + perturb),
        min(H, int(ys.max()) + perturb),
    ], dtype=np.int32)


# --------------------------------------------------------------------------- #
# Metrics (5-metric set, byte-identical to sam_baseline/infer.py)
# --------------------------------------------------------------------------- #
def dice_score(pred, gt):
    p, g = pred.astype(bool), gt.astype(bool)
    inter = np.logical_and(p, g).sum()
    denom = p.sum() + g.sum()
    return float(2.0 * inter / denom) if denom else 1.0


def iou_score(pred, gt):
    p, g = pred.astype(bool), gt.astype(bool)
    inter = np.logical_and(p, g).sum()
    union = np.logical_or(p, g).sum()
    return float(inter / union) if union else 1.0


def precision_recall(pred, gt):
    p, g = pred.astype(bool), gt.astype(bool)
    tp = np.logical_and(p, g).sum()
    fp = np.logical_and(p, ~g).sum()
    fn = np.logical_and(~p, g).sum()
    pr = float(tp / (tp + fp)) if (tp + fp) else 1.0
    rc = float(tp / (tp + fn)) if (tp + fn) else 1.0
    return pr, rc


def hd95_score(pred, gt):
    if pred.sum() == 0 or gt.sum() == 0:
        return float("nan")
    p_pts = np.argwhere(pred > 0)
    g_pts = np.argwhere(gt > 0)
    fwd = directed_hausdorff(p_pts, g_pts)[0]
    bwd = directed_hausdorff(g_pts, p_pts)[0]
    return float(max(fwd, bwd))


# --------------------------------------------------------------------------- #
# Dataset discovery
# --------------------------------------------------------------------------- #
def discover_kvasir_val() -> List[Tuple[Path, Path]]:
    val_txt = DATA_ROOT / "Kvasir-SEG" / "Data-split" / "val.txt"
    img_dir = DATA_ROOT / "Kvasir-SEG" / "images"
    msk_dir = DATA_ROOT / "Kvasir-SEG" / "masks"
    if not val_txt.exists():
        raise FileNotFoundError(val_txt)
    pairs = []
    with open(val_txt, "r") as f:
        stems = [ln.strip() for ln in f if ln.strip()]
    for stem in stems:
        img = img_dir / f"{stem}.jpg"
        if not img.exists():
            img = img_dir / f"{stem}.png"
        msk = msk_dir / f"{stem}.jpg"
        if not msk.exists():
            msk = msk_dir / f"{stem}.png"
        if img.exists() and msk.exists():
            pairs.append((img, msk))
    return pairs


def discover_clinicdb() -> List[Tuple[Path, Path]]:
    img_dir = DATA_ROOT / "CVC-ClinicDB" / "PNG" / "Original"
    msk_dir = DATA_ROOT / "CVC-ClinicDB" / "PNG" / "Ground Truth"
    pairs = []
    for img in sorted(img_dir.iterdir()):
        if img.suffix.lower() not in {".png", ".jpg"}:
            continue
        msk = msk_dir / img.name
        if msk.exists():
            pairs.append((img, msk))
    return pairs


DATASETS = {
    "Kvasir-SEG-val": discover_kvasir_val,
    "CVC-ClinicDB": discover_clinicdb,
}


# --------------------------------------------------------------------------- #
# Visualisation
# --------------------------------------------------------------------------- #
def save_triptych(image, gt_mask, pred_mask, out_path, dice):
    h, w = image.shape[:2]
    if gt_mask.shape[:2] != (h, w):
        gt_mask = cv2.resize(gt_mask, (w, h), interpolation=cv2.INTER_NEAREST)
    if pred_mask.shape[:2] != (h, w):
        pred_mask = cv2.resize(pred_mask, (w, h), interpolation=cv2.INTER_NEAREST)
    gt_overlay = image.copy()
    pred_overlay = image.copy()
    gt_overlay[gt_mask > 0] = (gt_overlay[gt_mask > 0] * 0.4 + np.array([0, 255, 0]) * 0.6).astype(np.uint8)
    pred_overlay[pred_mask > 0] = (pred_overlay[pred_mask > 0] * 0.4 + np.array([255, 0, 0]) * 0.6).astype(np.uint8)
    panel = np.concatenate([image, gt_overlay, pred_overlay], axis=1)
    sep = np.full((h, 4, 3), 0, dtype=np.uint8)
    panel = np.concatenate([panel[:, :w], sep, panel[:, w:2*w], sep, panel[:, 2*w:]], axis=1)
    cv2.putText(panel, f"Dice={dice:.3f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
    scale = 1200 / panel.shape[1]
    panel = cv2.resize(panel, (1200, int(panel.shape[0] * scale)))
    cv2.imwrite(str(out_path), cv2.cvtColor(panel, cv2.COLOR_RGB2BGR))


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, required=True, help="path to SAMed LoRA .pth")
    p.add_argument("--rank", type=int, default=4, help="LoRA rank used during training")
    p.add_argument("--out-tag", type=str, required=True, help="subdir under results/medsam_peft/")
    p.add_argument("--dataset", type=str, default=None, choices=list(DATASETS.keys()))
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--no-triptych", action="store_true")
    p.add_argument("--box-perturb", type=int, default=30,
                   help="GT box outward perturbation in pixels (v3 default 30)")
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = OUT_ROOT / args.out_tag
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[Config] device={DEVICE} ckpt={args.ckpt} out_dir={out_dir} box_perturb={args.box_perturb}")

    print("\n[Loading MedSAM ViT-B + LoRA...]")
    sam = sam_model_registry["vit_b"](checkpoint=str(SAM_CKPT))
    lora_model = LoRA_Sam(sam, r=args.rank).to(DEVICE)
    lora_model.load_lora_parameters(args.ckpt)
    # 2026-06-24 hotfix: load_lora_parameters() uses map_location='cpu'
    # which puts the LoRA A/B Linear layers on CPU; re-attach to GPU here.
    # Without this, forward fails with: "Expected all tensors to be on the
    # same device, but found at least two devices, cuda:0 and cpu!"
    lora_model.to(DEVICE)
    lora_model.sam.eval()
    predictor = SamPredictor(lora_model.sam)
    print("[OK] MedSAM+LoRA loaded.\n")

    datasets = [args.dataset] if args.dataset else list(DATASETS.keys())
    summary = {}

    for ds_name in datasets:
        pairs = DATASETS[ds_name]()
        if args.limit:
            pairs = pairs[: args.limit]
        print(f"=== {ds_name}: {len(pairs)} images ===")

        pred_dir = out_dir / ds_name / "preds"
        trip_dir = out_dir / ds_name / "triptychs"
        pred_dir.mkdir(parents=True, exist_ok=True)
        if not args.no_triptych:
            trip_dir.mkdir(parents=True, exist_ok=True)

        dices, ious, precs, recs, hd95s, fps_list = [], [], [], [], [], []
        per_image = []

        for i, (img_path, msk_path) in enumerate(pairs):
            image = load_image(img_path)
            gt_mask = load_mask(msk_path)
            if gt_mask.sum() == 0:
                continue
            box = gt_box_from_mask(gt_mask, perturb=args.box_perturb)

            t0 = time.time()
            predictor.set_image(image)
            masks, scores, _ = predictor.predict(
                point_coords=None, point_labels=None,
                box=box[None, :], multimask_output=False,
            )
            elapsed = time.time() - t0
            pred_mask = masks[0].astype(np.uint8)
            fps_list.append(1.0 / (elapsed + 1e-8))

            d = dice_score(pred_mask, gt_mask)
            j = iou_score(pred_mask, gt_mask)
            pr, rc = precision_recall(pred_mask, gt_mask)
            h = hd95_score(pred_mask, gt_mask)
            dices.append(d); ious.append(j); precs.append(pr); recs.append(rc); hd95s.append(h)
            per_image.append({"name": img_path.name, "dice": d, "iou": j,
                              "precision": pr, "recall": rc, "hd95": h,
                              "fps": 1.0 / (elapsed + 1e-8)})

            Image.fromarray((pred_mask * 255).astype(np.uint8)).save(pred_dir / img_path.name)
            if not args.no_triptych:
                save_triptych(image, gt_mask, pred_mask, trip_dir / f"{img_path.stem}.jpg", d)

            if (i + 1) % 20 == 0 or i == len(pairs) - 1:
                print(f"  [{i+1:3d}/{len(pairs)}] D={d:.4f} J={j:.4f}  "
                      f"running mean Dice={np.mean(dices):.4f}")

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
                "ckpt": str(args.ckpt),
                "rank": args.rank,
                "backbone": "medsam_vit_b",
                "method": "medsam_lora",
                "per_image": per_image,
            }
            print(f"\n  >>> {ds_name}: Dice={np.mean(dices):.4f}±{np.std(dices):.4f}  "
                  f"IoU={np.mean(ious):.4f}  P={np.mean(precs):.4f}  R={np.mean(recs):.4f}  "
                  f"HD95={np.nanmean(hd95s):.2f}\n")

    compact = {k: {kk: vv for kk, vv in v.items() if kk != "per_image"} for k, v in summary.items()}
    sum_path = out_dir / "summary.json"
    pi_path = out_dir / "per_image.json"

    # Merge with existing summary.json
    existing_summary = json.loads(sum_path.read_text()) if sum_path.exists() else {}
    existing_pi = json.loads(pi_path.read_text()) if pi_path.exists() else {}
    merged_compact = {**existing_summary, **compact}
    merged_full = {**existing_pi, **summary}

    sum_path.write_text(json.dumps(merged_compact, indent=2))
    pi_path.write_text(json.dumps(merged_full, indent=2))
    print(f"[DONE] merged -> {sum_path}")
    print(json.dumps(merged_compact, indent=2))


if __name__ == "__main__":
    main()
