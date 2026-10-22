"""
PPSAM Best-checkpoint evaluation on Kvasir-SEG (val split) + CVC-ClinicDB.

Key protocol:
- Load PolypSAM_freeze_mask_decoder_vit_b_..._FYP-best.pth (frozen-decoder, 1-shot)
- GT box + 30 px inference perturbation (paper's protocol)
- 5 metrics: Dice, IoU, Precision, Recall, HD95
- ResizeLongestSide 1024, multimask_output=False

Why this script exists
----------------------
train_ppsam.py's val loop aggregates Dice over the FULL 1000-image Kvasir-SEG
folder (no train/val split), so the 0.8432 best number from training is
**in-distribution 1-shot with no held-out test**. This script enforces a
clean val split (val.txt 120 imgs) and a cross-domain test (CVC-ClinicDB 612
imgs, fully unseen during training).

Usage
-----
    conda activate sam_env
    python E:/Project/FYP/code/pp_sam_baseline/eval_ppsam.py
    # or limit to 20 for sanity:
    python E:/Project/FYP/code/pp_sam_baseline/eval_ppsam.py --limit 20
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

# Allow importing from repos
sys.path.insert(0, "E:/Project/FYP/材料/repos/PP-SAM")

# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
DATA_ROOT = Path(os.environ.get("FYP_DATA_ROOT", "E:/Project/FYP/材料/datasets"))
SAM_CKPT = Path(os.environ.get("FYP_SAM_CKPT", "E:/Project/FYP/checkpoints/sam_vit_b.pth"))
CKPT = Path(os.environ.get(
    "FYP_PPSAM_CKPT",
    "E:/Project/FYP/results/pp_sam_baseline/work_dir/model_pth/PP-SAM/"
    "PolypSAM_freeze_mask_decoder_vit_b_train_p0_50_test_p30_Kvasirbest_"
    "bs1_random_shot1_e100_FYP-best.pth",
))
OUT_DIR = Path(os.environ.get(
    "FYP_OUT_DIR", "E:/Project/FYP/results/pp_sam_baseline/eval_best"))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
INFER_PERTURB = 30           # PP-SAM paper's protocol
IMG_SIZE = 1024


# --------------------------------------------------------------------------- #
# I/O
# --------------------------------------------------------------------------- #
def _read_image_bytes(path: Path) -> bytes:
    """Read a binary file, trying MSYS ``cat`` first and falling back to a direct
    Python read.

    The dataset ACL that originally motivated ``cat`` was re-measured as inactive
    on 2026-09-30, and ``cat`` is not on the PowerShell PATH
    (``C:\\Program Files\\Git\\usr\\bin``), so the direct read is the common path
    now while ``cat`` is kept for environments where it still works.
    """
    import subprocess
    try:
        r = subprocess.run(["cat", str(path)], capture_output=True)
        if r.returncode == 0 and r.stdout:
            return r.stdout
    except (FileNotFoundError, OSError):
        pass
    with open(path, "rb") as f:
        return f.read()


def load_image(path: Path) -> np.ndarray:
    data = _read_image_bytes(path)
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise IOError(f"cv2.imdecode failed for {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def load_mask(path: Path) -> np.ndarray:
    data = _read_image_bytes(path)
    arr = np.frombuffer(data, dtype=np.uint8)
    m = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
    if m is None:
        raise IOError(f"cv2.imdecode (gray) failed for {path}")
    return (m > 127).astype(np.uint8)


def gt_box_from_mask(mask: np.ndarray, perturb: int = INFER_PERTURB) -> Optional[np.ndarray]:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    H, W = mask.shape
    return np.array([
        max(0, int(xs.min()) - perturb),
        max(0, int(ys.min()) - perturb),
        min(W, int(xs.max()) + perturb),
        min(H, int(ys.max()) + perturb),
    ], dtype=np.int32)


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def dice_score(pred: np.ndarray, gt: np.ndarray) -> float:
    p, g = pred.astype(bool), gt.astype(bool)
    inter = np.logical_and(p, g).sum()
    denom = p.sum() + g.sum()
    if denom == 0:
        return 1.0
    return float(2.0 * inter / denom)


def iou_score(pred: np.ndarray, gt: np.ndarray) -> float:
    p, g = pred.astype(bool), gt.astype(bool)
    inter = np.logical_and(p, g).sum()
    union = np.logical_or(p, g).sum()
    if union == 0:
        return 1.0
    return float(inter / union)


def precision_recall(pred: np.ndarray, gt: np.ndarray) -> Tuple[float, float]:
    p, g = pred.astype(bool), gt.astype(bool)
    tp = np.logical_and(p, g).sum()
    fp = np.logical_and(p, ~g).sum()
    fn = np.logical_and(~p, g).sum()
    prec = float(tp / (tp + fp)) if (tp + fp) else 1.0
    rec = float(tp / (tp + fn)) if (tp + fn) else 1.0
    return prec, rec


def hd95_score(pred: np.ndarray, gt: np.ndarray) -> float:
    """95th-percentile Hausdorff distance in pixels. NaN if empty."""
    p, g = pred.astype(bool), gt.astype(bool)
    if p.sum() == 0 or g.sum() == 0:
        return float("nan")
    p_pts = np.argwhere(p)
    g_pts = np.argwhere(g)
    fwd = directed_hausdorff(p_pts, g_pts)[0]
    bwd = directed_hausdorff(g_pts, p_pts)[0]
    return float(max(fwd, bwd))


# --------------------------------------------------------------------------- #
# Dataset discovery
# --------------------------------------------------------------------------- #
def _read_split_stems(split_file: Path) -> List[str]:
    """Read a Data-split/*.txt file as a list of stems (``cat`` first, direct read
    as fallback -- see ``_read_image_bytes`` for why the fallback matters)."""
    import subprocess
    text = None
    try:
        r = subprocess.run(["cat", str(split_file)], capture_output=True, text=True)
        if r.returncode == 0 and r.stdout:
            text = r.stdout
    except (FileNotFoundError, OSError):
        pass
    if text is None:
        text = Path(split_file).read_text(encoding="utf-8-sig")
    return [line.strip() for line in text.splitlines() if line.strip()]


def discover_kvasir_val() -> List[Tuple[Path, Path]]:
    """Kvasir-SEG official val split (120 imgs, never seen by 1-shot training
    because train.txt has 880 different names)."""
    val_txt = DATA_ROOT / "Kvasir-SEG" / "Data-split" / "val.txt"
    img_dir = DATA_ROOT / "Kvasir-SEG" / "images"
    msk_dir = DATA_ROOT / "Kvasir-SEG" / "masks"
    if not val_txt.exists():
        raise FileNotFoundError(val_txt)
    pairs = []
    for stem in _read_split_stems(val_txt):
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
    """CVC-ClinicDB: 612 imgs, cross-domain test (unseen during 1-shot training)."""
    img_dir = DATA_ROOT / "CVC-ClinicDB" / "PNG" / "Original"
    msk_dir = DATA_ROOT / "CVC-ClinicDB" / "PNG" / "Ground Truth"
    if not img_dir.exists():
        raise FileNotFoundError(img_dir)
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
def save_triptych(image, gt, pred, out_path, dice):
    h, w = image.shape[:2]
    gt_o = image.copy()
    pr_o = image.copy()
    gt_o[gt > 0] = (gt_o[gt > 0] * 0.4 + np.array([0, 255, 0]) * 0.6).astype(np.uint8)
    pr_o[pred > 0] = (pr_o[pred > 0] * 0.4 + np.array([255, 0, 0]) * 0.6).astype(np.uint8)
    panel = np.concatenate([image, gt_o, pr_o], axis=1)
    sep = np.full((h, 4, 3), 0, dtype=np.uint8)
    panel = np.concatenate([panel[:, :w], sep, panel[:, w:2 * w], sep, panel[:, 2 * w:]], axis=1)
    cv2.putText(panel, f"Dice={dice:.3f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
    scale = 1200 / panel.shape[1]
    panel = cv2.resize(panel, (1200, int(panel.shape[0] * scale)))
    cv2.imwrite(str(out_path), cv2.cvtColor(panel, cv2.COLOR_RGB2BGR))


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--limit", type=int, default=None, help="cap images per dataset")
    p.add_argument("--dataset", type=str, default=None, choices=list(DATASETS.keys()),
                   help="only evaluate on this dataset (default: all)")
    p.add_argument("--no-triptych", action="store_true", help="skip 3-panel visualisations")
    p.add_argument("--box-perturb", type=int, default=30,
                   help="GT box outward perturbation in pixels (v3 default 30)")
    return p.parse_args()


def main():
    args = parse_args()
    print(f"[Config] device={DEVICE}  ckpt={CKPT.name}")
    print(f"[Config] infer_perturb={args.box_perturb}  out_dir={OUT_DIR}")

    if not CKPT.exists():
        raise FileNotFoundError(CKPT)

    # Load SAM architecture, then drop PPSAM weights in.
    # PPSAM script names the arch key 'vit_b'; checkpoint stores 'image_encoder.'
    # etc. The same key layout as vanilla SAM, so direct load_state_dict works.
    sam = sam_model_registry["vit_b"](checkpoint=str(SAM_CKPT))
    state = torch.load(str(CKPT), map_location="cpu", weights_only=True)
    sam.load_state_dict(state, strict=False)
    sam.to(DEVICE)
    sam.eval()
    predictor = SamPredictor(sam)
    print(f"[OK] PPSAM best loaded.\n")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary = {}

    datasets = [args.dataset] if args.dataset else list(DATASETS.keys())
    for ds_name in datasets:
        pairs = DATASETS[ds_name]()
        if not pairs:
            print(f"[SKIP] {ds_name}: no images found")
            continue
        if args.limit:
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
            dices.append(d)
            ious.append(j)
            precs.append(pr)
            recs.append(rc)
            hd95s.append(h)
            per_image.append({
                "name": img_path.name,
                "dice": d, "iou": j, "precision": pr, "recall": rc,
                "hd95": h, "fps": 1.0 / (elapsed + 1e-8),
            })

            Image.fromarray((pred * 255).astype(np.uint8)).save(pred_dir / img_path.name)
            if not args.no_triptych:
                save_triptych(image, gt, pred, trip_dir / f"{img_path.stem}.jpg", d)
            print(f"  [{i+1:3d}/{len(pairs)}] {img_path.name}  D={d:.4f} J={j:.4f} H={h:.2f} ({elapsed:.2f}s)")

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
            print(f"\n  >>> {ds_name}: Dice={np.mean(dices):.4f}±{np.std(dices):.4f}  "
                  f"IoU={np.mean(ious):.4f}  HD95={np.nanmean(hd95s):.2f}\n")

    compact = {k: {kk: vv for kk, vv in v.items() if kk != "per_image"}
               for k, v in summary.items()}

    # MERGE with existing summary.json (don't overwrite prior splits).
    summary_path = OUT_DIR / "summary.json"
    per_image_path = OUT_DIR / "per_image.json"
    existing_summary = {}
    existing_per_image = {}
    if summary_path.exists():
        try:
            existing_summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except Exception:
            existing_summary = {}
    if per_image_path.exists():
        try:
            existing_per_image = json.loads(per_image_path.read_text(encoding="utf-8"))
        except Exception:
            existing_per_image = {}
    merged_compact = {**existing_summary, **compact}
    merged_full = {**existing_per_image, **summary}

    with open(summary_path, "w") as f:
        json.dump(merged_compact, f, indent=2)
    with open(per_image_path, "w") as f:
        json.dump(merged_full, f, indent=2)
    print(f"[DONE] merged -> {summary_path}")
    print(json.dumps(merged_compact, indent=2))


if __name__ == "__main__":
    main()
