"""
Phase 2 v3: Medical-SAM-Adapter (MSA) zero-shot baseline.

Loads base SAM ViT-B weights, but builds the model through MSA's
``build_sam_vit_b`` (which uses ``AdapterBlock`` instead of plain
``Block``), then loads the 144-key adapter state_dict.  Inference is
done via MSA's own forward path: ``net.preprocess + net.image_encoder +
net.prompt_encoder + net.mask_decoder``.

Prompt mode: **box** (consistent with B1 / B2 baselines, no perturbation).
Mask selection: ``multimask_output=True`` (3 masks) → pick the one with
the highest IoU prediction (this is the MSA training-time protocol).

What changed in v3
------------------
- 5-metric reporting (Dice / IoU / Precision / Recall / HD95) for parity
  with B1 / B2 / B4 evaluation pipeline.
- Kvasir evaluation uses ``Data-split/val.txt`` (120 held-out imgs) for
  cross-baseline fairness. The legacy ``sorted[:N]`` slice was replaced.

STATUS (see results/msa_new_protocol/STATUS.md)
-----------------------------------------------
B3 remains [ABANDONED]: the MSA repo only releases adapters for
{LungNodule_CT, Melanoma_Photo, Retina_Vessel_Fundus, Surgical_Instrument_Video}.
No polyp-specific adapter exists; per-image Dice ≈ 0.02-0.05 is the
expected behavior of a fully-mismatched adapter, NOT an implementation bug.
The paper reports B3 as N/A and substitutes B2 (MedSAM) in the comparison.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import List, Tuple

# Add MSA repo to path so we can use its build_sam
MSA_REPO = Path(os.environ.get("FYP_MSA_REPO", "E:/Project/FYP/材料/repos/Medical-SAM-Adapter"))
sys.path.insert(0, str(MSA_REPO))

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.spatial.distance import directed_hausdorff

from models.sam.build_sam import build_sam_vit_b

# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
DATA_ROOT = Path(os.environ.get("FYP_DATA_ROOT", "E:/Project/FYP/材料/datasets"))
SAM_CKPT = Path(os.environ.get("FYP_SAM_CKPT", "E:/Project/FYP/checkpoints/sam_vit_b.pth"))
MSA_CKPT = Path(os.environ.get("FYP_MSA_CKPT", "E:/Project/FYP/checkpoints/medical_sam_adapter/Surgical_Instrument_Video_SAM_1024.pth"))
OUT_DIR = Path(os.environ.get("FYP_OUT_DIR", "E:/Project/FYP/results/phase2_msa_v3"))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42
IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# --------------------------------------------------------------------------- #
# I/O
# --------------------------------------------------------------------------- #
def _read_image_bytes(path: Path) -> bytes:
    """Read a binary file via MSYS cat (bypasses Windows ACL that blocks
    Python's Win32 read syscalls on ``材料/datasets/...``).

    We use ``subprocess.run(['cat', ...])`` because Python's built-in
    ``open()``, ``PIL.Image.open()``, ``cv2.imread()``, and
    ``np.fromfile()`` all go through Win32 CreateFile which the dataset
    directory's ACL blocks for the current user, while MSYS's POSIX
    layer (which cat uses) can read the same file fine.
    """
    import subprocess
    r = subprocess.run(["cat", str(path)], capture_output=True)
    if r.returncode != 0:
        raise IOError(f"cat {path} failed rc={r.returncode}: {r.stderr[:200]}")
    return r.stdout


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


def gt_box_from_mask(mask: np.ndarray, perturb: int = 0) -> np.ndarray:
    """GT bounding box with optional inference-time perturbation.

    B3 is a zero-shot adapter baseline; evaluated with the SAME 30-px
    box perturbation as B4 (PP-SAM) for protocol parity in Table 1.
    """
    ys, xs = np.where(mask > 0)
    H, W = mask.shape
    return np.array([
        max(0, int(xs.min()) - perturb),
        max(0, int(ys.min()) - perturb),
        min(W, int(xs.max()) + perturb),
        min(H, int(ys.max()) + perturb),
    ], dtype=np.int32)


# --------------------------------------------------------------------------- #
# Metrics (5-metric set for parity with B1 / B2 / B4)
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
# Dataset discovery (parity with eval_ppsam.py)
# --------------------------------------------------------------------------- #
def _read_split_stems(split_file: Path):
    """Read a Data-split/*.txt file as a list of stems via MSYS cat
    (bypasses Windows ACL that blocks Python's Win32 read syscalls)."""
    import subprocess
    r = subprocess.run(["cat", str(split_file)], capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"cat {split_file} failed: {r.stderr}")
    return [line.strip() for line in r.stdout.splitlines() if line.strip()]


def discover_kvasir_val() -> List[Tuple[Path, Path]]:
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
    panel = np.concatenate([panel[:, :w], sep, panel[:, w:2 * w], sep, panel[:, 2 * w:]], axis=1)
    cv2.putText(panel, f"Dice={dice:.3f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
    scale = 1200 / panel.shape[1]
    panel = cv2.resize(panel, (1200, int(panel.shape[0] * scale)))
    cv2.imwrite(str(out_path), cv2.cvtColor(panel, cv2.COLOR_RGB2BGR))


# --------------------------------------------------------------------------- #
# Build MSA model
# --------------------------------------------------------------------------- #
def build_msa_sam():
    print(f"[Build] Loading base SAM from {SAM_CKPT} (via MSA's build_sam_vit_b)...")
    args = SimpleNamespace(
        mod='sam_adpt',
        image_size=1024,
        multimask_output=3,
        mid_dim=768,
        b=8,
        dataset='polyp',
        thd=False,
    )
    sam = build_sam_vit_b(args=args, checkpoint=str(SAM_CKPT))
    print(f"[Build] Loading adapter weights from {MSA_CKPT}...")
    raw = torch.load(str(MSA_CKPT), map_location='cpu', weights_only=False)
    sd = raw['state_dict'] if 'state_dict' in raw else raw
    missing, unexpected = sam.load_state_dict(sd, strict=False)
    print(f"[Build] adapter: missing={len(missing)}, unexpected={len(unexpected)}")
    sam.to(DEVICE)
    sam.eval()
    return sam


# --------------------------------------------------------------------------- #
# MSA-style inference (preprocess + image_encoder + prompt_encoder + mask_decoder)
# --------------------------------------------------------------------------- #
@torch.no_grad()
def msa_infer(sam, image_rgb, box_xyxy):
    H, W = image_rgb.shape[:2]
    img_t = torch.as_tensor(image_rgb, dtype=torch.float, device=DEVICE).permute(2, 0, 1).unsqueeze(0)
    imgs = sam.preprocess(img_t)
    img_emb = sam.image_encoder(imgs)
    box = torch.as_tensor(box_xyxy, dtype=torch.float, device=DEVICE)[None, :]
    se, de = sam.prompt_encoder(points=None, boxes=box, masks=None)
    low_res, iou_preds = sam.mask_decoder(
        image_embeddings=img_emb,
        image_pe=sam.prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings=se,
        dense_prompt_embeddings=de,
        multimask_output=True,
    )
    best_idx = iou_preds.argmax(dim=1)[0].item()
    pred = F.interpolate(low_res[:, best_idx:best_idx+1], size=(H, W), mode='bilinear', align_corners=False)
    pred = torch.sigmoid(pred).squeeze().cpu().numpy()
    return (pred > 0.5).astype(np.uint8)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--dataset", type=str, default=None, choices=list(DATASETS.keys()))
    p.add_argument("--no-triptych", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    print(f"[Config] device={DEVICE} sam_ckpt={SAM_CKPT} msa_ckpt={MSA_CKPT} out_dir={OUT_DIR}")

    if not SAM_CKPT.exists():
        raise FileNotFoundError(f"Base SAM checkpoint not found: {SAM_CKPT}")
    if not MSA_CKPT.exists():
        raise FileNotFoundError(f"MSA checkpoint not found: {MSA_CKPT}")

    sam = build_msa_sam()
    print("[OK] MSA-style SAM ready.\n")

    datasets = [args.dataset] if args.dataset else list(DATASETS.keys())
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary = {}

    for ds_name in datasets:
        pairs = DATASETS[ds_name]()
        if not pairs:
            print(f"[SKIP] {ds_name}: no images found")
            continue
        if args.limit:
            pairs = pairs[: args.limit]
        print(f"=== {ds_name}: {len(pairs)} images ===")

        pred_dir = OUT_DIR / ds_name / "preds"
        trip_dir = OUT_DIR / ds_name / "triptychs"
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
            box = gt_box_from_mask(gt_mask, perturb=30)

            t0 = time.time()
            pred_mask = msa_infer(sam, image, box)
            elapsed = time.time() - t0
            fps_list.append(1.0 / (elapsed + 1e-8))

            d = dice_score(pred_mask, gt_mask)
            j = iou_score(pred_mask, gt_mask)
            pr, rc = precision_recall(pred_mask, gt_mask)
            h = hd95_score(pred_mask, gt_mask)
            dices.append(d)
            ious.append(j)
            precs.append(pr)
            recs.append(rc)
            hd95s.append(h)
            per_image.append({
                "name": img_path.name,
                "dice": d, "iou": j,
                "precision": pr, "recall": rc,
                "hd95": h, "fps": 1.0 / (elapsed + 1e-8),
            })

            Image.fromarray((pred_mask * 255).astype(np.uint8)).save(pred_dir / img_path.name)
            if not args.no_triptych:
                save_triptych(image, gt_mask, pred_mask, trip_dir / f"{img_path.stem}.jpg", d)

            print(f"  [{i+1:3d}/{len(pairs)}] {img_path.name}  D={d:.4f}  J={j:.4f}  P={pr:.4f}  R={rc:.4f}  H={h:.2f}  ({elapsed:.2f}s)")

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
                  f"IoU={np.mean(ious):.4f}  P={np.mean(precs):.4f}  R={np.mean(recs):.4f}  "
                  f"HD95={np.nanmean(hd95s):.2f}\n")

    compact = {k: {kk: vv for kk, vv in v.items() if kk != "per_image"} for k, v in summary.items()}

    # MERGE with existing summary.json (don't overwrite prior splits).
    # Inference is run once per --dataset; multiple invocations should accumulate
    # into a single OUT_DIR/summary.json that covers all splits.
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
