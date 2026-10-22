"""Eval polyp-trained MSA on Kvasir val (+ optional CVC) with v3 box±30.
Builds SAM+Adapter at the training image_size and loads the full train checkpoint.
"""
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

MSA_REPO = Path(os.environ.get("FYP_MSA_REPO", r"E:/Project/FYP/材料/repos/Medical-SAM-Adapter"))
sys.path.insert(0, str(MSA_REPO))

# Reuse infer helpers
CODE = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE))

import infer as base

import torch
import torch.nn.functional as F
from models.sam.build_sam import build_sam_vit_b

IMAGE_SIZE = int(os.environ.get("FYP_MSA_IMAGE_SIZE", "512"))
CKPT = Path(os.environ.get(
    "FYP_MSA_CKPT",
    r"E:/Project/FYP/checkpoints/medical_sam_adapter/polyp/best_dice_checkpoint.pth",
))
OUT_DIR = Path(os.environ.get("FYP_OUT_DIR", r"E:/Project/FYP/results/msa_polyp_train/eval_v3"))
SAM_CKPT = Path(os.environ.get("FYP_SAM_CKPT", r"E:/Project/FYP/checkpoints/sam_vit_b.pth"))
LIMIT = int(os.environ["FYP_LIMIT"]) if os.environ.get("FYP_LIMIT") else None
DATASETS = os.environ.get("FYP_DATASETS", "Kvasir-SEG-val").split(",")


def build_model():
    args = SimpleNamespace(
        mod="sam_adpt",
        image_size=IMAGE_SIZE,
        multimask_output=3,
        mid_dim=None,
        b=8,
        dataset="kvasir",
        thd=False,
    )
    print(f"[Build] base={SAM_CKPT} image_size={IMAGE_SIZE}")
    sam = build_sam_vit_b(args=args, checkpoint=str(SAM_CKPT))
    raw = torch.load(str(CKPT), map_location="cpu", weights_only=False)
    sd = raw["state_dict"] if isinstance(raw, dict) and "state_dict" in raw else raw
    # strip DataParallel prefix if any
    sd = { (k[7:] if k.startswith("module.") else k): v for k, v in sd.items() }
    missing, unexpected = sam.load_state_dict(sd, strict=False)
    print(f"[Build] load missing={len(missing)} unexpected={len(unexpected)} epoch={raw.get('epoch') if isinstance(raw, dict) else '?'}")
    # Prefer full load: if many missing, also try adapter-only onto freshly loaded base
    if len(missing) > 50:
        print("[Build] many missing; falling back to adapter-only load on base SAM")
        sam = build_sam_vit_b(args=args, checkpoint=str(SAM_CKPT))
        adapt = {k: v for k, v in sd.items() if "Adapter" in k}
        missing, unexpected = sam.load_state_dict(adapt, strict=False)
        print(f"[Build] adapter-only missing={len(missing)} unexpected={len(unexpected)} n_adapt={len(adapt)}")
    sam.to(base.DEVICE)
    sam.eval()
    return sam


@torch.no_grad()
def msa_infer_resized(sam, image_rgb, box_xyxy, image_size=IMAGE_SIZE):
    """Resize image+box to model image_size, infer, then resize mask back."""
    import cv2
    import numpy as np
    H, W = image_rgb.shape[:2]
    img_r = cv2.resize(image_rgb, (image_size, image_size), interpolation=cv2.INTER_LINEAR)
    sx, sy = image_size / W, image_size / H
    box_r = np.array([
        box_xyxy[0] * sx, box_xyxy[1] * sy,
        box_xyxy[2] * sx, box_xyxy[3] * sy,
    ], dtype=np.float32)
    img_t = torch.as_tensor(img_r, dtype=torch.float, device=base.DEVICE).permute(2, 0, 1).unsqueeze(0)
    imgs = sam.preprocess(img_t)
    img_emb = sam.image_encoder(imgs)
    box = torch.as_tensor(box_r, dtype=torch.float, device=base.DEVICE)[None, :]
    se, de = sam.prompt_encoder(points=None, boxes=box, masks=None)
    low_res, iou_preds = sam.mask_decoder(
        image_embeddings=img_emb,
        image_pe=sam.prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings=se,
        dense_prompt_embeddings=de,
        multimask_output=True,
    )
    best_idx = iou_preds.argmax(dim=1)[0].item()
    pred = F.interpolate(low_res[:, best_idx:best_idx + 1], size=(image_size, image_size), mode="bilinear", align_corners=False)
    pred = torch.sigmoid(pred).squeeze().cpu().numpy()
    pred = (pred > 0.5).astype("uint8")
    pred = cv2.resize(pred, (W, H), interpolation=cv2.INTER_NEAREST)
    return pred


def main():
    import argparse
    import numpy as np
    from PIL import Image

    ap = argparse.ArgumentParser()
    ap.add_argument("--box-perturb", type=int, default=30,
                    help="GT box outward perturbation in pixels (v3 default 30)")
    args = ap.parse_args()
    box_perturb = args.box_perturb

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[Config] ckpt={CKPT} out={OUT_DIR} datasets={DATASETS} limit={LIMIT} box_perturb={box_perturb}")
    sam = build_model()
    summary = {}
    per_all = {}

    def _pairs(ds_name):
        root = base.DATA_ROOT
        if ds_name == "Kvasir-SEG-val":
            stems = [ln.strip() for ln in (root / "Kvasir-SEG" / "Data-split" / "val.txt").read_text(encoding="utf-8").splitlines() if ln.strip()]
            img_dir, msk_dir = root / "Kvasir-SEG" / "images", root / "Kvasir-SEG" / "masks"
            out = []
            for s in stems:
                for ext in (".jpg", ".png"):
                    img = img_dir / f"{s}{ext}"
                    msk = msk_dir / f"{s}{ext}"
                    if img.exists() and msk.exists():
                        out.append((img, msk)); break
            return out
        if ds_name == "CVC-ClinicDB":
            img_dir = root / "CVC-ClinicDB" / "PNG" / "Original"
            msk_dir = root / "CVC-ClinicDB" / "PNG" / "Ground Truth"
            out = []
            for img in sorted(img_dir.iterdir()):
                if img.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
                    continue
                msk = msk_dir / img.name
                if msk.exists():
                    out.append((img, msk))
            return out
        raise KeyError(ds_name)

    def _load_image(path):
        import cv2, numpy as np
        data = Path(path).read_bytes()
        arr = np.frombuffer(data, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def _load_mask(path):
        import cv2, numpy as np
        data = Path(path).read_bytes()
        arr = np.frombuffer(data, dtype=np.uint8)
        m = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
        return (m > 127).astype(np.uint8)

    for ds_name in DATASETS:
        ds_name = ds_name.strip()
        try:
            pairs = _pairs(ds_name)
        except Exception as e:
            print(f"[SKIP] {ds_name}: {e}")
            continue
        if LIMIT:
            pairs = pairs[:LIMIT]
        print(f"=== {ds_name}: {len(pairs)} ===")
        pred_dir = OUT_DIR / ds_name / "preds"
        pred_dir.mkdir(parents=True, exist_ok=True)
        dices, ious, precs, recs, hd95s, fps_list = [], [], [], [], [], []
        per_image = []
        for i, (img_path, msk_path) in enumerate(pairs):
            image = _load_image(img_path)
            gt_mask = _load_mask(msk_path)
            if gt_mask.sum() == 0:
                continue
            box = base.gt_box_from_mask(gt_mask, perturb=box_perturb)
            t0 = time.time()
            pred_mask = msa_infer_resized(sam, image, box)
            elapsed = time.time() - t0
            d = base.dice_score(pred_mask, gt_mask)
            j = base.iou_score(pred_mask, gt_mask)
            pr, rc = base.precision_recall(pred_mask, gt_mask)
            h = base.hd95_score(pred_mask, gt_mask)
            dices.append(d); ious.append(j); precs.append(pr); recs.append(rc); hd95s.append(h)
            fps_list.append(1.0 / (elapsed + 1e-8))
            per_image.append({"name": img_path.name, "dice": d, "iou": j, "precision": pr, "recall": rc, "hd95": h})
            Image.fromarray((pred_mask * 255).astype("uint8")).save(pred_dir / img_path.name)
            if (i + 1) % 20 == 0 or i == 0:
                print(f"  [{i+1}/{len(pairs)}] {img_path.name} D={d:.4f} J={j:.4f} ({elapsed:.2f}s)")
        summary[ds_name] = {
            "n": len(dices),
            "dice_mean": float(np.mean(dices)),
            "dice_std": float(np.std(dices)),
            "iou_mean": float(np.mean(ious)),
            "iou_std": float(np.std(ious)),
            "precision_mean": float(np.mean(precs)),
            "recall_mean": float(np.mean(recs)),
            "hd95_mean": float(np.nanmean(hd95s)),
            "fps_mean": float(np.mean(fps_list)),
            "image_size": IMAGE_SIZE,
            "ckpt": str(CKPT),
            "box_perturb": box_perturb,
        }
        per_all[ds_name] = per_image
        print(f">>> {ds_name}: Dice={summary[ds_name]['dice_mean']:.4f}±{summary[ds_name]['dice_std']:.4f} IoU={summary[ds_name]['iou_mean']:.4f}")

    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (OUT_DIR / "per_image.json").write_text(json.dumps(per_all, indent=2), encoding="utf-8")
    print("[DONE]", OUT_DIR / "summary.json")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()