"""
SAMed training script — LoRA rank-4 adaptation of SAM for polyp segmentation.

Protocol: FYP v3 (train.txt 880, val.txt 120 held-out, box ±30px inference, 5 metrics)

Architecture (faithful to SAMed paper, Zhang & Liu 2023):
- LoRA rank 4 on image encoder qkv (Q and V projections)
- Freeze image encoder backbone; train LoRA + prompt_encoder + mask_decoder
- Loss: CE + Dice (0.8 weight on Dice) — SAMed's original loss
- Optimizer: AdamW, lr=0.005, weight_decay=0.1, warmup 250 iters, poly decay
- Image size: 1024 (SAM native) — SAMed uses 512 but we keep 1024 for protocol parity

Key difference from PP-SAM:
- PP-SAM: freeze mask_decoder, train image_encoder + prompt_encoder (full encoder)
- SAMed:   freeze image_encoder (LoRA only), train prompt_encoder + mask_decoder

Few-shot extension (2026-06-24):
- ``--shots K`` randomly samples K images from train.txt 880 (matches B4 PP-SAM K-shot
  protocol, same seed=1234 so 1-shot picks the same single image PP-SAM uses).
- ``--shots -1`` (default) uses the full 880 (SAMed paper's fully-supervised protocol).
- ``--smoke`` runs 1 epoch on 5 samples for pipeline validation.

Usage:
    python train_samed.py --smoke                        # pipeline smoke test
    python train_samed.py --shots 1                      # 1-shot LoRA (cf. B4 PP-SAM 1-shot)
    python train_samed.py --shots 5                      # 5-shot LoRA
    python train_samed.py --rank 4 --epochs 200          # full SAMed protocol
"""
import argparse
import json
import os
import random
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, 'E:/Project/FYP/材料/repos/PP-SAM')
from segment_anything import sam_model_registry, SamPredictor
from segment_anything.utils.transforms import ResizeLongestSide

from lora_sam import LoRA_Sam

# ── Config ────────────────────────────────────────────────────────────────── #
DATA_ROOT = Path(os.environ.get("FYP_DATA_ROOT", "E:/Project/FYP/材料/datasets"))
SAM_CKPT = Path(os.environ.get("FYP_SAM_CKPT", "E:/Project/FYP/checkpoints/sam_vit_b.pth"))
CKPT_DIR = Path(os.environ.get("FYP_CKPT_DIR", "E:/Project/FYP/checkpoints/samed"))
OUT_DIR = Path(os.environ.get("FYP_OUT_DIR", "E:/Project/FYP/results/samed_baseline"))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 1234  # SAMed's default seed

# SAMed paper hyperparameters
LR = 0.005
WEIGHT_DECAY = 0.1
WARMUP_PERIOD = 250
DICE_WEIGHT = 0.8
CLIP = 0.5
IMG_SIZE = 1024
INFER_PERTURB = 30

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ── Loss (SAMed's CE + Dice) ──────────────────────────────────────────────── #
def dice_loss(pred, target, smooth=1.0):
    """Soft Dice loss for binary segmentation.
    pred: [B, 1, H, W] logits (pre-sigmoid)
    target: [B, 1, H, W] binary
    """
    pred_sig = torch.sigmoid(pred)
    inter = (pred_sig * target).sum(dim=(2, 3))
    union = pred_sig.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
    return 1.0 - ((2.0 * inter + smooth) / (union + smooth)).mean()


def calc_loss(pred_logits, gt_mask):
    """SAMed's loss: (1 - dice_weight) * CE + dice_weight * Dice"""
    bce = F.binary_cross_entropy_with_logits(pred_logits, gt_mask)
    dl = dice_loss(pred_logits, gt_mask)
    return (1 - DICE_WEIGHT) * bce + DICE_WEIGHT * dl, bce.item(), dl.item()


# ── Utilities ─────────────────────────────────────────────────────────────── #
def clip_gradient(optimizer, grad_clip):
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)


def _read_image_bytes(path):
    import subprocess
    r = subprocess.run(["cat", str(path)], capture_output=True)
    if r.returncode != 0:
        raise IOError(f"cat {path} failed rc={r.returncode}")
    return r.stdout


def _imread(path, mode=cv2.IMREAD_COLOR):
    data = _read_image_bytes(path)
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, mode)
    if img is None:
        raise IOError(f"cv2.imdecode failed for {path}")
    return img


def _read_split_file(split_file, img_dir, msk_dir):
    import subprocess
    r = subprocess.run(["cat", str(split_file)], capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"cat {split_file} failed: {r.stderr}")
    pairs = []
    for line in r.stdout.splitlines():
        stem = line.strip()
        if not stem:
            continue
        img = img_dir / f"{stem}.jpg"
        if not img.exists():
            img = img_dir / f"{stem}.png"
        msk = msk_dir / f"{stem}.jpg"
        if not msk.exists():
            msk = msk_dir / f"{stem}.png"
        if img.exists() and msk.exists():
            pairs.append((img, msk))
    return pairs


def load_kvasir_train_paths():
    img_dir = DATA_ROOT / "Kvasir-SEG" / "images"
    msk_dir = DATA_ROOT / "Kvasir-SEG" / "masks"
    split_file = DATA_ROOT / "Kvasir-SEG" / "Data-split" / "train.txt"
    return _read_split_file(split_file, img_dir, msk_dir)


def load_kvasir_val_pairs():
    img_dir = DATA_ROOT / "Kvasir-SEG" / "images"
    msk_dir = DATA_ROOT / "Kvasir-SEG" / "masks"
    split_file = DATA_ROOT / "Kvasir-SEG" / "Data-split" / "val.txt"
    return _read_split_file(split_file, img_dir, msk_dir)


def load_clinicdb_paths():
    img_dir = DATA_ROOT / "CVC-ClinicDB" / "PNG" / "Original"
    msk_dir = DATA_ROOT / "CVC-ClinicDB" / "PNG" / "Ground Truth"
    images = sorted([p for p in img_dir.iterdir() if p.suffix.lower() in {".jpg", ".png"}])
    return [(img, msk_dir / img.name) for img in images if (msk_dir / img.name).exists()]


def dice_score(pred, gt):
    p, g = pred.astype(bool), gt.astype(bool)
    inter = np.logical_and(p, g).sum()
    denom = p.sum() + g.sum()
    if denom == 0:
        return 1.0
    return float(2.0 * inter / denom)


# ── Validation ────────────────────────────────────────────────────────────── #
def validate(model, pairs, infer_perturb=INFER_PERTURB):
    """Evaluate with box prompt ±30px on held-out set."""
    model.eval()
    predictor = SamPredictor(model)
    from scipy.spatial.distance import directed_hausdorff
    dices, ious, precs, recs, hd95s = [], [], [], [], []
    for img_path, msk_path in pairs:
        image = _imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = _imread(msk_path, cv2.IMREAD_GRAYSCALE)
        mask = (mask > 127).astype(np.uint8)
        H, W = mask.shape
        ys, xs = np.where(mask > 0)
        if len(xs) == 0:
            continue
        x_min = max(0, xs.min() - infer_perturb)
        x_max = min(W, xs.max() + infer_perturb)
        y_min = max(0, ys.min() - infer_perturb)
        y_max = min(H, ys.max() + infer_perturb)
        box = np.array([x_min, y_min, x_max, y_max])

        predictor.set_image(image)
        masks, scores, _ = predictor.predict(
            point_coords=None, point_labels=None,
            box=box[None, :], multimask_output=False,
        )
        pred = masks[0].astype(np.uint8)

        p_bool, g_bool = pred.astype(bool), mask.astype(bool)
        inter = np.logical_and(p_bool, g_bool).sum()
        denom = p_bool.sum() + g_bool.sum()
        d = float(2.0 * inter / denom) if denom else 1.0
        dices.append(d)

        union = np.logical_or(p_bool, g_bool).sum()
        ious.append(float(inter / union) if union else 1.0)

        tp = inter
        fp = np.logical_and(p_bool, ~g_bool).sum()
        fn = np.logical_and(~p_bool, g_bool).sum()
        precs.append(float(tp / (tp + fp)) if (tp + fp) else 1.0)
        recs.append(float(tp / (tp + fn)) if (tp + fn) else 1.0)

        if p_bool.sum() and g_bool.sum():
            p_pts = np.argwhere(p_bool)
            g_pts = np.argwhere(g_bool)
            fwd = directed_hausdorff(p_pts, g_pts)[0]
            bwd = directed_hausdorff(g_pts, p_pts)[0]
            hd95s.append(float(max(fwd, bwd)))

    extras = {
        "iou": float(np.mean(ious)) if ious else 0.0,
        "precision": float(np.mean(precs)) if precs else 0.0,
        "recall": float(np.mean(recs)) if recs else 0.0,
        "hd95": float(np.nanmean(hd95s)) if hd95s else float("nan"),
    }
    return float(np.mean(dices)) if dices else 0.0, len(dices), extras


# ── Training ──────────────────────────────────────────────────────────────── #
def train_samed(rank=4, epochs=200, stop_epoch=160, shots=-1, smoke=False):
    """Train SAMed LoRA on Kvasir-SEG train.txt.

    Args:
        rank: LoRA rank (SAMed paper default 4)
        epochs: max training epochs
        stop_epoch: early-stop epoch (SAMed default 160)
        shots: K-shot sampling. -1 = full 880 (paper protocol).
               K > 0: randomly sample K imgs from train.txt for few-shot LoRA.
        smoke: pipeline smoke test — forces 1 epoch on 5 samples.
    """
    if smoke:
        epochs, stop_epoch, shots = 1, 1, 5
        print("[Smoke] epochs=1, samples=5 — pipeline validation only")

    # Tag output dirs by shots config so K-shot runs don't collide with full
    run_tag = f"r{rank}_smoke" if smoke else (
        f"r{rank}_full" if shots < 0 else f"r{rank}_{shots}shot"
    )
    ckpt_dir = CKPT_DIR / run_tag
    out_dir = OUT_DIR / run_tag
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[Out] ckpt_dir={ckpt_dir}  out_dir={out_dir}")

    # 1) Build model
    print(f"[Build] Loading SAM from {SAM_CKPT}...")
    sam = sam_model_registry["vit_b"](checkpoint=str(SAM_CKPT))
    lora_model = LoRA_Sam(sam, r=rank).to(DEVICE)

    trainable = lora_model.trainable_param_count
    total = sum(p.numel() for p in lora_model.parameters())
    print(f"[Build] LoRA rank={rank}. Trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

    # 2) Load training data (Kvasir train.txt 880) and K-shot sample if requested
    train_pairs = load_kvasir_train_paths()
    print(f"[Data] Kvasir train.txt: {len(train_pairs)} images available")

    if shots > 0:
        # Use SEED=1234 already set globally; sort first for determinism, then sample
        train_pairs = sorted(train_pairs, key=lambda x: x[0].name)
        random.shuffle(train_pairs)  # uses module-level SEED
        train_pairs = train_pairs[:shots]
        print(f"[Data] K-shot sampling: {shots} images selected (seed={SEED})")
        for p in train_pairs:
            print(f"        - {p[0].name}")

    # 3) Preprocess all training images into memory
    transform = ResizeLongestSide(IMG_SIZE)
    transformed_data = {}
    bbox_coords = {}
    masks_dict = {}

    for img_path, msk_path in train_pairs:
        image = _imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = _imread(msk_path, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
        H, W = mask.shape
        ys, xs = np.where(mask > 0)
        if len(xs) == 0:
            continue

        # SAMed uses no perturbation during training (prompt-free semantic seg)
        # But we use box prompt from GT (like PP-SAM) for prompt-based training
        # Use SAMed's approach: tight GT box (no perturbation during training)
        x_min, x_max = xs.min(), xs.max()
        y_min, y_max = ys.min(), ys.max()
        bbox_coords[img_path.name] = np.array([x_min, y_min, x_max, y_max])

        input_image = transform.apply_image(image)
        input_image_torch = torch.as_tensor(input_image, device=DEVICE)
        transformed_image = input_image_torch.permute(2, 0, 1).contiguous()[None, :, :, :]
        input_image = sam.preprocess(transformed_image)
        transformed_data[img_path.name] = {
            "image": input_image,
            "input_size": tuple(transformed_image.shape[-2:]),
            "original_image_size": image.shape[:2],
        }
        masks_dict[img_path.name] = mask

    print(f"[Data] Loaded {len(transformed_data)} samples into memory")

    # 4) Optimizer (SAMed: AdamW, lr=0.005, wd=0.1)
    params = filter(lambda p: p.requires_grad, lora_model.parameters())
    optimizer = torch.optim.AdamW(params, lr=LR / WARMUP_PERIOD, betas=(0.9, 0.999), weight_decay=WEIGHT_DECAY)
    print(f"[Opt] AdamW lr={LR} wd={WEIGHT_DECAY}, warmup={WARMUP_PERIOD} iters")

    # 5) Training loop
    val_pairs = load_kvasir_val_pairs()
    print(f"[Val] Kvasir val.txt: {len(val_pairs)} images (held-out)")

    best_dice = 0.0
    best_epoch = 0
    best_path = ckpt_dir / f"samed_lora_r{rank}_best.pth"
    last_path = ckpt_dir / f"samed_lora_r{rank}_last.pth"
    history = []
    total_start = time.time()
    iter_num = 0
    max_iterations = epochs * len(transformed_data)
    keys = list(transformed_data.keys())

    print(f"{'#' * 20} Start Training {'#' * 20}")
    print(f"[Config] epochs={epochs}, stop_epoch={stop_epoch}, samples/epoch={len(keys)}")
    print(f"[Config] max_iterations={max_iterations}, batch_size=1 (SAMed default for prompt-based)")

    for epoch in range(epochs):
        lora_model.train()
        random.shuffle(keys)
        epoch_losses = []

        for k in keys:
            d = transformed_data[k]
            input_image = d["image"].to(DEVICE)
            input_size = d["input_size"]
            original_image_size = d["original_image_size"]

            optimizer.zero_grad()

            # Forward: image encoder (with LoRA) + prompt encoder + mask decoder
            image_embedding = lora_model.sam.image_encoder(input_image)
            prompt_box = bbox_coords[k]
            box = transform.apply_boxes(prompt_box, original_image_size)
            box_torch = torch.as_tensor(box, dtype=torch.float, device=DEVICE)[None, :, None, :]
            sparse_embeddings, dense_embeddings = lora_model.sam.prompt_encoder(
                points=None, boxes=box_torch, masks=None,
            )
            low_res_masks, _ = lora_model.sam.mask_decoder(
                image_embeddings=image_embedding,
                image_pe=lora_model.sam.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=False,
            )
            gt_mask = torch.from_numpy(masks_dict[k]).to(DEVICE)
            gt_binary = (gt_mask > 0.0).float().unsqueeze(0).unsqueeze(0)
            upscaled = lora_model.sam.postprocess_masks(low_res_masks, input_size, original_image_size)

            loss, loss_ce, loss_dice = calc_loss(upscaled, gt_binary)
            loss.backward()
            clip_gradient(optimizer, CLIP)
            optimizer.step()

            # LR schedule: warmup then poly decay
            if iter_num < WARMUP_PERIOD:
                lr_ = LR * (iter_num + 1) / WARMUP_PERIOD
            else:
                shift = iter_num - WARMUP_PERIOD
                lr_ = LR * (1.0 - shift / max_iterations) ** 0.9
            for pg in optimizer.param_groups:
                pg['lr'] = lr_

            epoch_losses.append(loss.item())
            iter_num += 1

        avg_loss = float(np.mean(epoch_losses))

        # Validate every 5 epochs (faster than every epoch for 880 samples)
        if (epoch + 1) % 5 == 0 or epoch == 0 or epoch >= stop_epoch - 5:
            dice_kv, n_kv, kv_extras = validate(lora_model.sam, val_pairs)
        else:
            dice_kv = history[-1]["kvasir_val_dice"] if history else 0.0
            kv_extras = {"iou": 0, "precision": 0, "recall": 0, "hd95": 0}

        elapsed = time.time() - total_start
        history.append({
            "epoch": epoch + 1,
            "loss": avg_loss,
            "lr": lr_,
            "kvasir_val_dice": dice_kv,
            "kvasir_val_iou": kv_extras.get("iou", 0),
            "kvasir_val_precision": kv_extras.get("precision", 0),
            "kvasir_val_recall": kv_extras.get("recall", 0),
            "kvasir_val_hd95": kv_extras.get("hd95", 0),
            "n_val": n_kv if (epoch + 1) % 5 == 0 or epoch == 0 or epoch >= stop_epoch - 5 else 0,
            "elapsed_s": elapsed,
        })

        log = (f"[{datetime.now().strftime('%H:%M:%S')}] Epoch {epoch+1:3d}/{epochs}  "
               f"loss={avg_loss:.4f}  lr={lr_:.6f}  KvasirVal_Dice={dice_kv:.4f}  "
               f"({elapsed/60:.1f}min)")
        print(log, flush=True)

        if dice_kv > best_dice:
            best_dice = dice_kv
            best_epoch = epoch + 1
            lora_model.save_lora_parameters(str(best_path))
            print(f"  >>> New best Dice {best_dice:.4f} (epoch {epoch+1})")

        if (epoch + 1) % 20 == 0:
            lora_model.save_lora_parameters(str(last_path))

        if epoch + 1 >= stop_epoch:
            print(f"[Early stop] Reached stop_epoch={stop_epoch}")
            break

    lora_model.save_lora_parameters(str(last_path))
    print(f"\n[DONE] {len(history)} epochs in {(time.time()-total_start)/60:.1f} min")
    print(f"[DONE] Best Kvasir Dice: {best_dice:.4f} (epoch {best_epoch})")
    print(f"[DONE] Best ckpt: {best_path}")
    print(f"[DONE] Last ckpt: {last_path}")

    with open(ckpt_dir / f"samed_lora_r{rank}_history.json", "w") as f:
        json.dump(history, f, indent=2)
    return best_path, best_dice


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rank", type=int, default=4, help="LoRA rank (default 4, SAMed paper)")
    parser.add_argument("--epochs", type=int, default=200, help="max epochs (SAMed default 200)")
    parser.add_argument("--stop_epoch", type=int, default=160, help="early stop epoch (SAMed default 160)")
    parser.add_argument("--shots", type=int, default=-1,
                        help="K-shot training (e.g. 1/5/10); -1 = full 880 (paper protocol)")
    parser.add_argument("--smoke", action="store_true",
                        help="pipeline smoke test (1 epoch, 5 samples; overrides --shots/--epochs)")
    args = parser.parse_args()

    if args.smoke:
        print(f"[Config] SMOKE TEST mode (1 epoch, 5 samples)")
    else:
        shots_label = f"{args.shots}-shot" if args.shots > 0 else "full 880"
        print(f"[Config] SAMed LoRA rank-{args.rank}, {shots_label}, "
              f"{args.epochs} epochs (stop @ {args.stop_epoch})")

    train_samed(
        rank=args.rank,
        epochs=args.epochs,
        stop_epoch=args.stop_epoch,
        shots=args.shots,
        smoke=args.smoke,
    )
