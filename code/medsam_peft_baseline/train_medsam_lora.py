"""
MedSAM + LoRA (PEFT) training — Index-1 row: medical-pretrained base + lightweight PEFT.

Reuses the proven SAMed LoRA pipeline (cloud AMP version), but initializes the SAM
ViT-B backbone from official MedSAM medical-pretrained weights
(checkpoints/medsam_vit_b.pth) instead of vanilla sam_vit_b.pth.

Protocol: FYP v3
- Train: Kvasir-SEG train.txt 880
- Val:   Kvasir-SEG val.txt 120 (held-out); cross-domain CVC-ClinicDB 612 (post-train eval)
- Prompt: GT box +/-30 px
- Metrics: Dice / IoU / Precision / Recall / HD95

Architecture (SAMed / Zhang & Liu 2023):
- LoRA rank 4 on image encoder qkv (Q and V)
- Freeze image encoder backbone; train LoRA + prompt_encoder + mask_decoder
- Loss: CE + Dice (0.8 Dice weight)
- Optimizer: AdamW; default lr=5e-4 (matches successful SAMed full-880 cloud run),
  weight_decay=0.1, warmup 250 iters, poly decay
- Image size: 1024 (MedSAM / SAM native)
- AMP fp16 enabled (needed for 1024^2 on 16GB 4070 Ti SUPER)

Why not MSA Adapter path:
- MSA polyp runs used sam_vit_b/l at 512; swapping to MedSAM at 1024 needs more rewrite
  and resolution mismatch vs MedSAM native. SAMed already loads any ViT-B ckpt via
  sam_model_registry — cleanest MedSAM+LoRA.

Usage:
    python train_medsam_lora.py --smoke
    python train_medsam_lora.py --rank 4 --epochs 200 --stop_epoch 160 --lr 0.0005
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
from torch.cuda.amp import autocast, GradScaler

from segment_anything import sam_model_registry, SamPredictor
from segment_anything.utils.transforms import ResizeLongestSide

# Allow running script either as module or by path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from lora_sam import LoRA_Sam

# ── Config ────────────────────────────────────────────────────────────────── #
# Default paths assume cloud layout (bundle root has data/ + checkpoints/).
# Override with env vars when running elsewhere.
# Project-root defaults for local FARUAJU layout
PROJECT_ROOT = Path(r"E:/Project/FYP")
DATA_ROOT = Path(os.environ.get("FYP_DATA_ROOT", PROJECT_ROOT / "datasets_link"))  # junction -> 材料/datasets
# Official MedSAM medical-pretrained ViT-B (NOT third-party polyp MedSAM adapters)
SAM_CKPT = Path(os.environ.get(
    "FYP_SAM_CKPT",
    os.environ.get("FYP_MEDSAM_CKPT", PROJECT_ROOT / "checkpoints" / "medsam_vit_b.pth"),
))
CKPT_DIR = Path(os.environ.get("FYP_CKPT_DIR", PROJECT_ROOT / "checkpoints" / "medsam_peft"))
OUT_DIR = Path(os.environ.get("FYP_OUT_DIR", PROJECT_ROOT / "results" / "medsam_peft"))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 1234  # SAMed's default seed

# SAMed paper hyperparameters
LR_DEFAULT = 0.0005  # matches SAMed full-880 cloud run (lr0005 log)
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


def soft_edge_from_mask(mask, kernel_size=3):
    """GT soft edge via dilation - mask (pure torch). Same construction as PP-SAM A1."""
    pad = kernel_size // 2
    dilated = F.max_pool2d(mask, kernel_size=kernel_size, stride=1, padding=pad)
    return (dilated - mask).clamp(min=0.0)


def calc_loss_with_edge(pred_logits, gt_mask, edge_weight=0.5):
    """Aux ablation: SAMed CE+Dice + isomorphic edge BCE (not PP-SAM structure_loss)."""
    base, loss_ce, loss_dice = calc_loss(pred_logits, gt_mask)
    edge = soft_edge_from_mask(gt_mask)
    edge_bce = F.binary_cross_entropy_with_logits(pred_logits, edge)
    total = base + float(edge_weight) * edge_bce
    return total, loss_ce, loss_dice


# ── Utilities ─────────────────────────────────────────────────────────────── #
def clip_gradient(optimizer, grad_clip):
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)


def _imread(path, mode=cv2.IMREAD_COLOR):
    """Plain cv2 read — no NTFS ACL workaround needed on Linux."""
    img = cv2.imread(str(path), mode)
    if img is None:
        raise IOError(f"cv2.imread failed for {path}")
    return img


def _read_split_file(split_file, img_dir, msk_dir):
    pairs = []
    with open(split_file, "r") as f:
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
def train_medsam_lora(rank=4, epochs=200, stop_epoch=160, shots=-1, smoke=False, amp=True, lr=None, loss_name="ce_dice", edge_weight=0.5):
    """Train MedSAM + LoRA on Kvasir-SEG train.txt.

    Args:
        rank: LoRA rank (SAMed paper default 4)
        epochs: max training epochs
        stop_epoch: early-stop epoch (SAMed default 160)
        shots: K-shot sampling. -1 = full 880 (paper protocol).
               K > 0: randomly sample K imgs from train.txt for few-shot LoRA.
        smoke: pipeline smoke test — forces 1 epoch on 5 samples.
        amp: enable mixed precision (fp16 autocast + GradScaler).
             Default True — required to fit SAM ViT-B + LoRA + grad on 24GB GPU.
    """
    if smoke:
        epochs, stop_epoch, shots = 1, 1, 5
        print("[Smoke] epochs=1, samples=5 — pipeline validation only")

    global_lr = float(lr) if lr is not None else LR_DEFAULT

    # Tag output dirs by shots config so K-shot runs don't collide with full
    run_tag = f"r{rank}_smoke" if smoke else (
        f"r{rank}_full" if shots < 0 else f"r{rank}_{shots}shot"
    )
    ckpt_dir = CKPT_DIR / run_tag
    out_dir = OUT_DIR / run_tag
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    run_config = {
        "loss": loss_name,
        "edge_weight": float(edge_weight) if loss_name == "structure_edge" else None,
        "shots": shots,
        "epochs": epochs,
        "stop_epoch": stop_epoch,
        "seed": SEED,
        "rank": rank,
        "note": "MedSAM+LoRA base is CE+Dice; structure_edge adds isomorphic edge BCE only",
    }
    with open(ckpt_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(run_config, f, indent=2)
    print(f"[Config] wrote {ckpt_dir / 'config.json'}: {run_config}")
    print(f"[Out] ckpt_dir={ckpt_dir}  out_dir={out_dir}")
    print(f"[GPU] {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}  "
          f"AMP={amp}")

    # 1) Build model
    print(f"[Build] Loading MedSAM ViT-B from {SAM_CKPT}...")
    sam = sam_model_registry["vit_b"](checkpoint=str(SAM_CKPT))
    lora_model = LoRA_Sam(sam, r=rank).to(DEVICE)

    trainable = lora_model.trainable_param_count
    total = sum(p.numel() for p in lora_model.parameters())
    print(f"[Build] MedSAM+LoRA rank={rank}. Trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

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

        # SAMed-style training: tight GT box (no train-time perturbation,
        # unlike PP-SAM which jitters U(0,50)). Inference still uses ±30 box.
        x_min, x_max = xs.min(), xs.max()
        y_min, y_max = ys.min(), ys.max()
        bbox_coords[img_path.name] = np.array([x_min, y_min, x_max, y_max])

        input_image = transform.apply_image(image)
        # Preprocess one image on GPU (sam.pixel_mean is on CUDA), then park float16 on CPU.
        # Keeping all 880 float32 tensors on GPU OOMs a 16GB card.
        input_image_torch = torch.as_tensor(input_image, device=DEVICE)
        transformed_image = input_image_torch.permute(2, 0, 1).contiguous()[None, :, :, :]
        input_image = sam.preprocess(transformed_image)
        transformed_data[img_path.name] = {
            "image": input_image.detach().half().cpu().contiguous(),
            "input_size": tuple(transformed_image.shape[-2:]),
            "original_image_size": image.shape[:2],
        }
        del input_image_torch, transformed_image, input_image
        masks_dict[img_path.name] = mask

    print(f"[Data] Loaded {len(transformed_data)} samples into memory")

    # 4) Optimizer (SAMed: AdamW, lr=0.005, wd=0.1) + AMP scaler
    params = filter(lambda p: p.requires_grad, lora_model.parameters())
    optimizer = torch.optim.AdamW(params, lr=global_lr / WARMUP_PERIOD, betas=(0.9, 0.999), weight_decay=WEIGHT_DECAY)
    scaler = GradScaler(enabled=amp)
    print(f"[Opt] AdamW lr={global_lr} wd={WEIGHT_DECAY}, warmup={WARMUP_PERIOD} iters, AMP={amp}")

    # 5) Training loop
    val_pairs = load_kvasir_val_pairs()
    print(f"[Val] Kvasir val.txt: {len(val_pairs)} images (held-out)")

    best_dice = 0.0
    best_epoch = 0
    best_path = ckpt_dir / f"medsam_lora_r{rank}_best.pth"
    last_path = ckpt_dir / f"medsam_lora_r{rank}_last.pth"
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
            input_image = d["image"].to(device=DEVICE, dtype=torch.float32, non_blocking=True)
            input_size = d["input_size"]
            original_image_size = d["original_image_size"]

            optimizer.zero_grad(set_to_none=True)

            with autocast(enabled=amp, dtype=torch.float16):
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
                if loss_name == "structure_edge":
                    loss, loss_ce, loss_dice = calc_loss_with_edge(upscaled, gt_binary, edge_weight=edge_weight)
                else:
                    loss, loss_ce, loss_dice = calc_loss(upscaled, gt_binary)

            # AMP backward
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            clip_gradient(optimizer, CLIP)
            scaler.step(optimizer)
            scaler.update()

            # LR schedule: warmup then poly decay
            if iter_num < WARMUP_PERIOD:
                lr_ = global_lr * (iter_num + 1) / WARMUP_PERIOD
            else:
                shift = iter_num - WARMUP_PERIOD
                lr_ = global_lr * (1.0 - shift / max_iterations) ** 0.9
            for pg in optimizer.param_groups:
                pg['lr'] = lr_

            epoch_losses.append(loss.item())
            iter_num += 1
            if iter_num % 50 == 0:
                print(f"  [iter {iter_num}] epoch={epoch+1} last_loss={loss.item():.4f} "
                      f"vram={torch.cuda.memory_allocated()/1e9:.2f}GB", flush=True)

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

    with open(ckpt_dir / f"medsam_lora_r{rank}_history.json", "w") as f:
        json.dump(history, f, indent=2)
    return best_path, best_dice


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MedSAM + LoRA (PEFT) on Kvasir 880 — FYP v3")
    parser.add_argument("--rank", type=int, default=4, help="LoRA rank (default 4, SAMed paper)")
    parser.add_argument("--epochs", type=int, default=200, help="max epochs (SAMed default 200)")
    parser.add_argument("--stop_epoch", type=int, default=160, help="early stop epoch (SAMed default 160)")
    parser.add_argument("--shots", type=int, default=-1,
                        help="K-shot training (e.g. 1/5/10); -1 = full 880 (paper protocol)")
    parser.add_argument("--lr", type=float, default=LR_DEFAULT,
                        help=f"peak AdamW lr (default {LR_DEFAULT}, matches SAMed full-880 cloud)")
    parser.add_argument("--smoke", action="store_true",
                        help="pipeline smoke test (1 epoch, 5 samples; overrides --shots/--epochs)")
    parser.add_argument("--no_amp", action="store_true",
                        help="disable mixed precision (not recommended)")
    parser.add_argument("--loss", type=str, default="ce_dice",
                        choices=["ce_dice", "structure_edge"],
                        help="ce_dice=default SAMed CE+Dice; structure_edge adds edge BCE aux (A1)")
    parser.add_argument("--edge-weight", type=float, default=0.5, dest="edge_weight",
                        help="edge BCE weight when --loss structure_edge (default 0.5)")
    args = parser.parse_args()

    print(f"[Config] backbone=MedSAM ({SAM_CKPT})")
    print(f"[Config] ckpt_dir={CKPT_DIR}  out_dir={OUT_DIR}")
    if args.smoke:
        print("[Config] SMOKE TEST mode (1 epoch, 5 samples)")
    else:
        shots_label = f"{args.shots}-shot" if args.shots > 0 else "full 880"
        print(f"[Config] MedSAM+LoRA rank-{args.rank}, {shots_label}, "
              f"{args.epochs} epochs (stop @ {args.stop_epoch}), lr={args.lr}, "
              f"loss={args.loss}, edge_weight={args.edge_weight}")

    train_medsam_lora(
        rank=args.rank,
        epochs=args.epochs,
        stop_epoch=args.stop_epoch,
        shots=args.shots,
        smoke=args.smoke,
        amp=not args.no_amp,
        lr=args.lr,
        loss_name=args.loss,
        edge_weight=args.edge_weight,
    )
