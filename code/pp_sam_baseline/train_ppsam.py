"""
PP-SAM 训练脚本 — 适配主人 E:/Project/FYP/ 数据布局.

使用: python E:/Project/FYP/code/pp_sam_baseline/train_ppsam.py --shots 1

默认 1-shot. 改 shots 即可 (1/5/10/full).

跟原 paper 行为一致:
- 冻 mask decoder, 训 image_encoder + prompt_encoder
- box perturbation: 0-50 px per side, variable
- 100 epochs, AdamW, lr=1e-4, batch=1
- ResizeLongestSide to 1024
- structure_loss (weighted BCE + IoU)
- 每 1 epoch 在 Kvasir val.txt 120 张 (held-out) 上算 val dice, 选 best

数据 split 协议 (2026-06-16 修复):
- 训练样本: 从 Kvasir-SEG Data-split/train.txt 880 张里 random sample N 张
  (旧协议从全部 1000 张 sample, 有 val leak 风险; 新协议严格 train/val 隔离)
- Epoch-end 验证: Kvasir-SEG Data-split/val.txt 120 张 (held-out, 训练期间永远不碰)
- 旧 best.pth (0.8432) 在 1000 张上算的 -> 新 best.pth 在 120 张 held-out 上算 (会更严)
- 最终测试: best.pth 加载后跑 val.txt 120 + CVC 612 拿 5 指标

只改数据 layout (主人 Kvasir 在 E:/Project/FYP/材料/datasets/Kvasir-SEG/),
其他全照 paper.
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
import torch.nn.functional as F

# Add PP-SAM repo for SamPredictor / sam_model_registry
sys.path.insert(0, 'E:/Project/FYP/材料/repos/PP-SAM')

from segment_anything import sam_model_registry, SamPredictor
from segment_anything.utils.transforms import ResizeLongestSide


# --------------------------------------------------------------------------- #
# Config (env-var driven, mirrors sam_baseline/ for symmetry)
# --------------------------------------------------------------------------- #
DATA_ROOT = Path(os.environ.get("FYP_DATA_ROOT", "E:/Project/FYP/材料/datasets"))
SAM_CKPT = Path(os.environ.get("FYP_SAM_CKPT", "E:/Project/FYP/checkpoints/sam_vit_b.pth"))
CKPT_DIR = Path(os.environ.get("FYP_CKPT_DIR", "E:/Project/FYP/checkpoints/pp_sam"))
OUT_DIR = Path(os.environ.get("FYP_OUT_DIR", "E:/Project/FYP/results/pp_sam_baseline"))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42

# Paper config (hardcoded, match paper exactly)
N_EPOCHS = 100
LR = 1e-4
BATCH = 1
PERTURB_L = 0
PERTURB_H = 50
INFER_PERTURB = 30          # paper's best-checkpoint selection uses 30-px inference perturb
WEIGHT_DECAY = 1e-4
DECAY_RATE = 0.1
DECAY_EPOCH = 300
CLIP = 0.5
IMG_SIZE = 1024

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# --------------------------------------------------------------------------- #
# Loss (paper's structure_loss)
# --------------------------------------------------------------------------- #
def structure_loss(pred, mask):
    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))
    pred_sig = torch.sigmoid(pred)
    inter = ((pred_sig * mask) * weit).sum(dim=(2, 3))
    union = ((pred_sig + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)
    return (wbce + wiou).mean()


# --------------------------------------------------------------------------- #
# Utilities
# --------------------------------------------------------------------------- #
def clip_gradient(optimizer, grad_clip):
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)


def adjust_lr(optimizer, init_lr, epoch, decay_rate, decay_epoch):
    decay = decay_rate ** (epoch // decay_epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] *= decay


def _read_image_bytes(path):
    """Read a binary file, trying MSYS ``cat`` first and falling back to a direct
    Python read. The dataset ACL that originally motivated ``cat`` was re-measured
    as inactive on 2026-09-30, and ``cat`` is not on the PowerShell PATH
    (``C:\\Program Files\\Git\\usr\\bin``), so the fallback is the common path now.
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


def _imread(path, mode=cv2.IMREAD_COLOR):
    """Replacement for cv2.imread that bypasses the dataset-directory ACL."""
    data = _read_image_bytes(path)
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, mode)
    if img is None:
        raise IOError(f"cv2.imdecode failed for {path}")
    return img


def dice_score(pred, gt):
    p, g = pred.astype(bool), gt.astype(bool)
    inter = np.logical_and(p, g).sum()
    denom = p.sum() + g.sum()
    if denom == 0:
        return 1.0
    return float(2.0 * inter / denom)


# --------------------------------------------------------------------------- #
# Data: load Kvasir (or any dataset with images/ + masks/ layout)
# --------------------------------------------------------------------------- #
def load_kvasir_paths():
    """All 1000 Kvasir-SEG images (legacy / debugging only)."""
    img_dir = DATA_ROOT / "Kvasir-SEG" / "images"
    msk_dir = DATA_ROOT / "Kvasir-SEG" / "masks"
    if not img_dir.exists() or not msk_dir.exists():
        raise FileNotFoundError(f"Kvasir-SEG not found: {img_dir}")
    images = sorted([p for p in img_dir.iterdir() if p.suffix.lower() in {".jpg", ".png"}])
    return [(img, msk_dir / img.name) for img in images if (msk_dir / img.name).exists()]


def load_kvasir_train_paths():
    """880 Kvasir-SEG images from official train.txt (strict train/val isolation).

    Used as the SAMPLE POOL for N-shot fine-tuning. Val.txt is never
    reachable from this set, eliminating train/val leak.

    NOTE: We read train.txt via ``subprocess.run(['cat', ...])`` instead of
    Python's built-in ``open()`` because Windows ACL on this directory
    blocks Win32 read syscalls for the current user, but MSYS's POSIX
    layer (which ``cat`` uses) can read it. See FYP_STRUCTURE.md for the
    ACL context.
    """
    img_dir = DATA_ROOT / "Kvasir-SEG" / "images"
    msk_dir = DATA_ROOT / "Kvasir-SEG" / "masks"
    split_file = DATA_ROOT / "Kvasir-SEG" / "Data-split" / "train.txt"
    if not split_file.exists():
        raise FileNotFoundError(f"Kvasir train.txt not found: {split_file}")
    pairs = _read_split_file(split_file, img_dir, msk_dir)
    if not pairs:
        raise RuntimeError(f"No train pairs loaded from {split_file}")
    return pairs


def load_kvasir_val_pairs():
    """120 Kvasir-SEG images from official val.txt (held-out evaluation set).

    Used by ``test_inference`` so the epoch-end val dice reflects a clean
    held-out score. train.txt 880 images never see this set.
    """
    img_dir = DATA_ROOT / "Kvasir-SEG" / "images"
    msk_dir = DATA_ROOT / "Kvasir-SEG" / "masks"
    split_file = DATA_ROOT / "Kvasir-SEG" / "Data-split" / "val.txt"
    if not split_file.exists():
        raise FileNotFoundError(f"Kvasir val.txt not found: {split_file}")
    pairs = _read_split_file(split_file, img_dir, msk_dir)
    if not pairs:
        raise RuntimeError(f"No val pairs loaded from {split_file}")
    return pairs


def _read_split_file(split_file: Path, img_dir: Path, msk_dir: Path):
    """Read the stem list from a Data-split/*.txt file (``cat`` first, direct read
    as fallback -- see ``_read_image_bytes`` for why the fallback matters).
    Returns list of (image_path, mask_path) pairs that exist on disk.
    """
    import subprocess
    text = None
    try:
        r = subprocess.run(
            ["cat", str(split_file)],
            capture_output=True, text=True,
        )
        if r.returncode == 0 and r.stdout:
            text = r.stdout
    except (FileNotFoundError, OSError):
        pass
    if text is None:
        text = Path(split_file).read_text(encoding="utf-8-sig")
    pairs = []
    for line in text.splitlines():
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


def load_clinicdb_paths():
    img_dir = DATA_ROOT / "CVC-ClinicDB" / "PNG" / "Original"
    msk_dir = DATA_ROOT / "CVC-ClinicDB" / "PNG" / "Ground Truth"
    if not img_dir.exists():
        raise FileNotFoundError(f"CVC-ClinicDB not found: {img_dir}")
    images = sorted([p for p in img_dir.iterdir() if p.suffix.lower() in {".jpg", ".png"}])
    return [(img, msk_dir / img.name) for img in images if (msk_dir / img.name).exists()]


def get_data_paths(dataset, split="all"):
    """dataset: 'kvasir' | 'cvc'. split: 'all' | 'train' | 'val' (kvasir only)."""
    if dataset.lower() == "kvasir":
        if split == "train":
            return load_kvasir_train_paths()
        elif split == "val":
            return load_kvasir_val_pairs()
        elif split == "all":
            return load_kvasir_paths()
        else:
            raise ValueError(f"Unknown Kvasir split: {split}")
    elif dataset.lower() == "cvc":
        if split != "all":
            raise ValueError(f"CVC-ClinicDB has no official split; got {split}")
        return load_clinicdb_paths()
    else:
        raise ValueError(f"Unknown dataset: {dataset}")


# --------------------------------------------------------------------------- #
# Test / eval (inference, with inference-time box perturbation)
# --------------------------------------------------------------------------- #
def test_inference(model, dataset, infer_perturb=INFER_PERTURB, split="val"):
    """Held-out evaluation. For Kvasir-SEG, defaults to official val.txt (120 imgs);
    train.txt 880 is NEVER reachable from here, so the dice reported each
    epoch is a clean held-out score. For CVC (no official split) uses all 612.

    Returns (mean_dice, n, extras_dict) where extras_dict has iou/precision/recall/hd95.
    """
    pairs = get_data_paths(dataset, split=split)
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
        # GT-derived box, then add inference-time perturbation
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

        # Dice
        p_bool, g_bool = pred.astype(bool), mask.astype(bool)
        inter = np.logical_and(p_bool, g_bool).sum()
        denom = p_bool.sum() + g_bool.sum()
        d = float(2.0 * inter / denom) if denom else 1.0
        dices.append(d)

        # IoU
        union = np.logical_or(p_bool, g_bool).sum()
        j = float(inter / union) if union else 1.0
        ious.append(j)

        # Precision / Recall
        tp = inter
        fp = np.logical_and(p_bool, ~g_bool).sum()
        fn = np.logical_and(~p_bool, g_bool).sum()
        pr = float(tp / (tp + fp)) if (tp + fp) else 1.0
        rc = float(tp / (tp + fn)) if (tp + fn) else 1.0
        precs.append(pr)
        recs.append(rc)

        # HD95
        if p_bool.sum() and g_bool.sum():
            p_pts = np.argwhere(p_bool)
            g_pts = np.argwhere(g_bool)
            fwd = directed_hausdorff(p_pts, g_pts)[0]
            bwd = directed_hausdorff(g_pts, p_pts)[0]
            hd95s.append(float(max(fwd, bwd)))
        # else: skip empty pair for hd95 (will be NaN-mean'd below)

    n = len(dices)
    extras = {
        "iou": float(np.mean(ious)) if ious else 0.0,
        "precision": float(np.mean(precs)) if precs else 0.0,
        "recall": float(np.mean(recs)) if recs else 0.0,
        "hd95": float(np.nanmean(hd95s)) if hd95s else float("nan"),
    }
    return float(np.mean(dices)) if dices else 0.0, n, extras


# --------------------------------------------------------------------------- #
# Train
# --------------------------------------------------------------------------- #
def train_ppsam(n_shots=1, freeze_decoder=True, freeze_image_encoder=False, save_dir=None, resume=False):
    save_dir = Path(save_dir) if save_dir else CKPT_DIR
    save_dir.mkdir(parents=True, exist_ok=True)

    # 1) Build model
    print(f"[Build] Loading base SAM from {SAM_CKPT}...")
    model = sam_model_registry["vit_b"](checkpoint=str(SAM_CKPT))
    if freeze_image_encoder:
        for p in model.image_encoder.parameters():
            p.requires_grad = False
    if freeze_decoder:
        for p in model.mask_decoder.parameters():
            p.requires_grad = False
    model.to(DEVICE)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"[Build] SAM loaded. Trainable: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")

    # 2) Sample n_shots training images from the OFFICIAL TRAIN split (880 imgs).
    # train.txt/val.txt are strictly disjoint; this guarantees val.txt 120
    # never appears in the 1-shot fine-tuning set (was the leak under the
    # old "sample from all 1000" protocol).
    train_pairs = get_data_paths("kvasir", split="train")
    print(f"[Data] Kvasir-SEG train split has {len(train_pairs)} images, sampling {n_shots}...")
    random.seed(SEED)  # for reproducibility
    train_sample = random.sample(train_pairs, n_shots)

    # 3) Preprocess: read image, mask, derive perturbed box
    transform = ResizeLongestSide(IMG_SIZE)
    transformed_data = {}
    bbox_coords = {}
    masks_dict = {}
    for img_path, msk_path in train_sample:
        image = _imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = _imread(msk_path, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
        H, W = mask.shape
        ys, xs = np.where(mask > 0)
        if len(xs) == 0:
            continue
        # Variable perturbation 0-50 px per side
        x_min = max(0, xs.min() - np.random.randint(PERTURB_L, PERTURB_H + 1))
        x_max = min(W, xs.max() + np.random.randint(PERTURB_L, PERTURB_H + 1))
        y_min = max(0, ys.min() - np.random.randint(PERTURB_L, PERTURB_H + 1))
        y_max = min(H, ys.max() + np.random.randint(PERTURB_L, PERTURB_H + 1))
        bbox_coords[img_path.name] = np.array([x_min, y_min, x_max, y_max])

        input_image_u8 = transform.apply_image(image)   # uint8 HWC, kept on CPU
        # FYP 2026-10-07: keep the preloaded image on CPU as uint8. The earlier
        # version stored the *preprocessed* float32 tensor on the GPU (~12 MB per
        # image). That is fine for 1/5/10-shot (<=120 MB) but 880 x 12 MB = 10.3 GB
        # of VRAM under --shots 880, which OOMs a 16 GB card. Moving the
        # .to(DEVICE) + model.preprocess() into the training loop keeps VRAM flat;
        # dtype semantics are unchanged (preprocess already received a uint8 tensor
        # before and relied on type promotion against pixel_mean/std).
        transformed_data[img_path.name] = {
            "image_u8": input_image_u8,
            "input_size": tuple(input_image_u8.shape[:2]),
            "original_image_size": image.shape[:2],
        }
        masks_dict[img_path.name] = mask
    print(f"[Data] Loaded {len(transformed_data)} samples into memory")

    # 4) Optimizer
    params = list(model.image_encoder.parameters()) + list(model.prompt_encoder.parameters()) + list(model.mask_decoder.parameters())
    optimizer = torch.optim.AdamW(params, lr=LR, weight_decay=WEIGHT_DECAY)
    print(f"[Opt] AdamW lr={LR} wd={WEIGHT_DECAY}, training {sum(p.numel() for p in params if p.requires_grad):,} params")

    # 5) Train loop
    best_dice = 0.0
    best_epoch = 0
    best_path = save_dir / f"pp_sam_vit_b_{n_shots}shot_best.pth"
    last_path = save_dir / f"pp_sam_vit_b_{n_shots}shot_last.pth"
    hist_path = save_dir / f"pp_sam_vit_b_{n_shots}shot_history.json"
    history = []
    start_epoch = 1
    total_start = time.time()

    # FYP 2026-10-07: --resume for long runs. full-880 is 10-30 h; without this a
    # crash at epoch 90 meant restarting from zero. Restores weights from last.pth
    # plus the epoch history, so elapsed_s stays monotone and best-epoch selection
    # continues (without restoring best_dice, the first epoch after a resume would
    # overwrite a better best.pth). NOTE: AdamW moments are not checkpointed, so
    # expect a small optimizer transient over the first few epochs after resuming.
    if resume:
        if last_path.exists() and hist_path.exists():
            model.load_state_dict(torch.load(last_path, map_location=DEVICE))
            with open(hist_path, encoding="utf-8") as f:
                history = json.load(f)
            start_epoch = int(history[-1]["epoch"]) + 1
            best_dice = max((h["kvasir_val_dice"] for h in history), default=0.0)
            best_epoch = max(history, key=lambda h: h["kvasir_val_dice"])["epoch"]
            total_start = time.time() - float(history[-1]["elapsed_s"])
            print(f"[Resume] {last_path.name}: continue at epoch {start_epoch}, "
                  f"best_dice={best_dice:.4f} (epoch {best_epoch})")
        else:
            print(f"[Resume] no last.pth/history.json in {save_dir} -> starting fresh")

    for epoch in range(start_epoch, N_EPOCHS + 1):
        model.train()
        adjust_lr(optimizer, LR, epoch, DECAY_RATE, DECAY_EPOCH)
        epoch_losses = []
        for k in transformed_data:
            d = transformed_data[k]
            arr = torch.as_tensor(d["image_u8"], device=DEVICE)   # uint8 HWC -> GPU (1 img)
            transformed_image = arr.permute(2, 0, 1).contiguous()[None, :, :, :]
            input_image = model.preprocess(transformed_image)
            input_size = d["input_size"]
            original_image_size = d["original_image_size"]
            optimizer.zero_grad()
            image_embedding = model.image_encoder(input_image)
            prompt_box = bbox_coords[k]
            box = transform.apply_boxes(prompt_box, original_image_size)
            box_torch = torch.as_tensor(box, dtype=torch.float, device=DEVICE)[None, :, None, :]
            sparse_embeddings, dense_embeddings = model.prompt_encoder(
                points=None, boxes=box_torch, masks=None,
            )
            low_res_masks, _ = model.mask_decoder(
                image_embeddings=image_embedding,
                image_pe=model.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=False,
            )
            gt_mask = torch.from_numpy(masks_dict[k]).to(DEVICE)
            gt_binary = (gt_mask > 0.0).float().unsqueeze(0).unsqueeze(0)
            upscaled = model.postprocess_masks(low_res_masks, input_size, original_image_size)
            loss = structure_loss(upscaled, gt_binary)
            loss.backward()
            clip_gradient(optimizer, CLIP)
            optimizer.step()
            epoch_losses.append(loss.item())

        avg_loss = float(np.mean(epoch_losses))
        # 6) Validate every epoch on the OFFICIAL HELD-OUT val split (120 imgs).
        # test_inference defaults to split="val" — train.txt 880 is never touched here.
        dice_kv, n_kv, kv_extras = test_inference(model, "kvasir", infer_perturb=INFER_PERTURB, split="val")
        elapsed = time.time() - total_start
        history.append({
            "epoch": epoch,
            "loss": avg_loss,
            "kvasir_val_dice": dice_kv,
            "kvasir_val_iou": kv_extras["iou"],
            "kvasir_val_precision": kv_extras["precision"],
            "kvasir_val_recall": kv_extras["recall"],
            "kvasir_val_hd95": kv_extras["hd95"],
            "n_val": n_kv,
            "elapsed_s": elapsed,
        })
        log = (f"[{datetime.now().strftime('%H:%M:%S')}] Epoch {epoch:3d}/{N_EPOCHS}  "
               f"loss={avg_loss:.4f}  KvasirVal_Dice={dice_kv:.4f}  "
               f"IoU={kv_extras['iou']:.4f}  P={kv_extras['precision']:.4f}  "
               f"R={kv_extras['recall']:.4f}  HD95={kv_extras['hd95']:.2f}  "
               f"({elapsed/60:.1f}min)")
        print(log, flush=True)
        if dice_kv > best_dice:
            best_dice = dice_kv
            best_epoch = epoch
            torch.save(model.state_dict(), best_path)
            print(f"  >>> New best Dice {best_dice:.4f} (epoch {epoch}), saved to {best_path}")
        torch.save(model.state_dict(), last_path)

    print(f"\n[DONE] {N_EPOCHS} epochs in {(time.time()-total_start)/60:.1f} min")
    print(f"[DONE] Best Kvasir 30-px dice: {best_dice:.4f}")
    print(f"[DONE] Best ckpt: {best_path}")
    print(f"[DONE] Last ckpt: {last_path}")
    # Save history
    with open(save_dir / f"pp_sam_vit_b_{n_shots}shot_history.json", "w") as f:
        json.dump(history, f, indent=2)
    return best_path, best_dice


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--shots", type=int, default=1, help="k-shot fine-tuning (1/5/10/...); 880 = full Kvasir train split")
    parser.add_argument("--epochs", type=int, default=N_EPOCHS, help="override epochs (default=100)")
    parser.add_argument("--resume", action="store_true",
                        help="continue from <shots>shot_last.pth + history.json if present")
    args = parser.parse_args()
    if args.epochs != N_EPOCHS:
        N_EPOCHS = args.epochs
    print(f"[Config] PP-SAM fine-tuning, {args.shots}-shot, {N_EPOCHS} epochs, resume={args.resume}")
    train_ppsam(n_shots=args.shots, resume=args.resume)
