
# Chen SAM-Adapter (tianrun-chen) — Kvasir polyp retrain

## Identity
- **NOT** Medical SAM Adapter (Wu / ImprintLab).
- Repo: E:/Project/FYP/repo_links/SAM-Adapter-PyTorch (junction → 材料/repos/SAM-Adapter-PyTorch)
- Paper: Chen et al., SAM-Adapter, ICCVW 2023. Official README tested Kvasir polyps.
- Retrain **only** on FYP 	rain.txt 880 / al.txt 120. No third-party Kvasir checkpoints as formal numbers.

## Protocol
| Split | N | Notes |
|-------|---|-------|
| Train | 880 | Kvasir-SEG train.txt |
| Val (train-time COD metrics) | 120 | Kvasir-SEG val.txt |
| Formal eval | 120 + CVC 612 | eval_polyp_v3.py |
| Prompt | **none (automatic)** | Method-native FFT/highpass visual adapter; GT box ±30 N/A for this arch |
| Metrics (formal) | Dice / IoU / P / R / HD95 / FPS | Same splits as v3 |

## Hyperparams
- Backbone init: E:/Project/FYP/checkpoints/sam_vit_b.pth (ViT-B)
- image/inp_size: **1024** (Chen default)
- batch_size: **1** (official forward hardcodes bs=1; looped)
- epochs: **40**, val every 5
- lr: AdamW 2e-4, CosineAnnealing → 1e-7
- Freeze: image_encoder except prompt_generator
- Trainable ≈ 4.12M / 93.8M total (smoke)

## Outputs
- Checkpoints: E:/Project/FYP/checkpoints/sam_adapter/polyp_kvasir_880/
  - model_epoch_best.pth, model_epoch_last.pth, history.json, config.yaml
- Results: E:/Project/FYP/results/sam_adapter_polyp_v3/
- Logs: E:/Project/FYP/results/sam_adapter_polyp_train/logs/
- Code: E:/Project/FYP/code/sam_adapter_baseline/

## Commands
`
# Train
E:\Project\FYP\code\sam_adapter_baseline\run_full_train.bat

# Eval v3 (after train)
E:\Project\FYP\code\sam_adapter_baseline\run_eval_v3.bat

# Or:
set FYP_SAM_ADAPTER_REPO=E:/Project/FYP/repo_links/SAM-Adapter-PyTorch
set FYP_DATA_ROOT=E:/Project/FYP/datasets_link
python E:/Project/FYP/code/sam_adapter_baseline/eval_polyp_v3.py --ckpt E:/Project/FYP/checkpoints/sam_adapter/polyp_kvasir_880/model_epoch_best.pth --out-dir E:/Project/FYP/results/sam_adapter_polyp_v3
`

## ETA (from smoke on RTX 4070 Ti SUPER)
- Smoke: 3 train steps ≈ 2.2s ⇒ ~0.73 s/img ⇒ **~10–11 min/epoch** train
- Val 120 @ ~7 it/s ≈ 17s every 5 epochs
- **40 epochs ≈ 7–8 hours**

## Local patches (clone)
- models/mmseg/__init__.py: mmcv 2.x pin → warning
- models/mmseg/models/__init__.py: skip mmcv1 Registry import
- models/mmseg/models/builder.py: mmengine fallback
- Single-GPU trainer (no NCCL DDP): code/sam_adapter_baseline/train_single_gpu.py

## Launch status (2026-10-16 Asia/Shanghai)
- Started: **2026-10-16 09:52:31 CST**
- Python PID: **36468** (DETACHED)
- Log: E:/Project/FYP/results/sam_adapter_polyp_train/logs/train_full_20261016_095231.log
- Err: E:/Project/FYP/results/sam_adapter_polyp_train/logs/train_full_20261016_095231.err
- Meta: E:/Project/FYP/results/sam_adapter_polyp_train/logs/train_full_launch.json
- Save: E:/Project/FYP/checkpoints/sam_adapter/polyp_kvasir_880/
- Observed: ~3.3 it/s at 1024 bs1 ⇒ **~4.5 min/epoch** ⇒ **ETA ~3–3.5 h** for 40 epochs (~13:00 CST)
- Resume B after A: E:/Project/FYP/code/sam_adapter_baseline/resume_after_A.bat
