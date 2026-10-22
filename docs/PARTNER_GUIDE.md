# 搭档操作指南：Prompt 消融（box ±0/10/30/50）

> 仓库：https://github.com/amoymasayukinobunagu-code/bnbu2026finalproject  
> 协议：**v3**（Kvasir-SEG `val.txt` 120 + CVC-ClinicDB 612，GT box 外扩 ±N px）  
> 只评测，不训练。  
> 关联：[PROMPT_ABLATION_RUNSHEET](experiments/prompt_ablation_v3/PROMPT_ABLATION_RUNSHEET.md) · [SUMMARY 模板](experiments/prompt_ablation_v3/SUMMARY.md) · [HANDOFF](HANDOFF_PROMPT_ABLATION_GITHUB.md)

---

## 1. 目标

在同一批已训好的 checkpoint 上，对比 box 扰动 `0 / 10 / 30 / 50` 的 **mean Dice**（主）与 IoU / HD95（副）。

- 数据集：Kvasir-SEG val 120 + CVC-ClinicDB 612
- Prompt：GT mask 紧致框向外扩展 `N` 像素（`--box-perturb N`）
- `±30` 可与主表锚点对齐，**可跳过实测、直接抄表**（见 §6）

### 必跑模型

| 模型 | 脚本 | ±30 锚点（Kvasir / CVC Dice） |
|---|---|---|
| SAM ViT-L zeroshot | `code/sam_baseline/infer.py` | 0.8140 / 0.6588 |
| MedSAM zeroshot | `code/pp_sam_baseline/eval_medsam_zeroshot.py` | 0.7034 / 0.5172 |
| SAMed LoRA full-880 | `code/samed_baseline/eval_samed.py` | **0.8955 / 0.8608**（本机 `r4_full_best` 实测） |
| MedSAM+LoRA r4 | `code/medsam_peft_baseline/eval_medsam_lora.py` | 0.9049 / 0.8476 |
| MSA polyp ViT-B | `code/medical_sam_adapter/eval_polyp_v3.py` | 0.8524 / 0.7685 |

### 选跑

| 模型 | 脚本 | ±30 锚点 | 备注 |
|---|---|---|---|
| MSA polyp ViT-L | `code/medical_sam_adapter/eval_polyp_vitl_v3.py` | 0.8555 / 0.8234 | 显存更大，后跑 |
| PP-SAM full-880 | `code/pp_sam_baseline/eval_ppsam.py` | 0.9226 / 0.8576 | **full-FT 对照**，表注勿写成 Ours |

### 跳过

Chen SAM-Adapter、MedSAM+MSA（已定负结果，不进消融主表）。

---

## 2. Clone 仓库

```powershell
git clone https://github.com/amoymasayukinobunagu-code/bnbu2026finalproject.git
cd bnbu2026finalproject
```

建议本机工作根目录与君涵习惯一致（可选）：

```text
E:\Project\FYP\          # 或你自己的目录，下文用 $ROOT 表示
  code\                  # 本仓库的 code/ 拷到这里，或直接在 clone 根目录跑
  checkpoints\           # 权重（见 §3）
  材料\datasets\         # 或 data\ / datasets_link\（见 §4）
  results\               # 评测输出
  msa_repo\              # Medical-SAM-Adapter 上游 clone（MSA 需要）
```

若目录不是 `E:\Project\FYP`，请用环境变量覆盖（见各命令的 `$env:...`）。

---

## 3. 权重：Release + 官方底座

### 3.1 本仓库 Release（微调 / Adapter 权重）

Release：https://github.com/amoymasayukinobunagu-code/bnbu2026finalproject/releases/tag/ckpt-v3-20261021

| Release 文件名 | 建议本机路径 |
|---|---|
| `ppsam_vitb_full880_best.pth` | `$ROOT\checkpoints\pp_sam\pp_sam_vit_b_880shot_best.pth` |
| `medsam_lora_r4_full880_best.pth` | `$ROOT\checkpoints\medsam_peft\r4_full\medsam_lora_r4_best.pth` |
| `samed_lora_r4_full880_best.pth` | `$ROOT\checkpoints\samed\r4_full\samed_lora_r4_best.pth` |
| `msa_polyp_vitb_880_best.pth` | `$ROOT\checkpoints\medical_sam_adapter\polyp\best_dice_checkpoint.pth` |
| `msa_polyp_vitl_880_best.pth` | `$ROOT\checkpoints\medical_sam_adapter\polyp_vitl\best_dice_checkpoint.pth` |

下载示例（需要 [GitHub CLI](https://cli.github.com/)）：

```powershell
gh release download ckpt-v3-20261021 -R amoymasayukinobunagu-code/bnbu2026finalproject -D $env:TEMP\ckpt-v3
# 再按上表手动移动/重命名到 checkpoints\...
```

### 3.2 官方底座（需自行下载，不在 Release 内）

| 底座 | 常见文件名 | 建议路径 | 来源 |
|---|---|---|---|
| SAM ViT-B | `sam_vit_b_01ec64.pth` → 可改名为 `sam_vit_b.pth` | `$ROOT\checkpoints\sam_vit_b.pth` | [segment-anything](https://github.com/facebookresearch/segment-anything#model-checkpoints) |
| SAM ViT-L | `sam_vit_l_0b3195.pth` | `$ROOT\checkpoints\sam_vit_l_0b3195.pth` | 同上 |
| MedSAM ViT-B | `medsam_vit_b.pth` | `$ROOT\checkpoints\medsam_vit_b.pth` | [MedSAM](https://github.com/bowang-lab/MedSAM) 官方 ckpt |

脚本默认读取（可用环境变量覆盖）：

- `FYP_SAM_CKPT`：SAM / SAMed / MSA 底座
- `FYP_MEDSAM_CKPT`：MedSAM zeroshot / MedSAM+LoRA 底座
- `FYP_PPSAM_CKPT`：PP-SAM 微调权重
- `FYP_MSA_CKPT`：MSA polyp 权重

---

## 4. 数据布局（不在仓库内）

数据需自行准备，或向**君涵**要一份。默认根目录：

```text
$env:FYP_DATA_ROOT = "E:\Project\FYP\材料\datasets"
```

不同脚本的默认略有差异（均可用 `FYP_DATA_ROOT` 覆盖）：

| 脚本族 | 默认 DATA_ROOT |
|---|---|
| `pp_sam_baseline/*`、`sam_baseline/infer.py`、MSA `infer` 辅助 | `E:/Project/FYP/材料/datasets` |
| `samed_baseline` | `<bundle>/data`（即仓库上两级的 `data/`） |
| `medsam_peft_baseline` | `E:/Project/FYP/datasets_link`（junction → 材料/datasets） |

**统一建议**：设好 `FYP_DATA_ROOT`，并保证下面结构存在：

```text
$FYP_DATA_ROOT\
  Kvasir-SEG\
    images\          # *.jpg / *.png
    masks\
    Data-split\
      val.txt        # 120 行 stem（无后缀）
      train.txt      # 880（训练用；本消融不训，可有可无）
  CVC-ClinicDB\
    PNG\
      Original\      # 612 张
      Ground Truth\  # 同名 mask
```

数据集名（脚本 `--dataset`）：

- `Kvasir-SEG-val`
- `CVC-ClinicDB`

---

## 5. 环境

- Python 3.10+（本机常用 conda env 名：`sam_env`）
- PyTorch + CUDA（与本机驱动匹配）
- 最小依赖：

```text
torch
torchvision
segment-anything          # pip install git+https://github.com/facebookresearch/segment-anything.git
opencv-python
numpy
Pillow
scipy
```

MSA 额外需要上游仓库：

```powershell
git clone https://github.com/KidsWithTokens/Medical-SAM-Adapter.git $ROOT\msa_repo
$env:FYP_MSA_REPO = "$ROOT\msa_repo"
```

（MSA 的 `models.sam.build_sam` 从该 repo 的 `sys.path` 导入。）

---

## 6. ±30 锚点（可直接抄进 SUMMARY）

| 模型 | Kvasir Dice | CVC Dice | 来源 |
|---|---|---|---|
| MedSAM zs | 0.7034 | 0.5172 | 主表 |
| MedSAM+LoRA r4 | 0.9049 | 0.8476 | 主表 |
| MSA-B | 0.8524 | 0.7685 | 主表 |
| MSA-L | 0.8555 | 0.8234 | 主表 |
| PP-SAM | 0.9226 | 0.8576 | 主表 |
| SAM ViT-L | 0.8140 | 0.6588 | 主表 |
| **SAMed LoRA full-880** | **0.8955** | **0.8608** | 本机 `results/samed_baseline_cloud/.../r4_full_best/summary.json` |

---

## 7. 命令模板（Windows PowerShell）

约定：

```powershell
$ROOT = "E:\Project\FYP"   # 按你的实际路径改
$CODE = "$ROOT\code"       # 或 clone 后的 bnbu2026finalproject\code
$OUT  = "$ROOT\results\prompt_ablation_v3"
$env:FYP_DATA_ROOT = "$ROOT\材料\datasets"
# 若 Python 不在 PATH：
# $PY = "C:\Users\Administrator\anaconda3\envs\sam_env\python.exe"
$PY = "python"
```

**输出目录**：每个偏移写到 `results/prompt_ablation_v3/<model>/box_<n>/`。  
多数脚本用 `FYP_OUT_DIR` 或 `--out-tag` 控制输出；**请为每个 `$n` 设不同输出路径**，避免互相覆盖。

循环骨架（对每个模型套用）：

```powershell
foreach ($n in 0, 10, 50) {   # 30 可跳过、抄 §6
  Write-Host "=== box_perturb=$n ==="
  # ... 下面各模型命令里把 $n 传给 --box-perturb，并设对应 OUT ...
}
```

### 7.1 SAM ViT-L zeroshot（必跑）

```powershell
foreach ($n in 0, 10, 50) {
  $env:FYP_SAM_CKPT = "$ROOT\checkpoints\sam_vit_l_0b3195.pth"
  $env:FYP_SAM_TYPE = "vit_l"
  $env:FYP_OUT_DIR  = "$OUT\sam_vitl\box_$n"
  & $PY "$CODE\sam_baseline\infer.py" --box-perturb $n --no-triptych
}
```

### 7.2 MedSAM zeroshot（必跑）

```powershell
foreach ($n in 0, 10, 50) {
  $env:FYP_MEDSAM_CKPT = "$ROOT\checkpoints\medsam_vit_b.pth"
  $env:FYP_OUT_DIR     = "$OUT\medsam\box_$n"
  & $PY "$CODE\pp_sam_baseline\eval_medsam_zeroshot.py" --box-perturb $n --no-triptych
}
```

### 7.3 SAMed LoRA full-880（必跑）

```powershell
foreach ($n in 0, 10, 50) {
  $env:FYP_SAM_CKPT = "$ROOT\checkpoints\sam_vit_b.pth"
  $env:FYP_OUT_DIR  = "$OUT"   # eval 用 --out-tag 建子目录；这里把 OUT_ROOT 指到 prompt_ablation 下
  # 注意：eval_samed 的 OUT_ROOT 默认是 results/samed_baseline；用 FYP_OUT_DIR 覆盖为父目录后：
  $env:FYP_OUT_DIR  = "$OUT\samed_lora_full880"
  & $PY "$CODE\samed_baseline\eval_samed.py" `
    --ckpt "$ROOT\checkpoints\samed\r4_full\samed_lora_r4_best.pth" `
    --rank 4 --out-tag "box_$n" --box-perturb $n --no-triptych
}
```

> 输出落在 `$OUT\samed_lora_full880\box_<n>\summary.json`。

### 7.4 MedSAM+LoRA r4（必跑）

```powershell
foreach ($n in 0, 10, 50) {
  $env:FYP_MEDSAM_CKPT = "$ROOT\checkpoints\medsam_vit_b.pth"
  $env:FYP_OUT_DIR     = "$OUT\medsam_lora_r4"
  & $PY "$CODE\medsam_peft_baseline\eval_medsam_lora.py" `
    --ckpt "$ROOT\checkpoints\medsam_peft\r4_full\medsam_lora_r4_best.pth" `
    --rank 4 --out-tag "box_$n" --box-perturb $n --no-triptych
}
```

### 7.5 MSA polyp ViT-B（必跑）

```powershell
$env:FYP_MSA_REPO = "$ROOT\msa_repo"
$env:FYP_SAM_CKPT = "$ROOT\checkpoints\sam_vit_b.pth"
$env:FYP_MSA_CKPT = "$ROOT\checkpoints\medical_sam_adapter\polyp\best_dice_checkpoint.pth"
$env:FYP_DATASETS = "Kvasir-SEG-val,CVC-ClinicDB"
foreach ($n in 0, 10, 50) {
  $env:FYP_OUT_DIR = "$OUT\msa_vitb\box_$n"
  & $PY "$CODE\medical_sam_adapter\eval_polyp_v3.py" --box-perturb $n
}
```

### 7.6 MSA polyp ViT-L（选跑，后跑）

```powershell
$env:FYP_MSA_REPO = "$ROOT\msa_repo"
$env:FYP_SAM_CKPT = "$ROOT\checkpoints\sam_vit_l_0b3195.pth"
$env:FYP_MSA_CKPT = "$ROOT\checkpoints\medical_sam_adapter\polyp_vitl\best_dice_checkpoint.pth"
$env:FYP_MSA_ENCODER = "vit_l"
$env:FYP_DATASETS = "Kvasir-SEG-val,CVC-ClinicDB"
foreach ($n in 0, 10, 50) {
  $env:FYP_OUT_DIR = "$OUT\msa_vitl\box_$n"
  & $PY "$CODE\medical_sam_adapter\eval_polyp_vitl_v3.py" --box-perturb $n
}
```

### 7.7 PP-SAM full-880（选跑，full-FT 对照）

```powershell
foreach ($n in 0, 10, 50) {
  $env:FYP_SAM_CKPT   = "$ROOT\checkpoints\sam_vit_b.pth"
  $env:FYP_PPSAM_CKPT = "$ROOT\checkpoints\pp_sam\pp_sam_vit_b_880shot_best.pth"
  $env:FYP_OUT_DIR    = "$OUT\ppsam\box_$n"
  & $PY "$CODE\pp_sam_baseline\eval_ppsam.py" --box-perturb $n --no-triptych
}
```

### 快速冒烟（可选）

任意命令加 `--limit 5`（MSA 用 `$env:FYP_LIMIT = "5"`）确认路径/权重无误后再全量跑。

---

## 8. 建议执行顺序（省 GPU）

1. **MedSAM+LoRA r4**：先 `0` 与 `50`
2. MedSAM zeroshot → SAMed → MSA-B → SAM ViT-L
3. 各模型补 `10`
4. `30` 抄 §6 锚点
5. 选跑：MSA-L、PP-SAM（ViT-L / MSA-L 放最后）

同一模型四个偏移串行；不同模型按显存排队。

---

## 9. 填表与回传

1. 填 [SUMMARY.md](experiments/prompt_ablation_v3/SUMMARY.md)（Kvasir / CVC 两张表，列 = 0/10/30/50）
2. 把各 `summary.json`（只要均值，不要整图 pred）拷到：

```text
docs/experiments/prompt_ablation_v3/summaries/
  medsam_lora_r4_box0.json
  ...
```

3. 开分支推送或开 PR（**不要** push 权重 / 大数据 / 逐图可视化）：

```powershell
cd <clone>
git checkout -b feat/prompt-ablation-v3-results
git add docs/experiments/prompt_ablation_v3/
git status   # 确认无 .pth
git commit -m "Add prompt ablation summaries (box 0/10/30/50)"
git push -u origin feat/prompt-ablation-v3-results
```

---

## 10. 完成标准

- [ ] 两数据集 × 四偏移 mean Dice（必跑五模型；30 可抄表）
- [ ] 可画「Dice vs box offset」折线（至少 5 个必跑模型）
- [ ] SUMMARY + 精简 summary.json 已 PR / push
- [ ] 无权重、无原始数据进入 git

---

## 11. 仍需向君涵确认 / 索取的东西

若本机没有：

1. **Kvasir-SEG + CVC-ClinicDB** 按 §4 布局的数据（或网盘链接）
2. Release 若下载失败：让君涵确认 GitHub 协作权限（当前推送账号示例：`Dazzle992`）
3. 官方 **SAM ViT-B/L**、**MedSAM** 底座若未缓存：按 §3.2 自行下；路径不一致时设 `FYP_SAM_CKPT` / `FYP_MEDSAM_CKPT`
4. **Medical-SAM-Adapter** 上游 clone（MSA 评测依赖）

有问题先对照本指南与 [RUNSHEET](experiments/prompt_ablation_v3/PROMPT_ABLATION_RUNSHEET.md)；仍卡再找君涵。