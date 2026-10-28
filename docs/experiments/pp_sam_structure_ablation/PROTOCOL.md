# PP-SAM 结构消融协议（Structure Ablation Protocol）

> 版本：v1（2026-10-28）  
> 原则：**公平优先**；本阶段只做 A1 边界损失，**暂不实现 B1 CNN 旁路**。  
> 工作区：`E:\Project\FYP`；对照 Git：`E:\Project\fyp_github`（可选同步）。

---

## 1. 目标

在**不拆 PP-SAM 冻结 mask decoder** 的前提下，探索弱边界 / 结构相关改进：

- **A1**：在现有 `structure_loss` 上增加显式边界 BCE（`structure_edge`），强化边界对齐。
- **B1**（下一阶段）：CNN 旁路等结构改动——**本协议阶段禁止开工实现**。
- 辅消融：在 MedSAM+LoRA 上挂同构的边界辅助项，验证边界项是否跨底座有效；主结论仍以 PP-SAM 为准。

不追求刷榜；追求与 A0 全量锚点**可公平对比**的消融证据。

---

## 2. 公平条款（固定，不得私自改）

| 条款 | 固定值 | 说明 |
|------|--------|------|
| SEED | 42 | PP-SAM 主实验；采样 / 初始化可复现 |
| 训练池 | `train.txt` **880** | 官方 Data-split；禁止从全 1000 采样 |
| 验证集 | `val.txt` **120** | held-out；训练期永不进入 shot 池 |
| 优化器 | AdamW，lr=`1e-4`，wd=`1e-4` | 与 A0 / paper 一致 |
| Epoch | **100**（烟测可用 `--epochs 50`） | 烟测结果不得当作正式表 |
| Batch | 1 | — |
| 输入边长 | 1024（ResizeLongestSide） | — |
| 训练 box 扰动 | 每边 U(0, 50) px | `PERTURB_L=0, PERTURB_H=50` |
| 选优 / 评测扰动 | `INFER_PERTURB=30` | **禁止改默认** |
| Decoder | `freeze_decoder=True` | **禁止解冻** mask decoder |
| 指标 | Dice / IoU / Precision / Recall / HD95 | 主报 Dice + HD95 |
| 5-shot 样本 | 共用固定 ids 文件 | 见 §4；禁止每次重 sample |

MedSAM+LoRA 辅消融保留其原协议默认（SEED=1234、CE+Dice、LoRA 等），仅增加可选边界项；与 PP-SAM 的 `structure_loss` **不完全相同**，但边界项构造同构。见 §3 辅行与 §6 注明。

---

## 3. 消融矩阵

| ID | 设定 | 状态 | 说明 |
|----|------|------|------|
| **A0** | `structure_loss`（默认），freeze_decoder，full-880 | **已有锚点，勿重训** | 见 §7 |
| **A1** | `structure_edge_loss` = structure + `edge_weight * edge_bce` | **本阶段实现** | CLI：`--loss structure_edge --edge-weight 0.5` |
| **B1** | CNN 旁路等结构改动 | **下一阶段** | 本协议禁止实现 |
| **AB** | A1 + B1 | 可选后续 | 须先各自成立 |
| **辅：MedSAM+LoRA A1** | 默认 CE+Dice；可选加 edge 辅助 | **本阶段实现开关** | 默认行为不变；边界项同构、主体 loss 不同 |

推荐汇报顺序：A0（锚点）→ A1 5-shot 烟测 → A1 full-880（若烟测正向）→ 辅 MedSAM A1 → 再开 B1。

---

## 4. 烟测 vs full-880；5-shot 共用 ids

### 4.1 规则

| 模式 | shots | epochs | 用途 | 可否写入正式表 |
|------|-------|--------|------|----------------|
| 烟测 | 5 | 50（可更短自检） | 验证 loss / 脚本 / 收敛方向 | **否**；仅过程证据 |
| 正式 | 880（或约定 shot） | 100 | 与 A0 公平对比 | **是** |

- **5-shot 烟测结果不得与 A0 full-880 数字直接横比。**
- 若需公平对照烟测：同一 5 ids、`--loss structure`、50 epoch → `a0_structure_5shot`，仅作 A1 相对对照。

### 4.2 固定 5-shot ids

- 路径（优先）：`E:\Project\FYP\checkpoints\pp_sam\shots5_ids_seed42.json`
- 行为：`--shots 5` 时若文件存在则**按文件加载同一列表**；不存在则 `SEED=42` sample 并写入。
- 文件内容示例：`{"seed": 42, "n": 5, "ids": ["xxx.jpg", ...]}`（ids 为 train 池图像文件名）。
- **所有 A0/A1 5-shot 对照必须共用该文件。**

---

## 5. 禁止项

1. **解冻** `mask_decoder`（`freeze_decoder` 必须为 True）。
2. 修改 `INFER_PERTURB` 默认 **30**，或训练扰动范围偏离 0–50。
3. 用 val 调参刷分（含反复改 `edge_weight` 盯 val）。
4. 每次 5-shot 重新 `random.sample`（破坏公平）。
5. 删除或覆盖现有 **full-880** checkpoint。
6. 本阶段实现 **B1** CNN 旁路。
7. force push 到远程。
8. 将烟测数字当作正式 full-880 结论。

---

## 6. Loss 定义（实现契约）

### 6.1 PP-SAM（主）

- **A0 / `structure`**（默认，保持不变）：

  `weit = 1 + 5 * |avg_pool(mask, 31) - mask|`  
  `L = mean( weighted_BCE + weighted_IoU )`

- **A1 / `structure_edge`**：

  - 边界图（纯 torch）：`edge = max_pool(mask, k) - mask`（膨胀减原 mask，取非负）。
  - `edge_bce`：预测 logits 对 GT soft edge 的 `binary_cross_entropy_with_logits`。
  - `L = structure_loss(pred, mask) + edge_weight * edge_bce`，默认 `edge_weight=0.5`。

- CLI：`--loss {structure,structure_edge}`（默认 `structure`）；`--edge-weight`（默认 0.5）；`--save-dir` 可覆盖输出目录。
- 每个 run 的 `save_dir` 写入 `config.json`：至少含 `loss, edge_weight, shots, epochs, seed, freeze_decoder`。

### 6.2 MedSAM+LoRA（辅）

- 默认仍为 SAMed 式 **CE + Dice**（`calc_loss`），**默认行为不变**。
- 可选：`--loss structure_edge` 时增加同构 edge 辅助项：`L = calc_loss + edge_weight * edge_bce`。
- **注明**：主体 loss 与 PP-SAM `structure_loss` 不完全相同；仅边界项同构。公平对比主表仍以 PP-SAM A0/A1 为准。

---

## 7. A0 锚点（勿重训）

来源：`results/pp_sam_baseline/eval_full880/summary.json`

| 数据集 | Dice | HD95 |
|--------|------|------|
| Kvasir-SEG val (n=120) | **0.9226** | **66.25** |
| CVC-ClinicDB (n=612) | **0.8576** | **27.72** |

对应 ckpt：`checkpoints/pp_sam/pp_sam_vit_b_880shot_best.pth`（勿删）。

---

## 8. 推荐命令

```text
# A1 5-shot 烟测（50 epoch）
python E:\Project\FYP\code\pp_sam_baseline\train_ppsam.py --shots 5 --epochs 50 --loss structure_edge --edge-weight 0.5 --save-dir E:\Project\FYP\checkpoints\pp_sam\a1_edge_5shot

# 对照烟测 A0 5-shot（同一 ids）
python E:\Project\FYP\code\pp_sam_baseline\train_ppsam.py --shots 5 --epochs 50 --loss structure --save-dir E:\Project\FYP\checkpoints\pp_sam\a0_structure_5shot

# MedSAM+LoRA 辅：A1 开关（默认仍 CE+Dice；显式开 edge）
python E:\Project\FYP\code\medsam_peft_baseline\train_medsam_lora.py --shots 5 --loss structure_edge --edge-weight 0.5
```

日志建议目录：`E:\Project\FYP\results\pp_sam_structure_ablation\logs\`。

---

## 9. 下一阶段（非本协议范围）

- 等 A1 烟测收敛方向明确后：决定是否上 A1 full-880。
- **B1** CNN 旁路：单独开协议小节；仍须遵守本节公平条款与冻结 decoder。
