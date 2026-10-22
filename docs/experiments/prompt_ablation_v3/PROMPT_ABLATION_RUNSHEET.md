# Prompt 消融执行清单（box 0 / 10 / 30 / 50）

> **搭档完整操作指南**：[docs/PARTNER_GUIDE.md](../../PARTNER_GUIDE.md)（clone、权重、数据、命令、填表一次看完）

> 依据 Notion 总览与 1016 阶段说明整理（2026-10-19）。协议仍是 v3：Kvasir `val.txt` 120 + CVC-ClinicDB 612，GT box **±N px**。不训新权重，只改已有权重的评测扰动。

## 目标

回应 Index 1 的 prompt quality RQ：同一批 checkpoint，对比 box 扰动 `0 / 10 / 30 / 50` 的 Dice（主）与 IoU / HD95（副）。`±30` 已是现有主表数，可作锚点，不必重跑时可直接抄表。

## 建议跑的模型（按叙事优先级）

| 优先级 | 模型 | 权重 / 结果锚点 | 角色 |
|---|---|---|---|
| 必跑 | SAM ViT-L | 零样本主表 | 零样本柱 |
| 必跑 | MedSAM | zeroshot；Kvasir 0.7034 / CVC 0.5172 @±30 | 医学预训练柱 |
| 必跑 | SAMed LoRA full-880 | 既有 full-880 ckpt | PEFT 柱 |
| 必跑 | MedSAM+LoRA r4 | `checkpoints/medsam_peft/r4_full/medsam_lora_r4_best.pth`；±30 = 0.9049 / 0.8476 | MedSAM PEFT |
| 必跑 | MSA polyp ViT-B | Adapter @880；±30 = 0.8524 / 0.7685 | Adapter 柱 |
| 选跑 | MSA polyp ViT-L | ±30 = 0.8555 / 0.8234 | Adapter 柱（L） |
| 选跑 | PP-SAM full-880 | ±30 = 0.9226 / 0.8576 | full-FT 对照（勿默认写成 Ours） |
| 可跳过 | Chen SAM-Adapter / MedSAM+MSA | 已定负结果，停扩 | 不必进消融主表 |

## 输出约定（本机）

根目录建议：`E:\Project\FYP\results\prompt_ablation_v3\`

每个模型一个子目录，例如：

- `sam_vitl/box_0` … `box_50`
- `medsam/box_0` …
- `samed_lora_full880/box_0` …
- `medsam_lora_r4/box_0` …
- `msa_vitb/box_0` …

每个子目录至少留下：逐图 metrics CSV（或既有 eval 输出）+ 一份 `summary.json`（mean Dice / IoU / HD95）。

汇总表：`results/prompt_ablation_v3/SUMMARY.md`（行=模型，列=`0/10/30/50` Dice；Kvasir 与 CVC 分两表或同表加后缀）。

## 执行顺序（省 GPU）

1. 先确认各 baseline 的 eval 脚本是否已支持 `--box_jitter` / `box_offset`（或等价参数）；没有就先改一处共用的 GT-box 生成，再批跑。
2. 每个模型：先跑 `0` 与 `50`（两端），再补 `10`；`30` 与主表一致则可跳过实测、直接引用。
3. 同一模型四个偏移串行即可；不同模型可按显存排队（ViT-L / MSA-L 放后）。
4. 跑完立刻写 `SUMMARY.md`，再同步 Notion 总览「Goals #3」勾选与一页 `1019阶段进度说明`（只写消融增量）。

## 完成标准

- Kvasir-120 与 CVC-612 在四个偏移上都有 mean Dice。
- 正文可画一张「Dice vs box offset」折线（至少 SAM-L、MedSAM、SAMed、MedSAM+LoRA、MSA-B）。
- 不扩 Adapter；PP-SAM 若跑了，表注写 full-FT 对照。

## 本机旁路文档

跑完后把 HQ-SAM / MSA-B·L / SAM3 / MedSAM+LoRA / Chen / MedSAM+MSA 一并并入 `results/EXPERIMENT_SUMMARY.md`（总览 Tasks 里仍未勾的台账项）。
