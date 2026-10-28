# bnbu2026finalproject

BNBU 2026 Final Year Project（Index 1）  
**Promptable Medical Image Segmentation Foundation Models for Endoscopic Polyp Segmentation**

基于 SAM / MedSAM 及轻量适配（LoRA、Adapter 等），在结肠镜息肉公开数据集上做系统评测与参数高效微调。评测协议为 **v3**：Kvasir-SEG val 120 + CVC-ClinicDB 612，GT box 外扩 ±N px。

导师：Dr. Donglong Chen

## 仓库结构

| 路径 | 说明 |
|---|---|
| [`PROJECT_BRIEF.md`](PROJECT_BRIEF.md) | 课题简介与完成标准 |
| [`code/`](code/) | 评测 / 训练脚本（统一 CLI `--box-perturb`，默认 30） |
| [`docs/`](docs/) | 实验说明与结果汇总模板 |
| GitHub Release [`ckpt-v3-20261021`](https://github.com/amoymasayukinobunagu-code/bnbu2026finalproject/releases/tag/ckpt-v3-20261021) | 已训好的 full-880 权重（不含官方底座） |

## 快速开始：Prompt 消融（box ±0 / 10 / 30 / 50）

同一批 checkpoint，只改评测时的 box 扰动，不重新训练。完整步骤见：

**[`docs/PARTNER_GUIDE.md`](docs/PARTNER_GUIDE.md)**（环境、数据布局、权重放置、各模型命令、填表）

相关文档：

- 执行清单：[`docs/experiments/prompt_ablation_v3/PROMPT_ABLATION_RUNSHEET.md`](docs/experiments/prompt_ablation_v3/PROMPT_ABLATION_RUNSHEET.md)
- 结果表模板：[`docs/experiments/prompt_ablation_v3/SUMMARY.md`](docs/experiments/prompt_ablation_v3/SUMMARY.md)

底座权重（SAM ViT-B/L、MedSAM）需从官方渠道自行下载；微调权重从上面的 Release 获取。

## 代码子目录

- `code/sam_baseline` — SAM 零样本
- `code/pp_sam_baseline` — MedSAM 零样本 / PP-SAM
- `code/samed_baseline` — SAMed LoRA
- `code/medsam_peft_baseline` — MedSAM+LoRA
- `code/medical_sam_adapter` — Medical SAM Adapter（息肉域）
