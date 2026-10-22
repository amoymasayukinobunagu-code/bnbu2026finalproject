# bnbu2026finalproject

BNBU 2026 Final Year Project — medical polyp segmentation with SAM-family models (v3 protocol).

## 快速开始（搭档跑 Prompt 消融）

请直接阅读：**[docs/PARTNER_GUIDE.md](docs/PARTNER_GUIDE.md)**

其中包含：必跑/选跑模型、Release 权重放置、数据布局、`--box-perturb` 命令模板、填表与 PR 说明。

- 执行清单：[docs/experiments/prompt_ablation_v3/PROMPT_ABLATION_RUNSHEET.md](docs/experiments/prompt_ablation_v3/PROMPT_ABLATION_RUNSHEET.md)
- 权重：https://github.com/amoymasayukinobunagu-code/bnbu2026finalproject/releases/tag/ckpt-v3-20261021
- 项目简介：[PROJECT_BRIEF.md](PROJECT_BRIEF.md)

## 仓库内容

- `code/` — 评测脚本（含统一 CLI `--box-perturb`，默认 30）
- `docs/` — 搭档指南与实验 runsheet / SUMMARY 模板