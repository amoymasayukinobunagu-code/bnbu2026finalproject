# Prompt 消融 Handoff（GitHub）

完整逐步指南 → **[PARTNER_GUIDE.md](PARTNER_GUIDE.md)**

| 资源 | 链接 |
|---|---|
| 仓库 | https://github.com/amoymasayukinobunagu-code/bnbu2026finalproject |
| 权重 Release | https://github.com/amoymasayukinobunagu-code/bnbu2026finalproject/releases/tag/ckpt-v3-20261021 |
| 执行清单 | [PROMPT_ABLATION_RUNSHEET.md](experiments/prompt_ablation_v3/PROMPT_ABLATION_RUNSHEET.md) |
| 填表模板 | [SUMMARY.md](experiments/prompt_ablation_v3/SUMMARY.md) |

## 一句话流程

1. Clone 仓库，按 PARTNER_GUIDE 下 Release 权重 + 官方底座 + 数据  
2. 对必跑五模型跑 `--box-perturb` ∈ {0,10,50}（30 抄锚点）  
3. 填 SUMMARY，把精简 `summary.json` 放进 `docs/experiments/prompt_ablation_v3/summaries/`  
4. 开分支 PR（不要 push `.pth` / 数据集 / 大可视化）

## 必跑 / 选跑 / 跳过

见 PARTNER_GUIDE §1。勿把 PP-SAM 写成 Ours；跳过 Chen SAM-Adapter 与 MedSAM+MSA。