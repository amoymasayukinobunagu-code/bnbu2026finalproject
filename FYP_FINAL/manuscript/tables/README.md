# 终稿用表（由正式 json 导出，2026-11-17）

2026-11-18 更新：正文主表使用 `table_main_boxprompt_surface_hd95.csv`。它覆盖
原主表所有 28 个结果行，并以保存的预测统一重算表面 HD95（原图像素）和边界 F1。
HD95 的有效样本数按行记录。旧文件作为存档。速度比较使用单独的
`analysis/outputs/core_speed_20261118/summary.csv`，覆盖三个模型。

这些表是从本目录已复制的正式 `summary.json` **机械导出**的，不是 6 月旧表。导出后已按 json 校验。

| 文件 | 内容 |
|---|---|
| `table_main_boxprompt.csv/.tex` | 章程主线 box-prompt 对比（含 PP-SAM，已标明不是 Ours） |
| `table_shot_curves.csv/.tex` | SAMed 与 PP-SAM 的 1/5/10/full 曲线 |
| `table_appendix_extra.csv/.tex` | 额外已验证 box-prompt 基线（附录） |
| `table_appendix_prompt_free.csv/.tex` | 无 box 方法（Chen / ASPS / OP-SAM），禁止与主表混排 |
| `table_ablation_box_expand.csv/.tex` | 固定外扩消融（不含用户提供的 MedSAM CVC 0/10/50） |

CSV 同时保留 json 原始浮点（`dice_m` 等）和四位展示值。± 为测试图像间标准差。FPS 仅供参考。

**不要**把 `../legacy_reference/stale_tables/` 拷进本目录。
