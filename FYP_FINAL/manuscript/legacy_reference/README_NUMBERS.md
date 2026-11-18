# 手稿数字说明（2026-11-17，Asia/Shanghai）

**不要从本目录的旧表抄数字写论文。**

- `tables/table1_sota.tex` 和 `tables/table2_ablation.tex` 是 **2026-06-10** 的过时表。
- 它们与正式 v3 结果不一致（含旧 1-shot / 不完整 shot 曲线），把评估扰动写成了 **random**，并把 PP-SAM 标成了 **ours**。
- `Partial Thesis/` 和 6 月章节正文同样停在更早的实验，**不要当终稿数字来用**。本章正文尚未按正式结果重写。

正式数字请从这里抄：

1. 本机正式评估：`E:\Project\FYP\results\**\summary.json`
2. 可读台账：`E:\Project\FYP\results\EXPERIMENT_SUMMARY.md`（2026-11-17 已按 json 更新）
3. 路径与限制索引：`E:\Project\FYP\docs\NUMBERS_INDEX.md`

必须记住的限制：

- 评估框是 GT 紧框 **固定外扩**（主表默认 30 像素），不是 random jitter。
- 表中 ± 是测试图像间标准差，不是多种子。
- FPS 口径不统一，写论文前需空闲 GPU 重测。
- **PP-SAM 是否写作 Ours 尚未决定。**
- A0/A1/B1 的 5-shot 只是烟测，不能进正式表。
- object-scale、弱边界系统分析、多种子都还没有做。
