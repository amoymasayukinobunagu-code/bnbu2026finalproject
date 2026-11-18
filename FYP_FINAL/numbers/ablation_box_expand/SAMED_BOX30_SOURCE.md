# SAMed Kvasir box 30：本机消融 vs 正式云端评估

这是同一类权重下的两套数字，**不能混进一张表还不加说明**。

| 来源 | 路径（本目录） | Kvasir-120 Dice / IoU | CVC-612 Dice / IoU |
|---|---|---|---|
| 正式主表（云端评估，2026-06-25） | `numbers/official_core/samed_baseline_cloud/samed_baseline/r4_full_best/summary.json` | **0.8955 / 0.8280** | 0.8608 / 0.7814 |
| 本机框消融 box 30（2026-11-17） | `numbers/ablation_box_expand/samed_baseline/prompt_ablation_box_expand/box30/summary.json` | **0.8951 / 0.8273** | 0.8608 / 0.7814（与正式几乎逐位一致） |

Kvasir Dice 相差约 0.0003。正式主表与 shot 曲线请用云端 `r4_full_best`；消融表可报本机 0.8951，并注明与正式 0.8955 的差异。不要在无说明的情况下把两个数当成同一个。
