> **FYP_FINAL 写作快照（2026-11-17，Asia/Shanghai）**  
> 本文件是从 `E:\\Project\\FYP` 复制到 `E:\\Project\\FYP_FINAL` 的写作材料，原项目未移动、未删除。  
> 请以本目录 `numbers/NUMBERS_INDEX.md` 为入口。文中若仍出现 `E:\\Project\\FYP\\results\\...`，那是**原路径**，对应副本见本目录相对路径（见 SOURCE_MANIFEST）。  
> 消融汇总副本。MedSAM CVC 与 SAMed 来源区别见 ablation_box_expand 下的说明文件。

# Prompt ablation: fixed outward box expand 0/10/30/50

- Time: 2026-11-17 17:28:30–17:57:05 Asia/Shanghai
- Machine: DESKTOP-FARUAJU, env sam_env, no training, formal result files not overwritten
- Protocol: v3 GT tight box + **fixed outward expand** in original-image pixels (`--box-perturb`); **not** random jitter
- Data: `FYP_DATA_ROOT=E:/Project/FYP/datasets_link`, Kvasir-SEG val.txt n=120, CVC-ClinicDB n=612
- HD95 was computed by the scripts but excluded from the primary table, same as the user CVC report
- Readable index: `docs/NUMBERS_INDEX.md`

## Results (Dice / IoU)

### MedSAM+LoRA r4 (`checkpoints/medsam_peft/r4_full/medsam_lora_r4_best.pth`)

| box | Kvasir-120 Dice / IoU | CVC-612 Dice / IoU |
|---|---|---|
| 0 | **0.9269 / 0.8712** | 0.8702 / 0.7980 (user, not re-run) |
| 10 | 0.9201 / 0.8614 | **0.8720 / 0.7959** (user, not re-run) |
| 30 | 0.9049 / 0.8375 | 0.8476 / 0.7596 (user; matches formal v3) |
| 50 | 0.8883 / 0.8120 | 0.8134 / 0.7151 (user, not re-run) |

This run: Kvasir only. Script: `code/medsam_peft_baseline/eval_medsam_lora.py`. Out: `results/medsam_peft/prompt_ablation_box_expand/kvasir_box{0,10,30,50}/`. Log: `results/medsam_peft/prompt_ablation_box_expand/logs/kvasir_all_20261117_172830.log`.

Kvasir box30 exact: dice=0.9048652980822726 iou=0.8375096292356836, matches formal `medsam_peft/r4_full_best` (2026-10-15).

### PP-SAM full880 (`checkpoints/pp_sam/pp_sam_vit_b_880shot_best.pth`)

| box | Kvasir-120 Dice / IoU | CVC-612 Dice / IoU |
|---|---|---|
| 0 | 0.9228 / 0.8637 | 0.8859 / 0.8113 |
| 10 | **0.9280 / 0.8720** | **0.8948 / 0.8221** |
| 30 | 0.9226 / 0.8630 | 0.8576 / 0.7699 |
| 50 | 0.9071 / 0.8394 | 0.8025 / 0.7016 |

Script: `code/pp_sam_baseline/eval_ppsam.py`. Backbone: `checkpoints/sam_vit_b.pth`. Out: `results/pp_sam_baseline/prompt_ablation_box_expand/box{0,10,30,50}/`. box30 exact match to formal `pp_sam_baseline_v3/eval_best_880shot`.

### SAMed r4 full (`results/samed_baseline_cloud/checkpoints/r4_full/samed_lora_r4_best.pth`)

| box | Kvasir-120 Dice / IoU | CVC-612 Dice / IoU |
|---|---|---|
| 0 | **0.9039 / 0.8419** | 0.8680 / 0.7975 |
| 10 | 0.9019 / 0.8385 | **0.8729 / 0.8012** |
| 30 | 0.8951 / 0.8273 | 0.8608 / 0.7814 |
| 50 | 0.8866 / 0.8147 | 0.8366 / 0.7506 |

Script: `code/samed_baseline/eval_samed.py`. Backbone: `checkpoints/sam_vit_b.pth`. Out: `results/samed_baseline/prompt_ablation_box_expand/box{0,10,30,50}/`.

box30 vs formal cloud eval (`samed_baseline_cloud/.../r4_full_best`, Kvasir 0.8955/0.8280, CVC 0.8608/0.7814):
- CVC nearly identical (dice 0.86083531 vs 0.86083550)
- Kvasir Dice **0.8951 vs formal 0.8955** (delta about -0.0003). Small local-vs-cloud difference; **formal numbers were not changed**.

### MSA-B (`checkpoints/medical_sam_adapter/polyp/best_dice_checkpoint.pth`)

| box | Kvasir-120 Dice / IoU | CVC-612 Dice / IoU |
|---|---|---|
| 0 | 0.8573 / 0.7869 | **0.7894 / 0.7082** |
| 10 | **0.8573 / 0.7871** | 0.7802 / 0.6998 |
| 30 | 0.8524 / 0.7826 | 0.7685 / 0.6896 |
| 50 | 0.8462 / 0.7770 | 0.7618 / 0.6840 |

Script: `code/medical_sam_adapter/eval_polyp_v3.py`, image_size=512, both splits. Out: `results/msa_polyp_train/prompt_ablation_box_expand/box{0,10,30,50}/`. box30 exact match to formal `msa_polyp_train/eval_v3` and `eval_v3_cvc`.

## Protocol notes

- All box30 reruns used the same existing eval scripts and official ckpts. Formal `summary.json` files still have their original timestamps and were not overwritten.
- MedSAM CVC 0/10/30/50 came from the user and were **not re-run** on this machine; CVC box30 already matched formal 0.8476/0.7596. There is **no local json** for MedSAM CVC box 0/10/50.
- **Not done:** zero-shot SAM-L and zero-shot MedSAM 0/10/30/50 ablations. No line plots yet.
- No A0/A1/B1 training. Those 5-shot smokes stay unofficial.
- ± / variation here is still test-image statistics, not multi-seed.

## Reading

- Default v3 ±30 is not the peak for these models. Tight or slightly loose boxes are better, especially on CVC.
- MedSAM Kvasir falls monotonically as the box loosens; CVC box10 is only +0.0018 Dice over box0. Do not write a single global optimum.
- PP-SAM peaks at box10 on both splits, and is the most CVC-sensitive among the completed comparison models when the box goes to ±50.
- SAMed Kvasir prefers box0; CVC prefers box10.
- MSA-B changes little across 0/10/30/50 relative to the others.
