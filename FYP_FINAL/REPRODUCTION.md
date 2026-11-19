# Reproducing the thesis comparisons

Run the examples from the thesis workspace on the machine holding the implementation, data and pretrained weights. The documented environment is `sam_env` with Python 3.10, PyTorch 2.11 with CUDA 12.6, NumPy, OpenCV, SciPy, pandas and Pillow. The implementation workspace is `E:/Project/FYP`. Dataset files are under its `datasets_link` directory. Compare its Kvasir training and validation lists against the copies in `splits/Kvasir-SEG/Data-split/` before reproducing the primary evaluation.

## Primary comparison

The mapping below uses paths relative to the implementation workspace. The archived reference outputs are indexed in [NUMBERS_INDEX.md](numbers/NUMBERS_INDEX.md). Use the archived best checkpoint for each training budget. The formal SAMed full-data result was generated on the cloud host. Its local prompt-expansion rerun has its own recorded scores.

| Configuration | Evaluation entry point | Weights |
|---|---|---|
| SAM-B and SAM-L | `code/sam_baseline/infer.py` | `checkpoints/sam_vit_b.pth` or `checkpoints/sam_vit_l_0b3195.pth`; set `FYP_SAM_TYPE` accordingly. |
| MedSAM | `code/medsam_baseline/infer.py` | `checkpoints/medsam_vit_b.pth` |
| SAMed | `code/samed_baseline/eval_samed.py` | `results/samed_baseline_cloud/checkpoints/r4_full/samed_lora_r4_best.pth`; select the corresponding shot folder for a few-shot evaluation. |
| MedSAM+LoRA | `code/medsam_peft_baseline/eval_medsam_lora.py` | `checkpoints/medsam_peft/r4_full/medsam_lora_r4_best.pth` |
| MSA-B and MSA-L | `code/medical_sam_adapter/eval_polyp_v3.py` or `eval_polyp_vitl_v3.py` | `checkpoints/medical_sam_adapter/polyp/best_dice_checkpoint.pth` or the corresponding `polyp_vitl` checkpoint. Set `FYP_DATASETS=Kvasir-SEG-val,CVC-ClinicDB`. |
| PP-SAM | `code/pp_sam_baseline/eval_ppsam.py` | `checkpoints/pp_sam/pp_sam_vit_b_880shot_best.pth`; set `FYP_PPSAM_CKPT` to the selected file. |

For example, rerun SAM-B at the default 30-pixel expansion. Choose a fresh output directory.

```powershell
$fypRoot = 'E:/Project/FYP'
$fypPython = 'C:/Users/Administrator/anaconda3/envs/sam_env/python.exe'
$env:FYP_DATA_ROOT = "$fypRoot/datasets_link"
$env:FYP_SAM_TYPE = 'vit_b'
$env:FYP_SAM_CKPT = "$fypRoot/checkpoints/sam_vit_b.pth"
$env:FYP_OUT_DIR = "$PWD/analysis/outputs/reproduce_sam_b"
& $fypPython "$fypRoot/code/sam_baseline/infer.py" --box-perturb 30 --no-triptych
```

The LoRA evaluators also require `--ckpt`, `--rank 4` and `--out-tag`. Use the correct base weights through `FYP_SAM_CKPT` before calling either evaluator. The chapter tables use recalculated surface HD95 from archived masks. On a working copy, `analysis/complete_supplement.py`, `analysis/build_core_hd95_table.py` and `analysis/build_full_hd95_table.py` regenerate the analysis and exports in that order. These scripts write to the established analysis and table paths.

## Prompts, external datasets and supporting analyses

See the [prompt toolkit](tools/README.md) for box and point generation. The dedicated prompt evaluator preserves box dimensions when capping a translation. Use a fresh PowerShell session for the following commands so the model-specific environment variables from the primary run do not carry over.

```powershell
$fypPython = 'C:/Users/Administrator/anaconda3/envs/sam_env/python.exe'
$env:PYTHONPATH = (Resolve-Path tools).Path
$env:FYP_MSA_REPO = 'E:/Project/FYP/msa_repo'
& $fypPython analysis/evaluate_prompt_variants.py --model sam-b --out analysis/outputs/reproduce_prompt_sam_b
& $fypPython analysis/evaluate_prompt_variants.py --model msa-b --out analysis/outputs/reproduce_prompt_msa_b
& $fypPython analysis/extend_external_evaluation.py --out analysis/outputs/reproduce_external_trained
& $fypPython analysis/run_external_and_speed.py --mode speed --out analysis/outputs/reproduce_speed
& $fypPython analysis/pseudo_label_prompt_pilot.py --out analysis/outputs/reproduce_pseudo_label
& $fypPython analysis/pseudo_label_prompt_pilot.py --auto-sam --out analysis/outputs/reproduce_pseudo_label
```

The external extension reads the original reviewed input audit and baseline scores, then verifies the selected frames and boxes before running the added checkpoints. `analysis/sequence_prompt_statistics.py` reads the archived prompt runs and sequence mapping to reproduce Section 5.3.3. See the [experiment index](analysis/REMAINING_EXPERIMENTS_RESULTS.md) for the exact run manifests, results and checkpoint checksums.
