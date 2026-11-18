# SAM ViT-B prompt performance evaluation

This supplementary experiment evaluates zero-shot SAM ViT-B on the 120 Kvasir-SEG
validation images and 612 CVC-ClinicDB images from the formal v3 run. It uses the
existing checkpoint and one cached image embedding per image. Every prompt uses
the predictor's single-mask setting. The 30-pixel ground-truth box condition is
matched against the official per-image Dice and IoU as a control.

## Prompt conditions

- **Reference box.** Use the historical v3 coordinates for the ground-truth
  tight box with 30 pixels of outward expansion.
- **Translated box.** Translate the reference box in each of the four cardinal
  directions by 5% of the corresponding image dimension, rounded to pixels.
  When the image edge restricts the translation, cap the displacement while
  preserving box width and height. Record both requested and actual translation.
  Report all frames and the subset receiving the full requested displacement
  for each direction. These subsets can differ by direction.
- **Interior point.** Use the foreground pixel with maximum distance from
  background. Ties follow row-major array order.
- **Interior point with negative point.** Pair the same positive point with
  one reproducibly sampled background pixel using base seed 24.
- **Random positive point.** Sample one uniform foreground pixel per image for
  each base seed 24, 42 and 1234. Mix the seed with a stable hash of the image ID.
  Average the three trials per image before aggregating. The seed and spread
  tables show variation from click sampling with the same checkpoint.

## Metrics and reproduction

Report Dice, IoU and boundary F1, plus the difference from the rerun reference
box on the same image. Boundary F1 matches native-image boundary pixels within
0.5% of the image diagonal, rounded to pixels with a minimum of one pixel.
Empty predictions contribute zero. Report image-level standard deviations as
descriptive dispersion. The experiment uses ground-truth-derived prompts.
Direction-specific subsets and differences in the information supplied by
boxes and points limit the interpretation. The analysis concerns one checkpoint
and uses image-level summaries without patient-level inference.

```powershell
$promptPython = 'C:/Users/Administrator/anaconda3/envs/sam_env/python.exe'
& $promptPython -u analysis/evaluate_prompt_variants.py --out analysis/outputs/prompt_variants_repeat
```

Choose a fresh output directory. `run.json` records software, input manifest
and checkpoint hashes and baseline reconciliation. `predictions.jsonl`
contains the prompt coordinates and scores logged during inference.
`per_image.csv` includes the paired deltas. `summary.csv` and the two
`random_point_*.csv` tables contain aggregate results.
