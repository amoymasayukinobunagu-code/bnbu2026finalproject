# Supplementary work completed on 2026-11-18

The later thesis follow-up results, including corrected core HD95, MSA-B prompting,
speed, additional datasets and pilot studies, are indexed in
`analysis/REMAINING_EXPERIMENTS_RESULTS.md`.

The subsequent SAM ViT-B point and translated-box run is documented in
`analysis/PROMPT_VARIANTS_RESULTS.md` and `analysis/PROMPT_VARIANTS_PROTOCOL.md`.
The status below describes the earlier supplementary report.

This dated status supplements the 2026-11-17 writing snapshot. Historical README,
scope notes and formal result JSON are preserved; their statements that size and
boundary analyses were not done describe the earlier snapshot.

## New deliverables

- Word report: `manuscript/drafts/FYP_Supplementary_Analysis_20261118.docx`.
- Analysis protocol: `analysis/PROTOCOL.md`; executable analysis:
  `analysis/complete_supplement.py`.
- Results, feature values, thresholds, prediction audit, input SHA256 and subgroup
  tables: `analysis/outputs/`.
- Five figures in PNG/SVG: `manuscript/figures/supplementary/`.
- Reusable annotation-derived prompt generator and instructions:
  `tools/prompt_toolkit.py`, `tools/README.md`.

## What is completed

1. Relative-area subgroup analysis of all 120 Kvasir validation and 612 CVC images
   for eight core models, with training-derived tertile sensitivity.
2. Exploratory boundary colour-contrast and gradient analyses; crossed size/contrast
   tables; band-width sensitivity; independently recomputed boundary F1 and surface
   HD95. This is an operational image-feature analysis, not clinical weak-boundary
   annotation or proof of a causal effect.
3. Plotting and provenance consolidation of the existing fixed-box-expansion study:
   28 available raw-JSON dataset/model/expansion combinations. No new inference.
4. Independent prompt-generation utility, documentation, and 732 GT-derived prompt
   records. Boxes, foreground/background points and translation generation are
   supported; point/translation segmentation performance is not evaluated.

## Checks and limitations

All 5,856 saved predictions matched the formal per-image Dice values within1e-6.
All formal Dice/IoU summary means and group-weighted reconstructions reconciled.
Five toolkit tests, two surface-metric tests, and400 box comparisons with the
original helper passed. Five empty predictions receive BF1=0 and undefined surface
HD95 with explicit denominators. No inputs, checkpoints or original scripts changed.

The inspected legacy SAM/PP-SAM helpers name a maximum foreground Hausdorff distance
`hd95`. Do not call the old field validated HD95. New CSVs preserve it as
`legacy_distance` and separately report `surface_hd95_px`. Before editing final
thesis tables, audit this metric across model implementations and use explicit,
consistent definitions and recomputed values. Formal snapshot JSON is not overwritten.

All added subgroup work is post hoc and descriptive. Proxy thresholds are not
clinical categories; image-relative size is not millimetre diameter. Contrast and
area are correlated, and some trends depend on proxy definition. Clinical expert
assessment, multi-seed training, automatic prompting, and point/translation model
evaluation remain uncompleted. These results have not yet been integrated into the
full thesis or approved by the author/supervisor.

## Reproduction

```powershell
$py = 'C:/Users/Administrator/anaconda3/envs/sam_env/python.exe'
& $py analysis/complete_supplement.py
& $py -m unittest discover -s tools -p 'test_*.py' -v
& $py -m unittest discover -s analysis -p 'test_*.py' -v
```

Word layout was checked via native Microsoft Word PDF export and page PNGs. The
packaged LibreOffice renderer is unavailable on this machine; QA PDFs and page
images in `analysis/qa/` are internal review artifacts.
