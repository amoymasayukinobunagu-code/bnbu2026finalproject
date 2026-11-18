# Annotation derived prompt toolkit

Generate reproducible box and point prompts from binary ground-truth masks without
loading a model or modifying images. Dependencies: Python, NumPy, Pillow and SciPy.
The existing `sam_env` already supplies these packages. Run from `FYP_FINAL`:

```powershell
$py = 'C:/Users/Administrator/anaconda3/envs/sam_env/python.exe'
& $py tools/prompt_toolkit.py --masks 'E:/Project/FYP/datasets_link/Kvasir-SEG/masks' --split splits/Kvasir-SEG/Data-split/val.txt --out analysis/outputs/prompts_kvasir_val.json
& $py tools/prompt_toolkit.py --masks 'E:/Project/FYP/datasets_link/CVC-ClinicDB/PNG/Ground Truth' --out analysis/outputs/prompts_cvc.json
& $py -m unittest discover -s tools -p 'test_*.py' -v
```

## Interface and outputs

`box_from_mask(mask, expand=30, convention='legacy_v3')` returns original-image
`[xmin, ymin, xmax, ymax]`. `generate_prompts` adds an interior positive point,
a seeded random positive point, a seeded random background point where available,
and four translated boxes. Point coordinates use `[x,y]`, foreground label1 and
background label0. The interior point maximises distance to the GT background;
ties use NumPy row-major order. It is not necessarily the centroid.

CLI options include `--expand 0 10 30 50`, `--seed 24`, `--split`, and
`--convention half_open`. The JSON records image dimensions, mask path and SHA256,
coordinate convention, stable per-image seed and empty-mask status. Missing IDs
and duplicate stems fail visibly. Mask threshold is grayscale >127, consistent
with the inspected evaluation loaders. Empty masks return null boxes and no points;
full masks have no negative point. Files are read-only inputs.

## Coordinates and integration

`legacy_v3` deliberately reproduces the existing helper: xmax/ymax are the maximum
foreground pixel indices plus expansion, clipped to width/height. With zero
expansion, a one-pixel mask yields a degenerate box. Do not silently alter this
convention when reproducing formal scores. `half_open` instead adds1 to the maximum
pixel indices before expansion and is suitable for standard array slicing.
Choose the convention explicitly when integrating a new model.

Translated boxes move the30-pixel expanded box left/right by5% of image width or
up/down by5% of image height. Coordinates are clipped independently, so clipping
can shrink or degenerate a box. An adapter must reject degenerate boxes before
model inference. Positive/negative points are stored as separate candidates: use
the `kind` field to choose a single point or construct an explicitly defined
multi-point protocol. Model-specific resize and coordinate conversion remain the
responsibility of the inference adapter; no universal model adapter is claimed.

The later SAM-B and MSA-B benchmarks evaluated point prompts and translated
boxes. The evaluator caps the displacement before applying it, preserving box
dimensions. That differs from the demonstration translations emitted by this
utility. See [the evaluation protocol](../analysis/PROMPT_VARIANTS_PROTOCOL.md)
and [the results](../analysis/REMAINING_EXPERIMENTS_RESULTS.md). Both workflows
derive location cues from ground-truth annotations.

## Existing evaluation adapter example

```python
from tools.prompt_toolkit import box_from_mask
box = box_from_mask(gt_mask, expand=30, convention='legacy_v3')
if box is None or box[2] <= box[0] or box[3] <= box[1]:
    raise ValueError('Invalid prompt for inference')
# Pass box through the selected model's existing coordinate transform.
```

The utility is newly consolidated; original training/evaluation scripts and formal
results are unchanged. See `analysis/PROTOCOL.md` and `analysis/outputs/` for the
supplementary analysis definitions, manifests and raw tables.
