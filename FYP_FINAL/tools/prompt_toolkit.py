"""Generate annotation-derived prompts without loading a segmentation model.

Coordinates are original-image (x, y). No prompt is an automatic lesion detector.
The legacy_v3 box convention exactly reproduces the existing evaluation helper.
"""
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt


def box_from_mask(mask, expand=30, convention='legacy_v3'):
    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 2 or expand < 0:
        raise ValueError('A 2D mask and nonnegative expansion are required')
    if convention not in ('legacy_v3', 'half_open'):
        raise ValueError('Unknown coordinate convention')
    y, x = np.where(mask)
    if not len(x):
        return None
    h, w = mask.shape
    extra = int(convention == 'half_open')
    return [max(0, int(x.min()) - expand), max(0, int(y.min()) - expand),
            min(w, int(x.max()) + extra + expand), min(h, int(y.max()) + extra + expand)]


def shift_box(box, shape, dx, dy):
    if box is None:
        return None
    h, w = shape
    return [float(np.clip(box[0] + dx, 0, w)), float(np.clip(box[1] + dy, 0, h)),
            float(np.clip(box[2] + dx, 0, w)), float(np.clip(box[3] + dy, 0, h))]


def generate_prompts(mask, seed=24, expansions=(0, 10, 30, 50), convention='legacy_v3'):
    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 2:
        raise ValueError('Expected a 2D mask')
    h, w = mask.shape
    boxes = {str(e): box_from_mask(mask, int(e), convention) for e in expansions}
    if not mask.any():
        return dict(status='empty_mask', width=w, height=h, boxes=boxes, points=[], shifted_boxes={})
    # Padding ensures the distance transform handles foreground at image edges.
    dt = distance_transform_edt(np.pad(mask, 1))[1:-1, 1:-1]
    cy, cx = np.unravel_index(np.argmax(dt), dt.shape)
    rng = np.random.default_rng(seed)
    fg = np.argwhere(mask)
    py, px = fg[rng.integers(len(fg))]
    points = [dict(kind='interior_positive', xy=[int(cx), int(cy)], label=1),
              dict(kind='random_positive', xy=[int(px), int(py)], label=1)]
    bg = np.argwhere(~mask)
    if len(bg):
        ny, nx = bg[rng.integers(len(bg))]
        points.append(dict(kind='random_negative', xy=[int(nx), int(ny)], label=0))
    base = box_from_mask(mask, 30, convention)
    shifted = {name: shift_box(base, mask.shape, dx*w, dy*h) for name, dx, dy in
               [('left_5pct', -.05, 0), ('right_5pct', .05, 0),
                ('up_5pct', 0, -.05), ('down_5pct', 0, .05)]}
    return dict(status='ok', width=w, height=h, coordinate_convention=convention,
                boxes=boxes, points=points, shifted_boxes=shifted)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--masks', type=Path, required=True)
    ap.add_argument('--split', type=Path, help='Optional filenames or stems, one per line')
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--seed', type=int, default=24)
    ap.add_argument('--expand', type=int, nargs='+', default=[0, 10, 30, 50])
    ap.add_argument('--convention', choices=['legacy_v3', 'half_open'], default='legacy_v3')
    args = ap.parse_args()
    files = sorted(p for p in args.masks.iterdir() if p.suffix.lower() in ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'))
    by_stem = {}
    for p in files:
        if p.stem in by_stem:
            raise ValueError(f'Duplicate mask stem: {p.stem}')
        by_stem[p.stem] = p
    if args.split:
        stems = [Path(s.strip()).stem for s in args.split.read_text(encoding='utf-8-sig').splitlines() if s.strip()]
        if len(stems) != len(set(stems)):
            raise ValueError('Duplicate split IDs')
        files = [by_stem[s] for s in stems]  # Missing masks fail visibly.
    records=[]
    for p in files:
        mask = np.asarray(Image.open(p).convert('L')) > 127
        # Stable per-image seed is independent of folder order and Python hash randomisation.
        seed = (args.seed + int(hashlib.sha256(p.stem.encode()).hexdigest()[:8], 16)) % 2**32
        records.append(dict(id=p.stem, mask=str(p.resolve()), sha256=hashlib.sha256(p.read_bytes()).hexdigest(),
                            seed=seed, **generate_prompts(mask, seed, args.expand, args.convention)))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(dict(origin='ground_truth_annotation', evaluated_prompt_types=['fixed_box_expansion'],
        unevaluated_prompt_types=['points', 'box_translation'], threshold='grayscale > 127', records=records), indent=2), encoding='utf-8')
    print(f'Wrote {len(records)} records to {args.out}')


if __name__ == '__main__':
    main()
