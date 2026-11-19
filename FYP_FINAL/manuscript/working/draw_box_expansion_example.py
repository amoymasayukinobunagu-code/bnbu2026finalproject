"""Reproduce the paired PP-SAM example and CVC prompt-area summaries."""
import csv
import json
from pathlib import Path
from statistics import mean
import sys

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
from PIL import Image

from draw_results_examples import MANUSCRIPT, RESULTS, load_mask, overlay

ROOT = MANUSCRIPT.parent
sys.path.insert(0, str(ROOT / "tools"))
from prompt_toolkit import box_from_mask

DATASET = "CVC-ClinicDB"
ABLATION = Path("pp_sam_baseline/prompt_ablation_box_expand")


def box_area(box):
    left, top, right, bottom = box
    return (right - left) * (bottom - top)


def main():
    with (ROOT / "analysis/outputs/image_features.csv").open(encoding="utf-8", newline="") as source:
        frames = {row["id"]: row for row in csv.DictReader(source) if row["dataset"] == DATASET}

    ratios = {"Small": [], "Medium": [], "Large": []}
    for row in frames.values():
        reference = load_mask(row["mask"])
        ratio = box_area(box_from_mask(reference, 50)) / box_area(box_from_mask(reference, 0))
        ratios[row["size_group"]].append(ratio)
    for group, values in ratios.items():
        print(f"{group}: n={len(values)}, mean area ratio={mean(values):.6f}")

    scores = {}
    for expansion in (0, 50):
        run = ROOT / "numbers/ablation_box_expand" / ABLATION / f"box{expansion}"
        record = json.loads((run / "per_image.json").read_text(encoding="utf-8"))[DATASET]
        scores[expansion] = {Path(row["name"]).stem: row["dice"] for row in record["per_image"]}
        assert scores[expansion].keys() == frames.keys()
        print(f"{expansion} px mean Dice: {mean(scores[expansion].values()):.6f}")

    selected = min(frames, key=lambda key: (scores[50][key] - scores[0][key], key))
    row = frames[selected]
    with Image.open(row["image"]) as source:
        rgb = np.asarray(source.convert("RGB"))
    reference = load_mask(row["mask"])
    figure, axes = plt.subplots(1, 3, figsize=(8.2, 2.4), layout="constrained")
    overlay(axes[0], rgb, reference, (74, 231, 227))
    axes[0].set_title("Reference contour", fontsize=11)

    for axis, expansion in zip(axes[1:], (0, 50)):
        prediction_dir = RESULTS / ABLATION / f"box{expansion}" / DATASET / "preds"
        prediction_path, = prediction_dir.glob(f"{selected}.*")
        mask = load_mask(prediction_path)
        dice = 2 * np.count_nonzero(reference & mask) / (reference.sum() + mask.sum())
        assert abs(dice - scores[expansion][selected]) < 1e-6, prediction_path
        overlay(axis, rgb, mask, (255, 202, 85))
        left, top, right, bottom = box_from_mask(reference, expansion)
        axis.add_patch(Rectangle((left - 0.5, top - 0.5), right - left, bottom - top,
                                 fill=False, edgecolor="white", linewidth=1.3,
                                 linestyle="--", clip_on=False))
        axis.set_title(f"{expansion} px expansion | Dice {dice:.3f}", fontsize=11)
        print(f"{DATASET} {selected}: {expansion} px, Dice={dice:.6f}, box={left, top, right, bottom}")

    figure.savefig(MANUSCRIPT / "figures/results/box_expansion_example.png", dpi=300, facecolor="white")
    plt.close(figure)


if __name__ == "__main__":
    main()
