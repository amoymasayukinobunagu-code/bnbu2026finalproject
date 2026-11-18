"""Plot audited default-box predictions near the median paired Dice change."""
import csv
from pathlib import Path
from statistics import median

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

MANUSCRIPT = Path(__file__).resolve().parents[1]
JOINED = MANUSCRIPT.parent / "analysis/outputs/per_image_joined.csv"
RESULTS = MANUSCRIPT.parent.parent / "FYP/results"
PREDICTIONS = {"SAM-B": "sam_v3", "MSA-B": "msa_polyp_train/eval_v3"}


def load_mask(path):
    with Image.open(path) as source:
        return np.asarray(source.convert("L")) > 127


def overlay(axes, rgb, mask=None, color=None):
    if mask is not None:
        mask = mask.astype("uint8")
        contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        rgb = cv2.drawContours(rgb.copy(), contours, -1, color, 2, cv2.LINE_AA)
    axes.imshow(rgb)
    axes.axis("off")


def main():
    groups = {}
    with JOINED.open(encoding="utf-8", newline="") as source:
        for row in csv.DictReader(source):
            if (row["size_group"] == "Medium" and row["contrast_main_group"] == "Low"
                    and row["model"] in PREDICTIONS and row["prediction_match"] == "True"):
                groups.setdefault(row["dataset"], {}).setdefault(row["id"], {})[row["model"]] = row

    figure, axes = plt.subplots(2, 4, figsize=(8.2, 3.9), layout="constrained")
    for axis, title in zip(axes[0], ("Input frame", "Reference", "SAM-B", "MSA-B")):
        axis.set_title(title, fontsize=11)

    for panels, dataset in zip(axes, ("Kvasir-SEG-val", "CVC-ClinicDB")):
        candidates = {key: pair for key, pair in groups[dataset].items() if pair.keys() == PREDICTIONS.keys()}
        differences = {key: float(pair["MSA-B"]["dice"]) - float(pair["SAM-B"]["dice"])
                       for key, pair in candidates.items()}
        centre = median(differences.values())
        selected = min(differences, key=lambda key: (abs(differences[key] - centre), key))
        pair = candidates[selected]
        row = pair["SAM-B"]
        with Image.open(row["image"]) as source:
            rgb = np.asarray(source.convert("RGB"))
        reference = load_mask(row["mask"])
        overlay(panels[0], rgb)
        short_dataset = dataset.replace("-val", "")
        panels[0].text(0.5, 0.04, short_dataset, transform=panels[0].transAxes, ha="center",
                       color="white", fontsize=9, bbox={"facecolor": "#172735", "pad": 2.3})
        overlay(panels[1], rgb, reference, (74, 231, 227))
        scores = []
        for panel, (model, folder) in zip(panels[2:], PREDICTIONS.items()):
            prediction_dir = RESULTS / folder / dataset / "preds"
            prediction_path, = prediction_dir.glob(f"{selected}.*")
            prediction = load_mask(prediction_path)
            dice = 2 * np.count_nonzero(reference & prediction) / (reference.sum() + prediction.sum())
            assert abs(dice - float(pair[model]["dice"])) < 1e-6, prediction_path
            overlay(panel, rgb, prediction, (255, 202, 85))
            panel.text(0.5, 0.04, f"Dice {dice:.3f}", transform=panel.transAxes, ha="center",
                       color="white", fontsize=9, bbox={"facecolor": "#172735", "pad": 2.3})
            scores.append(f"{model} {dice:.3f}")
        print(dataset, selected, f"{len(candidates)} candidates", ", ".join(scores))

    output = MANUSCRIPT / "figures/results/default_box_examples.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=300, facecolor="white")
    plt.close(figure)


if __name__ == "__main__":
    main()
