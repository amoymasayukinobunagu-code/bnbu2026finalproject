"""Draw the primary comparison workflow shown in Figure 4.1."""
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


def main():
    figure, axes = plt.subplots(figsize=(6.42, 3.06), layout="none")
    figure.subplots_adjust(left=0, right=1, top=1, bottom=0)
    axes.set(xlim=(0, 6.42), ylim=(0, 3.06))
    axes.set_aspect("equal")
    axes.axis("off")

    ink = "#1a2a3b"
    outline = "#526b83"
    blue = "#eaf2fa"
    green = "#e8f3ee"
    grey = "#f2f5f7"
    plt.rcParams["font.family"] = "DejaVu Sans"

    def box(x, y, width, height, title, detail, color):
        axes.add_patch(FancyBboxPatch(
            (x, y), width, height, boxstyle="round,pad=0.015,rounding_size=0.07",
            linewidth=0.8, edgecolor=outline, facecolor=color,
        ))
        axes.text(x + width / 2, y + height * 0.66, title, ha="center",
                  va="center", color=ink, fontsize=9.3, fontweight="semibold")
        axes.text(x + width / 2, y + height * 0.29, detail, ha="center",
                  va="center", color=ink, fontsize=8.0)

    def arrow(start, end):
        axes.add_patch(FancyArrowPatch(
            start, end, arrowstyle="-|>", mutation_scale=10,
            color=outline, linewidth=0.9, shrinkA=0, shrinkB=0,
        ))

    box(0.20, 2.39, 2.63, 0.54, "Released weights",
        "SAM ViT-B / ViT-L and MedSAM", blue)
    box(3.20, 2.39, 3.02, 0.54, "Polyp adaptation",
        "Fit on Kvasir-SEG training images", green)
    box(0.20, 1.64, 2.63, 0.54, "Evaluation inputs",
        "RGB frame + 30 px expanded box", grey)
    box(3.20, 1.64, 3.02, 0.54, "Checkpoint selection",
        "Kvasir-SEG validation Dice", green)
    box(1.11, 0.84, 4.20, 0.55, "Prompted inference",
        "Kvasir-SEG validation and CVC-ClinicDB", blue)
    box(1.11, 0.09, 4.20, 0.55, "Native-resolution evaluation",
        "Score masks and aggregate per-image metrics", grey)

    axes.plot([0.185, 0.095, 0.095], [2.66, 2.66, 1.115], color=outline, linewidth=0.9)
    arrow((0.095, 1.115), (1.09, 1.115))
    arrow((4.71, 2.38), (4.71, 2.20))
    arrow((1.97, 1.62), (1.97, 1.41))
    arrow((4.71, 1.62), (4.71, 1.41))
    arrow((3.21, 0.82), (3.21, 0.66))

    output = Path(__file__).resolve().parents[1] / "figures/implementation/experimental_pipeline.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=350, facecolor="white")
    plt.close(figure)


if __name__ == "__main__":
    main()
