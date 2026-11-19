"""Draw project-specific architecture details informed by the cited model papers."""
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Polygon, Rectangle


OUT = Path(__file__).resolve().parents[1] / "figures/implementation"
INK = "#193149"
FIXED = "#eef3f8"
UPDATED = "#d9ede8"
EDGE = "#516c84"


def canvas(height):
    fig, ax = plt.subplots(figsize=(7.5, height))
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    ax.set(xlim=(0, 7.5), ylim=(0, height), aspect="equal")
    ax.axis("off")
    return fig, ax


def box(ax, x, y, width, height, label, fill=FIXED, fontsize=10):
    ax.add_patch(FancyBboxPatch((x, y), width, height, boxstyle="round,pad=0.02,rounding_size=0.07",
                                linewidth=0.85, edgecolor=EDGE, facecolor=fill))
    ax.text(x + width / 2, y + height / 2, label, ha="center", va="center",
            color=INK, fontsize=fontsize)


def arrow(ax, start, end):
    ax.annotate("", xy=end, xytext=start, arrowprops={"arrowstyle": "->", "lw": 1, "color": INK})


def save(fig, name):
    fig.savefig(OUT / name, dpi=300, facecolor="white")
    plt.close(fig)


def draw_lora():
    fig, ax = canvas(2.15)
    ax.text(3.75, 1.96, "Image tokens enter the attention projections in a ViT block",
            ha="center", va="center", color=INK, fontsize=11)
    for x, name, trainable in [(0.20, "Query (Q)", True), (2.65, "Key (K)", False), (5.10, "Value (V)", True)]:
        box(ax, x, 1.05, 2.18, 0.59,
            f"{name}\nFrozen projection", FIXED)
        if trainable:
            box(ax, x, 0.27, 2.18, 0.54, "Add B(Ax)\nRank-four update", UPDATED)
        else:
            box(ax, x, 0.27, 2.18, 0.54, "Use frozen output", "white")
        arrow(ax, (x + 1.09, 1.02), (x + 1.09, 0.83))
    save(fig, "lora_projection_detail.png")


def draw_msa():
    fig, ax = canvas(2.35)
    ax.text(0.16, 2.10, "Attention path", color=INK, fontweight="bold", fontsize=10)
    box(ax, 0.16, 1.40, 1.65, 0.51, "Tokens x\nNorm + attention")
    box(ax, 2.23, 1.40, 1.58, 0.51, "Spatial adapter", UPDATED)
    box(ax, 4.27, 1.40, 2.85, 0.51, "Add input residual x\nIntermediate tokens z")
    arrow(ax, (1.83, 1.65), (2.21, 1.65))
    arrow(ax, (3.84, 1.65), (4.25, 1.65))
    ax.text(0.16, 1.06, "Feed-forward path", color=INK, fontweight="bold", fontsize=10)
    box(ax, 0.16, 0.20, 1.65, 0.60, "Tokens z\nLayer norm")
    box(ax, 2.23, 0.20, 2.34, 0.60, "Frozen MLP +\nscaled MLP adapter", UPDATED)
    box(ax, 5.02, 0.20, 2.10, 0.60, "Add residual z\nBlock output")
    arrow(ax, (1.83, 0.50), (2.21, 0.50))
    arrow(ax, (4.60, 0.50), (5.00, 0.50))
    save(fig, "msa_adapter_detail.png")


def draw_ppsam():
    fig, ax = canvas(2.35)
    shape = [(0.48, 1.08), (0.66, 0.90), (0.91, 0.98), (1.03, 1.28),
             (0.92, 1.53), (0.67, 1.56), (0.45, 1.38)]

    def prompt(x, bounds, title):
        ax.add_patch(Polygon([(px+x, py) for px, py in shape], color="#9bc7bf"))
        left, bottom, width, height = bounds
        ax.add_patch(Rectangle((x+left, bottom), width, height, fill=False,
                               edgecolor=INK, linewidth=1.3, linestyle="--"))
        ax.text(x+0.75, 1.97, title, color=INK, ha="center", fontsize=10)

    prompt(0.08, (0.45, 0.90, 0.58, 0.66), "Tight annotation box")
    prompt(2.64, (0.24, 0.74, 1.18, 1.01), "Training box")
    prompt(5.22, (0.20, 0.65, 1.08, 1.16), "Evaluation box")
    arrow(ax, (1.47, 1.24), (2.79, 1.24))
    arrow(ax, (1.00, 0.77), (5.35, 0.54))
    ax.text(3.39, 0.30, "Independent expansions per side\n0–50 original-image pixels",
            ha="center", va="center", fontsize=9, color=INK)
    ax.text(5.98, 0.30, "Same fixed expansion per side\n30 pixels in the main comparison",
            ha="center", va="center", fontsize=9, color=INK)
    save(fig, "ppsam_prompt_detail.png")


if __name__ == "__main__":
    draw_lora()
    draw_msa()
    draw_ppsam()
