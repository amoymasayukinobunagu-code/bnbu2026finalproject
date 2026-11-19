"""Draw the shared SAM architecture and the module updates used in the thesis."""
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch


def main():
    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    ax.set(xlim=(0, 7.5), ylim=(0, 4.4))
    ax.axis("off")
    ink = "#183149"
    pale = "#edf3f8"
    accent = "#d9ede8"

    def box(x, y, width, height, label, fill=pale):
        ax.add_patch(FancyBboxPatch(
            (x, y), width, height, boxstyle="round,pad=0.014,rounding_size=0.08",
            edgecolor="#55728b", facecolor=fill, lw=0.9,
        ))
        ax.text(x + width / 2, y + height / 2, label, ha="center", va="center",
                color=ink, fontsize=10)

    def arrow(start, end):
        ax.annotate("", xy=end, xytext=start,
                    arrowprops={"arrowstyle": "->", "color": ink, "lw": 1})

    box(0.16, 3.56, 1.12, 0.44, "RGB image")
    box(0.16, 2.86, 1.12, 0.44, "Box / point")
    box(1.66, 3.56, 1.80, 0.44, "ViT image encoder", accent)
    box(1.66, 2.86, 1.80, 0.44, "Prompt encoder", accent)
    box(4.06, 3.17, 1.52, 0.55, "Mask decoder", accent)
    box(6.04, 3.17, 1.30, 0.55, "Mask + score")
    arrow((1.28, 3.78), (1.66, 3.78))
    arrow((1.28, 3.08), (1.66, 3.08))
    arrow((3.46, 3.78), (4.06, 3.55))
    arrow((3.46, 3.08), (4.06, 3.34))
    arrow((5.58, 3.44), (6.04, 3.44))

    ax.text(0.18, 2.52, "Modules updated during Kvasir-SEG adaptation",
            fontsize=11, fontweight="bold", color=ink)
    columns = ("Model group", "Image encoder", "Prompt encoder", "Mask decoder")
    x = (0.17, 2.12, 4.29, 5.82)
    widths = (1.82, 2.05, 1.42, 1.50)
    for col, origin, width in zip(columns, x, widths):
        box(origin, 1.98, width, 0.37, col)
    rows = (
        ("SAM / MedSAM", "Fixed", "Fixed", "Fixed"),
        ("SAMed / MedSAM+LoRA", "Q, V low-rank updates", "Updated", "Updated"),
        ("MSA-B / MSA-L", "Spatial and MLP adapters", "Fixed", "Updated"),
        ("PP-SAM", "Full ViT updated", "Updated", "Fixed"),
    )
    for index, row in enumerate(rows):
        for col, origin, width in zip(row, x, widths):
            box(origin, 1.49 - index * 0.42, width, 0.34, col, "#ffffff")

    output = Path(__file__).resolve().parents[1] / "figures/implementation/model_architectures.png"
    fig.savefig(output, dpi=300, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
