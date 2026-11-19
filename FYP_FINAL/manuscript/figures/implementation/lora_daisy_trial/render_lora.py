"""Render the shared LoRA structure with Daisy's Scene scaffold.

Pass the path to paper-framework-diagram/scaffold (Daisy revision 9173f5c).
References: SAMed (arXiv:2304.13785), Figs. 2–3; LoRA (arXiv:2106.09685), Fig. 1.
"""

import argparse
import sys
from pathlib import Path

OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(OUT.parent / "msa_daisy_trial"))
import render_msa as artwork

INK, MUTED, LINE = artwork.INK, artwork.MUTED, artwork.LINE
TRAIN, FROZEN, ZOOM = artwork.TRAIN, artwork.FROZEN, artwork.ZOOM


def build(scene):
    def label(x, y, text, size=22, bold=False, fill=INK, anchor="start"):
        scene.text(x, y, text, size=size, bold=bold, fill=fill, anchor=anchor)

    def route(*points, color=LINE, arrow=True):
        pairs = list(zip(points, points[1:]))
        for index, (start, end) in enumerate(pairs):
            scene.line(*start, *end, sw=3, color=color,
                       arrow=arrow and index == len(pairs) - 1)

    def panel(y, height):
        scene.rect(22, y, 1436, height, "#FAFBFD", "#C7D3E2", rx=20, sw=1.5)

    def node(x, y, width, height, text, palette):
        scene.rect(x, y - height / 2, width, height, *palette, rx=13, sw=2)
        lines = text.split("\n")
        for index, line in enumerate(lines):
            offset = 34 * (index - (len(lines) - 1) / 2)
            label(x + width / 2, y + 11 + offset, line, bold=True, anchor="middle")

    def junction(x, y):
        scene.rect(x - 4, y - 4, 8, 8, LINE, LINE, rx=4, sw=0.5)

    def addition(x, y):
        scene.rect(x - 20, y - 20, 40, 40, "#FFFFFF", LINE, rx=20, sw=2.3)
        scene.line(x - 8, y, x + 8, y, sw=2.4, color=INK, arrow=False)
        scene.line(x, y - 8, x, y + 8, sw=2.4, color=INK, arrow=False)

    label(30, 50, "A   Shared SAM architecture and initial weights", size=30, bold=True)
    panel(76, 247)
    label(52, 121, "Starting weights:", bold=True)
    label(362, 121, "SAM ViT-B  →  SAMed")
    label(809, 121, "MedSAM ViT-B  →  MedSAM+LoRA")
    scene.line(52, 143, 1426, 143, sw=1.6, color="#C7D3E2", arrow=False)
    label(56, 191, "Box prompt")
    label(56, 263, "RGB frame")
    route((247, 181), (356, 181))
    route((247, 252), (356, 252))
    node(356, 181, 344, 56, "Prompt encoder", TRAIN)
    scene.rect(356, 219, 344, 91, *FROZEN, rx=13, sw=2)
    label(528, 252, "ViT image encoder", bold=True, anchor="middle")
    scene.rect(412, 262, 232, 39, *TRAIN, rx=8, sw=1.7)
    label(528, 291, "Q/V LoRA", size=21, bold=True, anchor="middle")
    route((700, 181), (800, 181), (800, 211), (880, 211))
    route((700, 253), (800, 253), (800, 233), (880, 233))
    node(880, 222, 258, 75, "Mask decoder", TRAIN)
    route((1138, 222), (1219, 222))
    label(1233, 213, "Polyp mask", bold=True)
    label(1233, 250, "+ mask score", size=21)

    route((528, 301), (528, 339), (829, 339), (829, 407),
          color=ZOOM, arrow=False)
    label(30, 388, "B   LoRA inside ViT self-attention", size=30, bold=True)
    panel(409, 475)
    label(113, 638, "Input", bold=True, anchor="middle")
    label(113, 676, "tokens", anchor="middle")
    route((166, 650), (235, 650))
    node(235, 650, 240, 112, "QKV\nprojection", FROZEN)

    # The frozen linear layer emits the three chunks of its fused projection.
    query_y, key_y, value_y = 552, 650, 749
    route((475, 613), (542, 613), (542, query_y), (1003, query_y))
    route((475, 687), (542, 687), (542, value_y), (1003, value_y))

    # Both low-rank branches use the original input tokens.
    branch_x = 201
    for branch_y, sum_y, label_y, name, name_y in (
        (472, query_y, 537, "Query update", 459),
        (832, value_y, 793, "Value update", 819),
    ):
        route((branch_x, key_y), (branch_x, branch_y), (562, branch_y))
        label(417, name_y, name, size=21, fill=MUTED, anchor="middle")
        node(562, branch_y, 124, 60, "A", TRAIN)
        route((686, branch_y), (827, branch_y))
        node(827, branch_y, 124, 60, "B", TRAIN)
        route((951, branch_y), (1023, branch_y),
              (1023, sum_y - 20 if branch_y < sum_y else sum_y + 20))
        label(624, label_y, "dimension → 4", size=21, fill=MUTED, anchor="middle")
        label(889, label_y, "4 → dimension", size=21, fill=MUTED, anchor="middle")
        addition(1023, sum_y)
    junction(branch_x, key_y)

    node(1204, 650, 218, 272, "Attention\nmixing", ("#FFFFFF", LINE))
    for text, y, origin in (("Q", query_y, 1043), ("K", key_y, 475),
                            ("V", value_y, 1043)):
        route((origin, y), (1204, y))
        label(1110, y - 11, text, bold=True, anchor="middle")

    for x, text, palette in ((34, "Trainable module", TRAIN),
                             (423, "Frozen layer", FROZEN)):
        scene.rect(x, 911, 29, 24, *palette, rx=5, sw=1.8)
        label(x + 40, 934, text)
    label(820, 934, "Token dimension = 768", fill=MUTED)
    return scene


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scaffold", type=Path)
    args = parser.parse_args()
    sys.path.insert(0, str(args.scaffold.resolve()))
    import scene_export as daisy

    artwork.daisy = daisy
    daisy.W, daisy.H = 1480, 960
    daisy.SLIDE_W_IN, daisy.SLIDE_H_IN = 13.333, 13.333 * daisy.H / daisy.W
    daisy.PT_TO_SVG = daisy.W / (daisy.SLIDE_W_IN * 72)
    artwork.export(build(daisy.Scene()), "lora_samed_medsam_architecture", OUT)


if __name__ == "__main__":
    main()
