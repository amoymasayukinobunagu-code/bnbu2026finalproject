"""Render the shared SAM architecture with Daisy's Scene scaffold.

Pass Daisy's scaffold directory (revision 9173f5c) and the original FYP project.
Diagram references: Kirillov et al., Fig. 4, and Ma et al., Fig. 2b.
"""

import argparse
import json
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

from PIL import Image, ImageDraw, ImageOps

OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(OUT.parent / "msa_daisy_trial"))
import render_msa as artwork

INK, MUTED, LINE = artwork.INK, artwork.MUTED, artwork.LINE
VISION = ("#E6EDF9", "#5E729C")
PROMPT = ("#E0F1EC", "#437768")
DECODER = ("#EEE9F6", "#775F95")
SAMPLE = "cju0ue769mxii08019zqgdbxn"


def make_images(project, directory):
    recorded = OUT.parents[3] / "analysis/outputs/prompt_variants_20261118/predictions.jsonl"
    with recorded.open(encoding="utf-8") as log:
        prompts = {row["condition"]: json.loads(row["prompt_json"])
                   for row in map(json.loads, log)
                   if row["id"] == SAMPLE and row["dataset"] == "Kvasir-SEG-val"}
    frame = Image.open(project / f"datasets_link/Kvasir-SEG/images/{SAMPLE}.jpg").convert("RGB")
    result_dir = project / "results/sam_v3/Kvasir-SEG-val/preds"
    prediction = Image.open(result_dir / f"{SAMPLE}.jpg").convert("L")
    assert prediction.size == frame.size
    bx = frame.copy()
    drawing = ImageDraw.Draw(bx)
    drawing.rectangle(prompts["box_30"]["box"], outline="#153C47", width=16)
    drawing.rectangle(prompts["box_30"]["box"], outline="#A4EEFF", width=9)
    pt = frame.copy()
    x, y = prompts["interior_positive"]["point_coords"][0]
    ImageDraw.Draw(pt).ellipse((x - 14, y - 14, x + 14, y + 14),
                              fill="#89FFD3", outline="#153C47", width=4)
    alpha = prediction.point(lambda value: 140 if value > 127 else 0)
    overlay = Image.composite(Image.new("RGB", frame.size, "#50EAC7"), frame, alpha)
    for name, image, size in (
        ("frame", frame, (226, 192)),
        ("box", bx, (130, 112)),
        ("point", pt, (130, 112)),
        ("output", overlay, (226, 192)),
    ):
        ImageOps.pad(image, tuple(dim * 2 for dim in size), color="#151B22").save(
            directory / f"{name}.png"
        )


def build(scene, directory):
    def label(x, y, text, size=22, bold=False, fill=INK, anchor="middle"):
        scene.text(x, y, text, size=size, bold=bold, fill=fill, anchor=anchor)

    def route(*points):
        pairs = list(zip(points, points[1:]))
        for index, (start, end) in enumerate(pairs):
            scene.line(*start, *end, sw=3, color=LINE, arrow=index == len(pairs) - 1)

    def encoder(x, y, text, palette, stacked=False):
        width, height = 238, 104
        if stacked:
            for offset in (16, 8):
                scene.rect(x + offset, y - height / 2 - offset, width, height,
                           *palette, rx=13, sw=1.8)
        scene.rect(x, y - height / 2, width, height, *palette, rx=13, sw=2)
        for offset, line in zip((-6, 32), text.split("\n"), strict=True):
            label(x + width / 2, y + offset, line, bold=True)

    label(30, 51, "Shared SAM architecture for polyp segmentation",
          size=31, bold=True, anchor="start")
    scene.rect(22, 80, 1436, 523, "#FAFBFD", "#C7D3E2", rx=20, sw=1.5)

    scene.image(directory / "frame.png", 83, 126, 226, 192, stroke="#A6B6C7")
    label(196, 361, "Endoscopy frame", bold=True)
    route((309, 222), (421, 222))
    encoder(421, 222, "ViT image\nencoder", VISION, stacked=True)
    label(540, 318, "ViT-B or ViT-L", size=21, fill=MUTED)
    route((659, 222), (731, 222))
    for row in range(3):
        for col in range(5):
            fill = ["#D3DFF4", "#BECDEB", "#93AAD6"][(col + 2 * row) % 3]
            scene.rect(743 + col * 22, 190 + row * 22, 19, 19, fill, "#647B9F", rx=3, sw=0.7)
    label(807, 168, "Image embedding", size=21)

    scene.rect(40, 406, 312, 167, "#F0F8F6", "#9EBEB5", rx=12, sw=1.5)
    scene.image(directory / "box.png", 50, 420, 130, 112)
    scene.image(directory / "point.png", 212, 420, 130, 112)
    label(115, 562, "Box", bold=True)
    label(196, 562, "or", size=21, fill=MUTED)
    label(277, 562, "Point", bold=True)
    route((352, 479), (421, 479))
    encoder(421, 479, "Prompt\nencoder", PROMPT)
    route((659, 479), (731, 479))
    for index, fill in enumerate(("#AFD7CD", "#87BDAF", "#C5E4DD", "#A2CFC3")):
        scene.rect(743 + 28 * index, 468, 22, 22, fill, PROMPT[1], rx=5, sw=1)
    label(815, 540, "Prompt embeddings", size=21)

    route((853, 222), (934, 222), (934, 321), (978, 321))
    route((853, 479), (934, 479), (934, 378), (978, 378))
    scene.rect(978, 284, 188, 130, *DECODER, rx=16, sw=2)
    label(1072, 342, "Mask", bold=True)
    label(1072, 380, "decoder", bold=True)
    route((1166, 349), (1211, 349))
    scene.image(directory / "output.png", 1213, 253, 226, 192, stroke="#A6B6C7")
    label(1326, 483, "Predicted mask", bold=True)
    label(1326, 520, "+ estimated IoU", size=21)
    label(1326, 555, "SAM-B with box", size=21, fill=MUTED)

    label(32, 657, "The same image embedding can be reused with different prompts.",
          fill=MUTED, anchor="start")
    return scene


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scaffold", type=Path)
    parser.add_argument("project", type=Path)
    args = parser.parse_args()
    sys.path.insert(0, str(args.scaffold.resolve()))
    import scene_export as daisy

    artwork.daisy = daisy
    daisy.W, daisy.H = 1480, 682
    daisy.SLIDE_W_IN, daisy.SLIDE_H_IN = 13.333, 13.333 * daisy.H / daisy.W
    daisy.PT_TO_SVG = daisy.W / (daisy.SLIDE_W_IN * 72)
    with TemporaryDirectory() as temporary:
        directory = Path(temporary)
        make_images(args.project, directory)
        artwork.export(build(daisy.Scene(), directory), "sam_shared_architecture", OUT)


if __name__ == "__main__":
    main()
