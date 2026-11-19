"""Render PP-SAM with the Daisy Scene scaffold and a Kvasir-SEG training example.

Pass the paths to Daisy's scaffold (revision 9173f5c) and the original FYP project.
The architecture follows Rahman et al., CVPRW 2024, Fig. 1 and the local training code.
"""

import argparse
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

from PIL import Image, ImageDraw, ImageOps

OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(OUT.parent / "msa_daisy_trial"))
import render_msa as artwork

INK, MUTED, LINE = artwork.INK, artwork.MUTED, artwork.LINE
TRAIN, FROZEN, ZOOM = artwork.TRAIN, artwork.FROZEN, artwork.ZOOM
SAMPLE = "cju1csmlc4ht10799b8ymmghg"
TIGHT_COLOUR = "#A4EEFF"
PERTURBED_COLOUR = "#FFD078"
TILE = (250, 223)


def make_tiles(project, directory):
    dataset = project / "datasets_link/Kvasir-SEG"
    split = OUT.parents[3] / "splits/Kvasir-SEG/Data-split/train.txt"
    assert SAMPLE in {Path(line.strip()).stem for line in split.read_text().splitlines()}
    frame = Image.open(dataset / f"images/{SAMPLE}.jpg").convert("RGB")
    mask = Image.open(dataset / f"masks/{SAMPLE}.jpg").convert("L")
    assert frame.size == mask.size
    mask = mask.point(lambda value: 255 if value else 0)
    left, top, right, bottom = mask.getbbox()
    tight = (left, top, right - 1, bottom - 1)

    # Illustrative in-range offsets, in native image pixels, left/top/right/bottom.
    expanded = (max(0, left - 45), max(0, top - 22),
                min(frame.width, right - 1 + 28),
                min(frame.height, bottom - 1 + 50))
    tight_frame = frame.copy()
    expanded_frame = frame.copy()
    for image in (tight_frame, expanded_frame):
        draw = ImageDraw.Draw(image)
        draw.rectangle(tight, outline="#194756", width=10)
        draw.rectangle(tight, outline=TIGHT_COLOUR, width=5)
    draw = ImageDraw.Draw(expanded_frame)
    draw.rectangle(expanded, outline="#503420", width=13)
    draw.rectangle(expanded, outline=PERTURBED_COLOUR, width=7)

    for name, image in (("annotation", mask.convert("RGB")),
                        ("tight", tight_frame), ("expanded", expanded_frame)):
        ImageOps.pad(image, tuple(2 * dim for dim in TILE), color="#111820").save(
            directory / f"{name}.png"
        )


def build(scene, directory):
    def label(x, y, text, size=22, bold=False, fill=INK, anchor="start"):
        scene.text(x, y, text, size=size, bold=bold, fill=fill, anchor=anchor)

    def route(*points, color=LINE, arrow=True):
        pairs = list(zip(points, points[1:]))
        for index, (start, end) in enumerate(pairs):
            scene.line(*start, *end, sw=3, color=color,
                       arrow=arrow and index == len(pairs) - 1)

    def node(x, y, width, text, palette):
        scene.rect(x, y - 35, width, 70, *palette, rx=13, sw=2)
        label(x + width / 2, y + 11, text, bold=True, anchor="middle")

    def panel(y, height):
        scene.rect(22, y, 1436, height, "#FAFBFD", "#C7D3E2", rx=20, sw=1.5)

    label(30, 50, "A   PP-SAM modules during fine-tuning", size=30, bold=True)
    panel(76, 241)
    label(52, 119, "Initial checkpoint: SAM ViT-B", size=22, bold=True)
    scene.line(52, 140, 1425, 140, sw=1.6, color="#C7D3E2", arrow=False)
    label(56, 192, "RGB frame")
    label(56, 275, "Perturbed box")
    route((295, 182), (374, 182))
    route((295, 264), (374, 264))
    node(374, 182, 337, "ViT image encoder", TRAIN)
    node(374, 264, 337, "Prompt encoder", TRAIN)
    route((711, 182), (815, 182), (815, 209), (888, 209))
    route((711, 264), (815, 264), (815, 234), (888, 234))
    node(888, 222, 263, "Mask decoder", FROZEN)
    route((1151, 222), (1214, 222))
    label(1231, 215, "Predicted", bold=True)
    label(1231, 253, "polyp mask", bold=True)

    label(30, 377, "B   How the training prompt is constructed", size=30, bold=True)
    panel(402, 395)
    label(58, 447, "Sample each side’s expansion independently (0–50 px), then clip to image bounds.",
          size=22)
    centres = (268, 740, 1212)
    for centre, name, caption in zip(
        centres,
        ("annotation", "tight", "expanded"),
        ("Annotation mask", "Tight box", "Example perturbed box"),
        strict=True,
    ):
        scene.image(directory / f"{name}.png", centre - TILE[0] / 2, 478,
                    *TILE, stroke="#A6B6C7")
        label(centre, 747, caption, bold=True, anchor="middle")
    route((418, 589), (594, 589))
    route((892, 589), (1067, 589))
    label(506, 573, "Enclose", size=21, fill=MUTED, anchor="middle")
    label(980, 573, "Expand", size=21, fill=MUTED, anchor="middle")
    route((171, 286), (171, 329), (1177, 329), (1177, 402),
          color=ZOOM, arrow=False)

    for x, text, palette in ((34, "Trainable module", TRAIN), (423, "Frozen module", FROZEN)):
        scene.rect(x, 825, 29, 24, *palette, rx=5, sw=1.8)
        label(x + 40, 847, text)
    label(34, 903, "Main evaluation: fixed 30-pixel expansion on each side.", fill=MUTED)
    return scene


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scaffold", type=Path)
    parser.add_argument("project", type=Path)
    args = parser.parse_args()
    sys.path.insert(0, str(args.scaffold.resolve()))
    import scene_export as daisy

    artwork.daisy = daisy
    daisy.W, daisy.H = 1480, 930
    daisy.SLIDE_W_IN, daisy.SLIDE_H_IN = 13.333, 13.333 * daisy.H / daisy.W
    daisy.PT_TO_SVG = daisy.W / (daisy.SLIDE_W_IN * 72)
    with TemporaryDirectory() as temporary:
        directory = Path(temporary)
        make_tiles(args.project, directory)
        artwork.export(build(daisy.Scene(), directory), "ppsam_architecture", OUT)


if __name__ == "__main__":
    main()
