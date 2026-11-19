"""Render the 2D MSA block with Daisy's Scene scaffold.

Run with the path to Daisy's paper-framework-diagram/scaffold directory.
Requires python-pptx, Pillow and resvg-py. Scaffold revision used: 9173f5c.
"""

import argparse
import json
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import resvg_py
from lxml import etree
from PIL import Image, ImageDraw, ImageOps
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.dml import MSO_LINE_DASH_STYLE
from pptx.enum.shapes import MSO_CONNECTOR
from pptx.enum.text import PP_ALIGN
from pptx.oxml.ns import qn
from pptx.util import Inches, Pt

INK = "#26364A"
MUTED = "#536579"
LINE = "#455A64"
TRAIN = ("#FADECD", "#9E5539")
FROZEN = ("#E6EDF9", "#5E729C")
NORM = ("#FFF2D9", "#92733D")
OUT = Path(__file__).resolve().parent
STEM = "msa_adapter_architecture"
ZOOM = "#9E5539"


def build():
    scene = daisy.Scene()

    def label(x, y, text, size=22, bold=False, fill=INK, anchor="start"):
        scene.text(x, y, text, size=size, bold=bold, fill=fill, anchor=anchor)

    def module(x, y, width, text, palette):
        scene.rect(x, y - 39, width, 78, *palette, rx=13, sw=2)
        lines = text.split("\n")
        for index, line in enumerate(lines):
            offset = 33 * (index - (len(lines) - 1) / 2)
            label(x + width / 2, y + 11 + offset, line, bold=True, anchor="middle")

    def route(*points, arrow=True):
        pairs = list(zip(points, points[1:]))
        for index, (start, end) in enumerate(pairs):
            scene.line(*start, *end, sw=3, color=LINE,
                       arrow=arrow and index == len(pairs) - 1)

    def junction(x, y):
        scene.rect(x - 4, y - 4, 8, 8, LINE, LINE, rx=4, sw=0.5)

    def sum_node(x, y):
        scene.rect(x - 20, y - 20, 40, 40, "#FFFFFF", LINE, rx=20, sw=2.3)
        scene.line(x - 8, y, x + 8, y, sw=2.4, color=INK, arrow=False)
        scene.line(x, y - 8, x, y + 8, sw=2.4, color=INK, arrow=False)

    label(30, 60, "Medical SAM Adapter", size=34, bold=True)
    label(31, 106, "One image-encoder block in MSA-B and MSA-L", fill=MUTED)
    scene.rect(22, 137, 766, 389, "#FAFBFD", "#C7D3E2", rx=20, sw=1.5)
    scene.rect(798, 137, 670, 389, "#FAFBFD", "#C7D3E2", rx=20, sw=1.5)
    label(62, 189, "Attention sublayer", size=25, bold=True)
    label(825, 189, "Feed-forward sublayer", size=25, bold=True)
    label(447, 241, "Residual", fill=MUTED, anchor="middle")
    label(1090, 241, "Residual", fill=MUTED, anchor="middle")

    # Attention → Spatial Adapter (with its own skip) → outer residual sum.
    y = 340
    label(88, y - 4, "Input", bold=True, anchor="middle")
    label(88, y + 33, "tokens", anchor="middle")
    route((142, y), (188, y))
    route((168, y), (168, 260), (747, 260), (747, y - 20))
    junction(168, y)
    module(188, y, 123, "Norm", NORM)
    route((311, y), (334, y))
    module(334, y, 168, "Attention", FROZEN)
    route((502, y), (523, y))
    module(523, y, 187, "Spatial\nAdapter", TRAIN)
    label(617, 422, "Internal skip", fill=MUTED, anchor="middle")
    route((710, y), (727, y))
    sum_node(747, y)

    # Both feed-forward branches take the same normalised token tensor.
    route((767, y), (837, y))
    route((810, y), (810, 260), (1286, 260), (1286, y - 20))
    junction(810, y)
    module(837, y, 112, "Norm", NORM)
    route((949, y), (1000, y))
    module(1000, y, 179, "MLP", FROZEN)
    route((1179, y), (1266, y))
    route((971, y), (971, 455), (1000, 455))
    junction(971, y)
    module(1000, 455, 179, "MLP\nAdapter", TRAIN)
    label(1232, 438, "×0.5", bold=True, anchor="middle")
    route((1179, 455), (1286, 455), (1286, y + 20))
    sum_node(1286, y)
    route((1306, y), (1335, y))
    label(1392, y - 4, "Output", bold=True, anchor="middle")
    label(1392, y + 33, "tokens", anchor="middle")

    scene.rect(36, 550, 29, 24, *TRAIN, rx=5, sw=1.8)
    label(76, 572, "Trainable module")
    scene.rect(389, 550, 29, 24, *FROZEN, rx=5, sw=1.8)
    label(429, 572, "Frozen SAM layer")
    scene.rect(739, 550, 29, 24, *NORM, rx=5, sw=1.8)
    label(779, 572, "Frozen layer norm")
    return scene


def example_images(project, temporary):
    """Reconstruct one visual example from the recorded MSA-B evaluation."""
    sample = "cju0vtox5ain6099360pu62rp"
    recorded = OUT.parents[3] / "analysis/outputs/prompt_variants_msa_20261118/predictions.jsonl"
    with recorded.open(encoding="utf-8") as log:
        row = next(row for row in map(json.loads, log)
                   if (row["id"], row["condition"]) == (sample, "box_30"))
    box = json.loads(row["prompt_json"])["box"]
    prediction_path = project / "results/msa_polyp_train/eval_v3/Kvasir-SEG-val/preds"
    image_path = project / "datasets_link/Kvasir-SEG/images"
    frame = Image.open(image_path / f"{sample}.jpg").convert("RGB")
    prediction = Image.open(prediction_path / f"{sample}.jpg").convert("L")
    assert prediction.size == frame.size
    box_image = frame.copy()
    canvas = ImageDraw.Draw(box_image)
    canvas.rectangle(box, outline="#143F53", width=12)
    canvas.rectangle(box, outline="#A4EEFF", width=7)
    alpha = prediction.point(lambda value: 124 if value > 127 else 0)
    overlay = Image.composite(Image.new("RGB", frame.size, "#39DFC0"), frame, alpha)
    input_path = temporary / "input.png"
    output_path = temporary / "output.png"
    for path, thumbnail in ((input_path, box_image), (output_path, overlay)):
        ImageOps.pad(thumbnail, (346, 296), color="#151B22").save(path)
    return input_path, output_path


def build_overview(input_path, output_path):
    scene = daisy.Scene()

    def label(x, y, text, size=22, bold=False, anchor="start"):
        scene.text(x, y, text, size=size, fill=INK, bold=bold, anchor=anchor)

    def node(x, y, width, name, palette):
        scene.rect(x, y - 36, width, 72, *palette, rx=13, sw=2)
        label(x + width / 2, y + 11, name, bold=True, anchor="middle")

    def route(*points, color=LINE, arrow=True):
        pairs = list(zip(points, points[1:]))
        for index, (start, end) in enumerate(pairs):
            scene.line(*start, *end, sw=3, color=color,
                       arrow=arrow and index == len(pairs) - 1)

    label(30, 52, "A   MSA in the SAM segmentation pipeline", size=30, bold=True)
    scene.rect(22, 78, 1446, 234, "#FAFBFD", "#C7D3E2", rx=20, sw=1.5)
    scene.image(input_path, 46, 98, 173, 148, stroke="#C7D3E2")
    scene.image(output_path, 1269, 98, 173, 148, stroke="#C7D3E2")
    label(132, 283, "Input with box", anchor="middle")
    label(1355, 283, "MSA-B output", anchor="middle")

    route((219, 124), (470, 124))
    label(342, 111, "Box coordinates", size=21, anchor="middle")
    node(470, 124, 310, "Prompt encoder", FROZEN)
    route((780, 124), (882, 124), (882, 169), (931, 169))

    route((219, 222), (470, 222))
    label(342, 210, "RGB frame", size=21, anchor="middle")
    scene.rect(470, 177, 310, 100, *FROZEN, rx=13, sw=2)
    label(625, 209, "ViT image encoder", bold=True, anchor="middle")
    scene.rect(505, 221, 240, 43, *TRAIN, rx=9, sw=2)
    label(625, 253, "Adapter blocks", size=21, bold=True, anchor="middle")
    route((780, 222), (900, 222), (900, 190), (931, 190))
    node(931, 180, 262, "Mask decoder", TRAIN)
    route((1193, 180), (1269, 180))

    route((625, 264), (625, 326), (780, 326), (780, 401),
          color=ZOOM, arrow=False)
    label(30, 381, "B   Inside one adapted ViT block", size=30, bold=True)
    label(1437, 381, "MSA-B and MSA-L", size=21, anchor="end")
    for element in build().els[2:]:
        child = dict(element)
        for key in ("y", "y1", "y2"):
            if key in child:
                child[key] += 267
        if child.get("content") == "Internal skip":
            continue
        scene.els.append(child)
    return scene


def add_native_connectors(scene, path):
    """Replace Daisy's line placeholders with native directional connectors."""
    presentation = Presentation(path)
    slide = presentation.slides[0]
    scale = daisy.SLIDE_W_IN / daisy.W
    for element, shape in zip(scene.els, list(slide.shapes), strict=True):
        if element["kind"] == "line":
            coords = [Inches(element[key] * scale)
                      for key in ("x1", "y1", "x2", "y2")]
            connector = slide.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, *coords)
            connector.line.width = Pt(element["sw"] / daisy.PT_TO_SVG)
            connector.line.color.rgb = RGBColor.from_string(element["color"][1:])
            if element["color"] == ZOOM:
                connector.line.dash_style = MSO_LINE_DASH_STYLE.DASH
            if element["arrow"]:
                etree.SubElement(connector.line._get_or_add_ln(), qn("a:tailEnd"),
                                 type="triangle", w="lg", len="lg")
            shape._element.addnext(connector._element)
            shape._element.getparent().remove(shape._element)
        elif element["kind"] == "rect" and element["rx"] >= element["h"] / 2:
            shape.adjustments[0] = 0.5
        elif element["kind"] == "text":
            if element["anchor"] == "end":
                shape.left = Inches(element["x"] * scale) - shape.width
            shape.text_frame.margin_left = 0
            shape.text_frame.margin_right = 0
            shape.text_frame.margin_top = 0
            shape.text_frame.margin_bottom = 0
            for paragraph in shape.text_frame.paragraphs:
                if element["anchor"] == "end":
                    paragraph.alignment = PP_ALIGN.RIGHT
                for run in paragraph.runs:
                    run.font.name = "Arial"
    presentation.save(path)


def export(scene, stem, output_dir=OUT):
    svg = daisy.to_svg(scene).replace(' filter="url(#sh)"', "")
    svg = svg.replace(f'stroke="{ZOOM}" stroke-width="3"',
                      f'stroke="{ZOOM}" stroke-width="3" stroke-dasharray="8 6"')
    (output_dir / f"{stem}.svg").write_text(svg, encoding="utf-8")
    png = resvg_py.svg_to_bytes(svg_string=svg, zoom=2, sans_serif_family="Arial")
    (output_dir / f"{stem}.png").write_bytes(png)
    path = daisy.export_pptx(scene, output_dir, stem)
    add_native_connectors(scene, path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scaffold", type=Path)
    parser.add_argument("--overview", type=Path, metavar="FYP_PROJECT")
    args = parser.parse_args()
    sys.path.insert(0, str(args.scaffold.resolve()))
    global daisy
    import scene_export as daisy

    daisy.W, daisy.H = 1480, 875 if args.overview else 600
    daisy.SLIDE_W_IN, daisy.SLIDE_H_IN = 13.333, 13.333 * daisy.H / daisy.W
    daisy.PT_TO_SVG = daisy.W / (daisy.SLIDE_W_IN * 72)
    if args.overview:
        with TemporaryDirectory() as directory:
            images = example_images(args.overview, Path(directory))
            export(build_overview(*images), "msa_architecture_overview")
    else:
        export(build(), STEM)


if __name__ == "__main__":
    main()
