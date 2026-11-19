"""Render a Chapter 1 task example from a recorded Kvasir-SEG prediction."""

import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps


MANUSCRIPT = Path(__file__).resolve().parents[1]
PROJECT = MANUSCRIPT.parent.parent / "FYP"
SAMPLE = "cju15ptjtppz40988odsm9azx"
PANEL_SIZE = (474, 378)
FONT_DIR = Path("C:/Windows/Fonts")


def binary_mask(path):
    with Image.open(path) as image:
        return image.convert("L").point(lambda value: 255 if value > 127 else 0)


def main():
    dataset = PROJECT / "datasets_link/Kvasir-SEG"
    records = MANUSCRIPT.parent / "analysis/outputs/prompt_variants_20261118/predictions.jsonl"
    with records.open(encoding="utf-8") as stream:
        record = next(row for row in map(json.loads, stream)
                      if (row["dataset"], row["id"], row["condition"])
                      == ("Kvasir-SEG-val", SAMPLE, "box_30"))
    box = json.loads(record["prompt_json"])["box"]
    with Image.open(dataset / "images" / f"{SAMPLE}.jpg") as image:
        rgb = image.convert("RGB")
    reference = binary_mask(dataset / "masks" / f"{SAMPLE}.jpg")
    predictor = PROJECT / "results/sam_v3/Kvasir-SEG-val/preds"
    prediction = binary_mask(predictor / f"{SAMPLE}.jpg")
    assert rgb.size == reference.size == prediction.size

    measured_dice = 2 * sum(bool(a and b) for a, b in zip(
        reference.tobytes(), prediction.tobytes(), strict=True
    )) / (reference.histogram()[255] + prediction.histogram()[255])
    assert abs(measured_dice - record["dice"]) < 1e-6

    prompted = rgb.copy()
    draw = ImageDraw.Draw(prompted)
    draw.rectangle(box, outline="#16333C", width=12)
    draw.rectangle(box, outline="#A4EEFF", width=7)

    def overlay(mask, colour):
        tint = Image.new("RGB", rgb.size, colour)
        return Image.composite(tint, rgb, mask.point(lambda value: 145 if value else 0))

    panels = (
        (prompted, "Bounding-box prompt", "Model input"),
        (overlay(prediction, "#FFD36E"), "SAM-B prediction", "Model output"),
        (overlay(reference, "#56E8E0"), "Reference annotation", "Evaluation reference"),
    )
    figure = Image.new("RGB", (1560, 538), "white")
    draw = ImageDraw.Draw(figure)
    title_font = ImageFont.truetype(str(FONT_DIR / "arialbd.ttf"), 38)
    note_font = ImageFont.truetype(str(FONT_DIR / "arial.ttf"), 34)
    for index, (image, title, note) in enumerate(panels):
        x = 28 + index * 515
        thumb = ImageOps.pad(image, PANEL_SIZE, color="#101820")
        figure.paste(thumb, (x, 21))
        centre = x + PANEL_SIZE[0] / 2
        draw.text((centre, 437), title, font=title_font, fill="#26364A", anchor="mm")
        draw.text((centre, 487), note, font=note_font, fill="#536579", anchor="mm")

    destination = MANUSCRIPT / "figures/introduction/prompted_polyp_task.png"
    destination.parent.mkdir(exist_ok=True)
    figure.save(destination)
    print(destination)


if __name__ == "__main__":
    main()
