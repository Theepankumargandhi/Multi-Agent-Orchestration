"""Authored scan and manual annotations, never represented as an actual human approval."""

from pathlib import Path


def create_review_fixture(directory: Path):
    from PIL import Image, ImageDraw, ImageFont

    from agentforge_sdk import Box, PageCorrection, ReviewedRegionInput, load_multimodal
    from agentforge_sdk.models import fingerprint

    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "review-table.png"
    image = Image.new("RGB", (1000, 600), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default(size=32)
    draw.text((60, 40), "Authored scan review control", font=font, fill="black")
    rows = [("Metric", "Before", "After"), ("Requests", "100", "125"), ("Errors", "2", "1")]
    xs, ys = [60, 400, 660, 940], [140, 260, 380, 500]
    regions = [ReviewedRegionInput(text="Authored scan review control",
                                   box=Box(x0=0.06, y0=0.06, x1=0.7, y1=0.15))]
    for x in xs:
        draw.line((x, ys[0], x, ys[-1]), fill="black", width=2)
    for y in ys:
        draw.line((xs[0], y, xs[-1], y), fill="black", width=2)
    for row, values in enumerate(rows):
        for column, value in enumerate(values):
            draw.text((xs[column] + 16, ys[row] + 35), value, font=font, fill="black")
            regions.append(ReviewedRegionInput(kind="table_cell", text=value, table_id="review-table",
                                              row=row, column=column,
                                              box=Box(x0=xs[column] / 1000, y0=ys[row] / 600,
                                                      x1=xs[column + 1] / 1000, y1=ys[row + 1] / 600)))
    image.save(path)
    image.close()
    original = load_multimodal(path, document_id="review-control", version="r1")
    correction = PageCorrection(page=1, original_page_sha256=fingerprint(original.pages[0].model_dump(mode="json")),
                                regions=tuple(regions), attestation="complete_page_review")
    return path, original, correction
