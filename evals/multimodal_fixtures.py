"""Authored fixtures, not a held-out document benchmark. No downloaded/private documents."""

from __future__ import annotations

from pathlib import Path


def create_fixtures(directory: Path) -> dict[str, Path]:
    from PIL import Image, ImageDraw, ImageFont
    from pypdf import PdfReader, PdfWriter
    from reportlab.lib.utils import ImageReader
    from reportlab.pdfgen.canvas import Canvas

    directory.mkdir(parents=True, exist_ok=True)
    paths = {name: directory / (name + (".png" if name == "scan" else ".pdf")) for name in
             ("native", "scanned", "mixed", "merged", "rotated", "blank", "scan")}

    def draw_native(canvas):
        canvas.setTitle("Authored Atlas evidence control - not customer data")
        canvas.setFont("Helvetica-Bold", 20)
        canvas.drawString(50, 738, "Atlas evidence report")
        canvas.setFont("Helvetica", 12)
        canvas.drawString(50, 708, "Atlas p95 latency is 180 ms. Owner approval is required.")
        rows = [("Metric", "Q1", "Q2"), ("Revenue", "USD 120.50", "USD 150.75"),
                ("Requests", "100", "125"), ("Errors", "0", "0"),
                ("Ambiguous locale", "1,25", "2,50")]
        xs, top, height = [50, 240, 395, 550], 660, 42
        for x in xs:
            canvas.line(x, top, x, top - height * len(rows))
        for r in range(len(rows) + 1):
            canvas.line(xs[0], top - height * r, xs[-1], top - height * r)
        for row, values in enumerate(rows):
            for column, value in enumerate(values):
                canvas.drawString(xs[column] + 12, top - height * row - 26, value)
        canvas.setFont("Helvetica", 10)
        canvas.drawString(50, 405, "Units are explicit per cell. The SDK does not infer financial meaning.")
        canvas.drawString(50, 50, "Authored SDK control | Page 1")

    canvas = Canvas(str(paths["native"]), pagesize=(612, 792), invariant=1)
    draw_native(canvas)
    canvas.save()

    image = Image.new("RGB", (1500, 800), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default(size=48)
    for y, text in zip((140, 240, 340), ("Atlas recovery drill", "Recovery time is 42 seconds.", "Approved by operator."), strict=True):
        draw.text((100, y), text, font=font, fill="black")
    image.save(paths["scan"])
    for name in ("scanned", "mixed"):
        canvas = Canvas(str(paths[name]), pagesize=(750, 400), invariant=1)
        if name == "mixed":
            canvas.setPageSize((612, 792))
            draw_native(canvas)
            canvas.showPage()
            canvas.setPageSize((750, 400))
        canvas.drawImage(ImageReader(image), 0, 0, width=750, height=400)
        canvas.save()
    image.close()

    canvas = Canvas(str(paths["merged"]), pagesize=(612, 792), invariant=1)
    canvas.setFont("Helvetica", 12)
    canvas.drawString(50, 720, "Merged cells are intentionally unsupported")
    for y in (650, 610, 570):
        canvas.line(50, y, 550, y)
    for x in (50, 550):
        canvas.line(x, 650, x, 570)
    canvas.line(300, 610, 300, 570)
    for x, y, text in ((65, 625, "Merged header"), (65, 585, "10"), (315, 585, "20")):
        canvas.drawString(x, y, text)
    canvas.save()
    canvas = Canvas(str(paths["blank"]), pagesize=(612, 792), invariant=1)
    canvas.showPage()
    canvas.save()
    writer = PdfWriter()
    writer.add_page(PdfReader(paths["native"]).pages[0].rotate(90))
    with paths["rotated"].open("wb") as stream:
        writer.write(stream)
    return paths
