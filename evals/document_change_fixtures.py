"""Small authored PDF revisions, not a held-out document change benchmark."""

from pathlib import Path


def create_change_fixtures(directory: Path) -> dict[str, Path]:
    from reportlab.pdfgen.canvas import Canvas

    directory.mkdir(parents=True, exist_ok=True)
    paths = {}
    for name in ("v1", "v2", "unit-change", "ambiguous", "same-content"):
        path = directory / f"{name}.pdf"
        canvas = Canvas(str(path), pagesize=(612, 792), invariant=1)
        canvas.setTitle("Authored revision control: " + name)
        if name == "v2":
            canvas.setFont("Helvetica-Bold", 20)
            canvas.drawString(50, 730, "New cover page")
            canvas.setFont("Helvetica", 12)
            canvas.drawString(50, 695, "Authored control: the report moved to physical page 2.")
            canvas.showPage()
        canvas.setFont("Helvetica-Bold", 20)
        canvas.drawString(50, 735, "Atlas revision report")
        canvas.setFont("Helvetica", 12)
        canvas.drawString(50, 705, "Atlas p95 latency is 180 ms.")
        revenue = ("Revenue", "USD 120.50", "EUR 150.75" if name == "unit-change" else "USD 150.75")
        requests = ("Requests", "100", "150" if name == "v2" else "125")
        rows = [("Metric", "Before", "After"), revenue, requests]
        if name == "v2":
            rows = [rows[0], requests, revenue]
        if name == "ambiguous":
            rows[1] = ("Requests", "80", "90")
        xs, ys = [50, 240, 395, 550], [650, 602, 554, 506]
        for x in xs:
            canvas.line(x, ys[0], x, ys[-1])
        for y in ys:
            canvas.line(xs[0], y, xs[-1], y)
        for row, values in enumerate(rows):
            for column, value in enumerate(values):
                canvas.drawString(xs[column] + 10, ys[row] - 30, value)
        canvas.setFont("Helvetica", 10)
        canvas.drawString(50, 465, "Authored PDF control. No real-world accuracy claim.")
        canvas.save()
        paths[name] = path
    return paths
