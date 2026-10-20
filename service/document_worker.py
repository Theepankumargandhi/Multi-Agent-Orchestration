"""Container-only document extraction. No credentials, network or host mounts."""

from __future__ import annotations

import base64
import io
import json
from contextlib import closing
from pathlib import Path

from agentforge_sdk.multimodal_inputs import load_multimodal
from agentforge_sdk.ocr import TesseractOCR
from agentforge_sdk.spatial import ExtractionConfig


def extract() -> None:
    payload = json.loads(Path("/tmp/input.json").read_bytes())
    raw = base64.b64decode(payload["data"], validate=True)
    if not 0 < len(raw) <= 3_000_000:
        raise ValueError("input budget")
    path = Path("/tmp/source" + payload["suffix"])
    path.write_bytes(raw)
    document = load_multimodal(
        path,
        document_id=payload["document_id"],
        source=payload["name"],
        version=payload["version"],
        config=ExtractionConfig(
            max_file_bytes=3_000_000,
            max_pages=10,
            max_pixels=6_000_000,
            max_regions=400,
            max_text_chars=100_000,
            render_dpi=100,
        ),
        ocr=TesseractOCR(timeout_seconds=12) if payload["ocr"] else None,
    )
    previews = []

    def thumbnail(image):
        # Re-encode pixels only: strip metadata, active PDF objects and original compressed streams.
        with image.convert("RGB") as clean:
            clean.thumbnail((1000, 1400))
            output = io.BytesIO()
            clean.save(output, format="PNG")
        data = output.getvalue()
        if len(data) > 700_000:
            raise ValueError("preview budget")
        return base64.b64encode(data).decode("ascii")

    if document.media_type == "application/pdf":
        import pypdfium2

        with closing(pypdfium2.PdfDocument(raw)) as pdf:
            for item in document.pages:
                scale = min(1000 / item.width, 1400 / item.height, 1.5)
                with closing(pdf[item.page - 1]) as page:
                    width, height = page.get_size()
                    if width * height * scale * scale > 2_000_000:
                        raise ValueError("preview geometry")
                    with closing(page.render(scale=scale)) as bitmap, bitmap.to_pil() as image:
                        previews.append(thumbnail(image))
    else:
        from PIL import Image

        with Image.open(path) as image:
            previews.append(thumbnail(image))
    result = json.dumps({"manifest": document.model_dump(mode="json"), "previews": previews})
    if len(result.encode()) > 8_000_000:
        raise ValueError("output budget")
    manifest = document.model_dump_json()
    # Controller reads only bounded text, never an archive extracted by Docker cp on the host.
    if len(manifest.encode()) > 900_000:
        raise ValueError("manifest budget")
    Path("/tmp/manifest.json").write_text(manifest, encoding="utf-8")
    for number, preview in enumerate(previews, 1):
        Path(f"/tmp/preview-{number}.txt").write_text(preview, encoding="ascii")


if __name__ == "__main__":
    try:
        extract()
    except Exception:
        # Do not print document contents, parser errors or local paths into Docker logs.
        raise SystemExit(2) from None
