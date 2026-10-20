"""Container-only PDFium text probe. Never receives expected cell text or invokes OCR."""

import base64
import hashlib
import json
from contextlib import closing
from importlib.metadata import version
from pathlib import Path

from agentforge_sdk.models import fingerprint
from service.document_cell_contracts import CellProbe, CellProbeResult, CellSample
from service.document_cell_geometry import page_geometry


def probe_pdf(raw: bytes, probe: CellProbe) -> CellProbeResult:
    # This function is only invoked in the bounded worker for untrusted originals.
    import pypdfium2

    probe = CellProbe.model_validate(probe.model_dump())
    if not 0 < len(raw) <= 3_000_000 or not raw.startswith(b"%PDF-"):
        raise ValueError("PDF input budget")
    if hashlib.sha256(raw).hexdigest() != probe.file_sha256:
        raise ValueError("PDF byte binding")
    samples, total_chars = {}, 0
    with closing(pypdfium2.PdfDocument(raw)) as pdf:
        if len(pdf) != probe.page_count:
            raise ValueError("PDF page binding")
        for number in sorted({target.page for target in probe.targets}):
            targets = [target for target in probe.targets if target.page == number]
            with closing(pdf[number - 1]) as page:
                eligible = [target for target in targets if page_geometry(page, target)]
                for target in targets:
                    if target not in eligible:
                        samples[target.region_id] = CellSample(
                            region_id=target.region_id, page=number, status="unsupported_geometry"
                        )
                if not eligible:
                    continue
                with closing(page.get_textpage()) as textpage:
                    count = textpage.count_chars()
                    total_chars += count
                    for target in eligible:
                        status, text = "budget_exceeded", None
                        if 0 <= count <= 20_000 and total_chars <= 100_000:
                            b = target.box
                            # Primary coordinates are top-left normalized; PDFium uses bottom-left PDF units.
                            text = " ".join(
                                textpage.get_text_bounded(
                                    left=b.x0 * target.width,
                                    bottom=(1 - b.y1) * target.height,
                                    right=b.x1 * target.width,
                                    top=(1 - b.y0) * target.height,
                                    errors="strict",
                                ).split()
                            )
                            if len(text) > 4000 or "\x00" in text:
                                text = None
                            else:
                                status = "observed" if text else "empty"
                                text = text or None
                        samples[target.region_id] = CellSample(
                            region_id=target.region_id, page=number, status=status, text=text
                        )
    payload = dict(
        protocol="agentforge-pdf-cell-samples-v1",
        file_sha256=probe.file_sha256,
        request_fingerprint=fingerprint(probe.model_dump(mode="json")),
        page_count=probe.page_count,
        engine="pypdfium2:" + version("pypdfium2"),
        samples=[samples[target.region_id].model_dump(mode="json") for target in probe.targets],
    )
    return CellProbeResult.model_validate_json(json.dumps({**payload, "fingerprint": fingerprint(payload)}))


def main():
    payload = json.loads(Path("/tmp/input.json").read_bytes())
    if set(payload) != {"data", "probe"}:
        raise ValueError("PDF probe envelope")
    result = probe_pdf(
        base64.b64decode(payload["data"], validate=True),
        CellProbe.model_validate_json(json.dumps(payload["probe"])),
    ).model_dump_json()
    if len(result.encode()) > 600_000:
        raise ValueError("PDF probe output budget")
    Path("/tmp/cells.json").write_text(result, encoding="utf-8")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        raise SystemExit(2) from None
