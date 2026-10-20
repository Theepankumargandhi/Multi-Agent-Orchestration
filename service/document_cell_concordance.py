"""Cited-cell text concordance, not a second numerical verdict or source approval."""

import json
import re

from agentforge_sdk.models import fingerprint
from agentforge_sdk.reconciliation import FactReconciliation
from service.document_cell_contracts import (
    MAX_CELLS,
    MAX_PIXEL_CELLS,
    CellProbe,
    CellProbeResult,
    CellTarget,
    PixelProbe,
    PixelProbeResult,
)


def concordance_defaults(*, pixels=False):
    return {
        "probe_executed": False,
        "engine": None,
        "probe_fingerprint": None,
        "requested_cells": 0,
        "counts": {"agree": 0, "different": 0, "uncheckable": 0},
        "cells": [],
        "numeric_verdicts_rechecked": False,
        "other_sources_checked": False,
        **(
            {
                "raster_pages": 0,
                "rendered_pixels": 0,
                "ocr_calls": 0,
                "renderer": None,
                "profile": "pdfium216-edge3-tesseract6-eng-v1",
            }
            if pixels
            else {}
        ),
    }


def check_cells(runtime, raw, source_id, document, inputs, *, pixels=False):
    if document.media_type != "application/pdf":
        return {"status": "unsupported_source"}
    cited = set()
    for _, receipt, parents in inputs.values():
        for fact in [receipt] if isinstance(receipt, FactReconciliation) else parents:
            for observation in fact.observations:
                if observation.source_id == source_id:
                    cited.update(c.region_id for c in observation.citations)
    if not cited:
        return {"status": "no_cited_cells"}
    if len(cited) > (MAX_PIXEL_CELLS if pixels else MAX_CELLS):
        return {"status": "cell_budget_exceeded", "requested_cells": len(cited)}
    regions = {r.region_id: (page, r) for page in document.pages for r in page.regions}
    expected, targets = {}, []
    for region_id in sorted(cited):
        page, region = regions[region_id]
        if page.method != "native" or page.status != "ready" or region.kind != "table_cell":
            return {"status": "unsupported_evidence", "requested_cells": len(cited)}
        expected[region_id] = region.text
        targets.append(
            CellTarget(
                region_id=region_id, page=page.page, width=page.width, height=page.height, box=region.box
            )
        )
    if pixels and len({target.page for target in targets}) > 2:
        return {"status": "page_budget_exceeded", "requested_cells": len(targets)}
    probe_type = PixelProbe if pixels else CellProbe
    probe = probe_type(
        file_sha256=document.file_sha256, page_count=len(document.pages), targets=tuple(targets)
    )
    extracted = runtime.probe_pixels(raw, probe) if pixels else runtime.probe_cells(raw, probe)
    result_type = PixelProbeResult if pixels else CellProbeResult
    result = result_type.model_validate_json(json.dumps(extracted["result"]))
    report = {"probe_executed": True, "requested_cells": len(targets)}
    if (
        result.file_sha256 != probe.file_sha256
        or result.page_count != probe.page_count
        or result.request_fingerprint != fingerprint(probe.model_dump(mode="json"))
        or [(s.region_id, s.page) for s in result.samples] != [(t.region_id, t.page) for t in targets]
    ):
        return {**report, "status": "probe_binding_mismatch"}
    cells, counts = [], dict(agree=0, different=0, uncheckable=0)
    for sample in result.samples:
        status = "uncheckable"
        if sample.status == "observed":
            status = "agree" if sample.text == expected[sample.region_id] else "different"
        counts[status] += 1
        cells.append(
            {
                "region_id": sample.region_id,
                "page": sample.page,
                "status": status,
                "probe_status": sample.status,
                **({"min_confidence": sample.min_confidence} if pixels else {}),
            }
        )
    image_id = extracted.get("runtime_image_id")
    return {
        **report,
        "status": "disagreement"
        if counts["different"]
        else "inconclusive"
        if counts["uncheckable"]
        else "all_cells_agree",
        "engine": result.engine,
        "probe_fingerprint": result.fingerprint,
        "runtime_image_id": image_id
        if isinstance(image_id, str) and re.fullmatch(r"sha256:[a-f0-9]{64}", image_id)
        else None,
        "counts": counts,
        "cells": cells,
        **(
            {
                key: getattr(result, key)
                for key in ("raster_pages", "rendered_pixels", "ocr_calls", "renderer", "profile")
            }
            if pixels
            else {}
        ),
    }
