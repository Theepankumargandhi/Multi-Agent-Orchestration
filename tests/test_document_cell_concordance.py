"""Cross-engine citation checks never upgrade numerical verdicts or owner authority."""

import base64
import hashlib
import io
import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing

import pytest
from test_document_claim_audits import advance, item, report, start
from test_document_source_replay import assert_unleased, body, forged_audit, prepare, stored
from test_document_workbench_reconciliation import create, fact
from test_document_workbench_reconciliation import setup as setup
from test_document_workbench_versions import client_for, headers, upload

from agentforge_sdk.models import fingerprint
from agentforge_sdk.spatial import Region, region_fingerprint
from service.document_cell_contracts import CellProbe, CellSample
from service.document_runtime import DocumentRuntime
from service.document_store import DocumentLibrary


def endpoint(set_id, audit_id, source_id):
    return f"/workbench/documents/reconciliations/{set_id}/audits/{audit_id}/sources/{source_id}/crosscheck-cells"


def crosscheck(client, set_id, audit_id, source_id, **extra):
    request = body(client, set_id, audit_id)
    return client.post(
        endpoint(set_id, audit_id, source_id),
        headers=headers(),
        json={
            "checkpoint_fingerprint": request["checkpoint_fingerprint"],
            "confirm_crosscheck": True,
            **extra,
        },
    )


class ProbeControl:
    def __init__(self, library, mode="same"):
        self.library, self.delegate, self.mode = library, library.runtime, mode
        self.calls, self.requests = 0, []
        self.entered, self.release = threading.Event(), threading.Event()

    def extract(self, *args, **kwargs):
        return self.delegate.extract(*args, **kwargs)

    def probe_cells(self, raw, probe):
        self.calls += 1
        self.requests.append(probe)
        if self.mode == "wait":
            self.entered.set()
            assert self.release.wait(8)
        if self.mode == "fail":
            raise RuntimeError("PRIVATE_ENGINE_ERROR")
        with closing(self.library.connect()) as db:
            document = json.loads(
                db.execute(
                    "SELECT manifest FROM documents WHERE digest=? LIMIT 1", (probe.file_sha256,)
                ).fetchone()[0]
            )
        regions = {r["region_id"]: r["text"] for page in document["pages"] for r in page["regions"]}
        samples = []
        for target in probe.targets:
            text = regions[target.region_id]
            if self.mode == "different" and text == "125":
                text = "1250"  # A page substring search would incorrectly accept this.
            status = (
                self.mode if self.mode in {"empty", "unsupported_geometry", "budget_exceeded"} else "observed"
            )
            samples.append(
                CellSample(
                    region_id=target.region_id,
                    page=target.page,
                    status=status,
                    text=text if status == "observed" else None,
                ).model_dump(mode="json")
            )
        payload = dict(
            protocol="agentforge-pdf-cell-samples-v1",
            file_sha256=probe.file_sha256,
            request_fingerprint=fingerprint(probe.model_dump(mode="json")),
            page_count=probe.page_count,
            engine="pypdfium2:5.14.0",
            samples=samples,
        )
        if self.mode == "wrong_hash":
            payload["file_sha256"] = "0" * 64
        if self.mode == "wrong_request":
            payload["request_fingerprint"] = "0" * 64
        if self.mode == "wrong_page":
            payload["samples"][0]["page"] = 10
        if self.mode == "missing":
            payload["samples"].pop()
        return {
            "result": {**payload, "fingerprint": fingerprint(payload)},
            "runtime_image_id": "sha256:" + "a" * 64,
        }


def test_blind_request_covers_headers_and_values_without_rewriting_or_disclosing_text(setup):
    client, library = setup
    set_id, audit_id, ids = prepare(client)
    original = stored(library)
    runtime = library.runtime = ProbeControl(library)
    result = crosscheck(client, set_id, audit_id, ids[0])
    assert result.status_code == 200 and result.headers["cache-control"] == "no-store"
    value = result.json()
    assert value["status"] == "all_cells_agree" and value["requested_cells"] == 5
    assert value["counts"] == dict(agree=5, different=0, uncheckable=0)
    assert value["probe_executed"] and value["original_bytes_match"]
    assert not value["numeric_verdicts_rechecked"] and not value["other_sources_checked"]
    assert (
        not value["publisher_identity_verified"]
        and not value["release_authorized"]
        and value["model_calls"] == 0
    )
    assert value["fingerprint"] == fingerprint({k: v for k, v in value.items() if k != "fingerprint"})
    assert "Requests" not in result.text and '"text"' not in result.text and '"original"' not in result.text
    probe = runtime.requests[0]
    assert '"text"' not in probe.model_dump_json() and "expected" not in probe.model_dump_json()
    assert len({t.region_id for t in probe.targets}) == 5
    assert stored(library) == original and runtime.calls == 1
    assert report(client, set_id, audit_id).json()["counts"] == dict(supported=2, refuted=1, unresolved=0)
    assert_unleased(library)


@pytest.mark.parametrize(
    "mode,status",
    [
        ("different", "disagreement"),
        ("empty", "inconclusive"),
        ("unsupported_geometry", "inconclusive"),
        ("budget_exceeded", "inconclusive"),
        ("wrong_hash", "probe_binding_mismatch"),
        ("wrong_request", "probe_binding_mismatch"),
        ("wrong_page", "probe_binding_mismatch"),
        ("missing", "probe_binding_mismatch"),
        ("fail", "extraction_failed"),
    ],
)
def test_disagreement_holds_bad_bindings_and_failures_are_not_matches(setup, mode, status):
    client, library = setup
    set_id, audit_id, ids = prepare(client)
    original = stored(library)
    library.runtime = ProbeControl(library, mode)
    response = crosscheck(client, set_id, audit_id, ids[0])
    assert response.status_code == 200 and response.json()["status"] == status
    assert "PRIVATE_ENGINE_ERROR" not in response.text and not response.json()["release_authorized"]
    assert stored(library) == original
    assert_unleased(library)


@pytest.mark.parametrize(
    "extra",
    [
        {"confirm_crosscheck": 1},
        {"confirm_crosscheck": False},
        {"allow_original_ocr": True},
        {"targets": "PRIVATE_TARGETS"},
        {"command": "PRIVATE_COMMAND"},
    ],
)
def test_strict_confirmation_and_no_caller_tools_or_coordinates(setup, extra):
    client, library = setup
    set_id, audit_id, ids = prepare(client)
    runtime = library.runtime = ProbeControl(library)
    response = crosscheck(client, set_id, audit_id, ids[0], **extra)
    assert response.status_code == 422 and "PRIVATE_" not in response.text and runtime.calls == 0


def test_owner_membership_partial_stale_and_disabled_checks_precede_execution(setup, monkeypatch):
    client, library = setup
    set_id, audit_id, ids = prepare(client)
    runtime = library.runtime = ProbeControl(library)
    path = endpoint(set_id, audit_id, ids[0])
    request = dict(
        checkpoint_fingerprint=body(client, set_id, audit_id)["checkpoint_fingerprint"],
        confirm_crosscheck=True,
    )
    assert client.post(path, headers=headers("bob"), json=request).status_code == 404
    assert crosscheck(client, set_id, audit_id, "f" * 32).status_code == 404
    assert crosscheck(client, set_id, audit_id, ids[0], checkpoint_fingerprint="0" * 64).status_code == 409
    partial = start(
        client,
        set_id,
        [item(report(client, set_id, audit_id).json()["items"][0]["result_id"])],
        "Partial cell audit",
    )
    assert crosscheck(client, set_id, partial, ids[0]).status_code == 409
    monkeypatch.setenv("WORKBENCH_DOCUMENTS_ENABLED", "false")
    assert client.post(path, headers=headers(), json=request).status_code == 404
    assert runtime.calls == 0


@pytest.mark.parametrize(
    "raw", [b"%PDF-changed", b"", b"x" * 3_000_001], ids=["changed", "empty", "oversize"]
)
def test_original_hash_and_storage_budget_preflight_never_run_probe(setup, raw):
    client, library = setup
    set_id, audit_id, ids = prepare(client)
    runtime = library.runtime = ProbeControl(library)
    with closing(library.connect()) as db, db:
        db.execute("UPDATE documents SET original=? WHERE id=?", (raw, ids[0]))
    result = crosscheck(client, set_id, audit_id, ids[0]).json()
    assert result["status"] == (
        "original_bytes_mismatch" if raw == b"%PDF-changed" else "original_outside_budget"
    )
    assert not result["probe_executed"] and runtime.calls == 0
    assert_unleased(library)


def test_crosscheck_shares_runtime_slot_with_upload_and_same_parser_replay(setup):
    client, library = setup
    set_id, audit_id, ids = prepare(client)
    runtime = library.runtime = ProbeControl(library, "wait")
    request = dict(
        checkpoint_fingerprint=body(client, set_id, audit_id)["checkpoint_fingerprint"],
        confirm_crosscheck=True,
    )
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(client.post, endpoint(set_id, audit_id, ids[0]), headers=headers(), json=request)
        try:
            assert runtime.entered.wait(5)
            assert crosscheck(client, set_id, audit_id, ids[1]).status_code == 429
            from test_document_source_replay import replay

            assert replay(client, set_id, audit_id, ids[1]).status_code == 429
            assert upload(client, b"%PDF-new", name="new.pdf").status_code == 429
            assert report(client, set_id, audit_id).status_code == 200
        finally:
            runtime.release.set()
        assert future.result().json()["status"] == "all_cells_agree"
    assert_unleased(library)


@pytest.mark.parametrize("change", ["original", "delete_audit", "lease"])
def test_late_context_changes_cannot_return_success_or_release_another_lease(setup, change):
    client, library = setup
    set_id, audit_id, ids = prepare(client)
    runtime = library.runtime = ProbeControl(library, "wait")
    request = dict(
        checkpoint_fingerprint=body(client, set_id, audit_id)["checkpoint_fingerprint"],
        confirm_crosscheck=True,
    )
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(client.post, endpoint(set_id, audit_id, ids[0]), headers=headers(), json=request)
        try:
            assert runtime.entered.wait(5)
            with closing(library.connect()) as db, db:
                if change == "original":
                    db.execute("UPDATE documents SET original=? WHERE id=?", (b"%PDF-late", ids[0]))
                elif change == "delete_audit":
                    db.execute("DELETE FROM evidence_claim_audits WHERE id=?", (audit_id,))
                else:
                    db.execute("UPDATE document_runtime_lease SET token='replacement' WHERE slot=1")
        finally:
            runtime.release.set()
        assert future.result().status_code == (404 if change == "delete_audit" else 409)
    if change == "lease":
        with closing(library.connect()) as db:
            assert db.execute("SELECT token FROM document_runtime_lease").fetchone()[0] == "replacement"
    else:
        assert_unleased(library)


@pytest.mark.skipif(
    os.getenv("AF_RUN_DOCUMENT_DOCKER") != "1", reason="explicit isolated PDFium integration opt-in"
)
def test_actual_pdfium_headers_values_page_two_and_wrong_localization(tmp_path, monkeypatch):
    from reportlab.pdfgen import canvas

    monkeypatch.setenv("WORKBENCH_DOCUMENTS_ENABLED", "true")
    runtime = DocumentRuntime()
    library = DocumentLibrary(tmp_path / "cells.db", runtime)
    with client_for(library) as client:
        ids, reports = [], []
        for index in range(2):
            output = io.BytesIO()
            pdf = canvas.Canvas(output, pagesize=(600, 800))
            if index:
                pdf.drawString(60, 720, "Atlas cover: table follows.")
                pdf.showPage()
            pdf.drawString(60, 720, f"Atlas source {index}. Elsewhere reference: 125.")
            pdf.grid([60, 240, 360, 480], [440, 480, 520, 560])
            for row, values in enumerate(
                [["Metric", "Previous", "Current"], ["Requests", "100", "125"], ["Errors", "10", "20"]]
            ):
                for column, text in enumerate(values):
                    pdf.drawString([70, 250, 370][column], 535 - row * 40, text)
            pdf.save()
            (tmp_path / f"concordance-{index}.pdf").write_bytes(output.getvalue())
            result = upload(client, output.getvalue(), name=f"concordance-{index}.pdf").json()
            assert result["status"] == "ready", result
            ids.append(result["id"])
        set_id = create(client, ids)
        current = fact(client, set_id).json()["id"]
        audit_id = start(client, set_id, [item(current), item(current, "wrong", "999")])
        advance(client, set_id, audit_id)
        original = stored(library)
        for source_id in ids:
            response = crosscheck(client, set_id, audit_id, source_id)
            assert response.status_code == 200, response.text
            result = response.json()
            assert result["status"] == "all_cells_agree" and result["counts"]["agree"] == 3, result
            assert result["engine"].startswith("pypdfium2:")
            reports.append(result)
        assert {c["page"] for c in reports[1]["cells"]} == {2}
        assert stored(library) == original
        forged_set, forged_id = forged_audit(client, library, ids)
        unchanged = stored(library)
        changed = crosscheck(client, forged_set, forged_id, ids[1]).json()
        assert changed["status"] == "disagreement" and changed["counts"] == dict(
            agree=2, different=1, uncheckable=0
        )
        assert stored(library) == unchanged
        reports.append(changed)
        # Preserve the (forged) text but move its cell to Previous=100. 125 still exists elsewhere on the page.
        with closing(library.connect()) as db, db:
            for source_id in ids:
                doc = json.loads(
                    db.execute("SELECT manifest FROM documents WHERE id=?", (source_id,)).fetchone()[0]
                )
                for page in doc["pages"]:
                    values = {r["column"]: r for r in page["regions"] if r.get("row") == 1}
                    if 2 in values:
                        values[2]["text"] = "125"
                        values[1]["box"], values[2]["box"] = values[2]["box"], values[1]["box"]
                        for cell in values.values():
                            cell["region_id"] = region_fingerprint(
                                doc["file_sha256"], page["page"], Region(**cell)
                            )
                doc["fingerprint"] = fingerprint({k: v for k, v in doc.items() if k != "fingerprint"})
                db.execute("UPDATE documents SET manifest=? WHERE id=?", (json.dumps(doc), source_id))
        misplaced_set = create(client, ids, "Wrong localization control")
        current = fact(client, misplaced_set).json()["id"]
        misplaced_id = start(client, misplaced_set, [item(current)])
        advance(client, misplaced_set, misplaced_id)
        assert report(client, misplaced_set, misplaced_id).json()["counts"]["supported"] == 1
        misplaced = crosscheck(client, misplaced_set, misplaced_id, ids[1]).json()
        assert misplaced["status"] == "disagreement" and misplaced["counts"]["different"] == 1
        reports.append(misplaced)
        (tmp_path / "cell-concordance-reports.json").write_text(
            json.dumps(reports, indent=2), encoding="utf-8"
        )
        image = client.get(f"/workbench/documents/{ids[1]}/pages/2", headers=headers()).json()["image"]
        (tmp_path / "concordance-page-two.png").write_bytes(base64.b64decode(image.split(",")[1]))
        assert_unleased(library)
    rotated = io.BytesIO()
    pdf = canvas.Canvas(rotated, pagesize=(600, 800))
    pdf.setPageRotation(90)
    pdf.drawString(60, 500, "125")
    pdf.save()
    (tmp_path / "concordance-rotated.pdf").write_bytes(rotated.getvalue())
    from agentforge_sdk.spatial import Box
    from service.document_cell_contracts import CellTarget

    probe = CellProbe(
        file_sha256=hashlib.sha256(rotated.getvalue()).hexdigest(),
        page_count=1,
        targets=(
            CellTarget(
                region_id="a" * 64, page=1, width=800.0, height=600.0, box=Box(x0=0.0, y0=0.0, x1=1.0, y1=1.0)
            ),
        ),
    )
    result = runtime.probe_cells(rotated.getvalue(), probe)["result"]
    assert result["samples"][0]["status"] == "unsupported_geometry"
