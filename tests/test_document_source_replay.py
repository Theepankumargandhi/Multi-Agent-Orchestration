"""Source-byte replay never confuses a consistent snapshot with its original extraction."""

import base64
import io
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing

import pytest
from test_document_claim_audits import advance, item, report, seed, start
from test_document_workbench_reconciliation import BASE, create, fact, sources
from test_document_workbench_reconciliation import report as set_report
from test_document_workbench_reconciliation import setup as setup
from test_document_workbench_versions import client_for, headers, upload

from agentforge_sdk import load_claim_replay_bundle, verify_claim_replay_bundle
from agentforge_sdk.models import fingerprint
from agentforge_sdk.spatial import MultimodalDocument, Region, region_fingerprint
from service.document_runtime import DocumentRuntime
from service.document_store import DocumentLibrary


def prepare(client):
    set_id, count, calc = seed(client)
    audit_id = start(
        client, set_id, [item(count), item(calc, "growth", "25", "%"), item(count, "wrong", "999")]
    )
    advance(client, set_id, audit_id)
    advance(client, set_id, audit_id)
    ids = [s["id"] for s in set_report(client, set_id).json()["sources"]]
    return set_id, audit_id, ids


def endpoint(set_id, audit_id, source_id):
    return f"{BASE}/{set_id}/audits/{audit_id}/sources/{source_id}/replay-original"


def body(client, set_id, audit_id, **overrides):
    return {
        "checkpoint_fingerprint": report(client, set_id, audit_id).json()["checkpoint_fingerprint"],
        "confirm_reextract": True,
        "allow_original_ocr": False,
        **overrides,
    }


def replay(client, set_id, audit_id, source_id, **overrides):
    return client.post(
        endpoint(set_id, audit_id, source_id),
        headers=headers(),
        json=body(client, set_id, audit_id, **overrides),
    )


class RuntimeControl:
    def __init__(self, delegate, mode="same"):
        self.delegate, self.mode, self.calls = delegate, mode, 0
        self.entered, self.release = threading.Event(), threading.Event()

    def extract(self, raw, **meta):
        self.calls += 1
        if self.mode == "wait":
            self.entered.set()
            assert self.release.wait(8), "authored runtime was not released"
        if self.mode == "fail":
            raise RuntimeError("PRIVATE_RUNTIME_ERROR")
        result = self.delegate.extract(raw, **meta)
        doc = result["manifest"]
        if self.mode == "profile":
            doc["extractor"] = "changed-profile-control"
        elif self.mode == "binding":
            doc["source"] = "wrong-source.pdf"
        elif self.mode == "held":
            doc["pages"][0].update(status="held", method="none", reason="empty_page", regions=[])
        if self.mode in {"profile", "binding", "held"}:
            doc["fingerprint"] = fingerprint({k: v for k, v in doc.items() if k != "fingerprint"})
        return result


def stored(library):
    with closing(library.connect()) as db:
        return {
            table: [tuple(r) for r in db.execute(f"SELECT * FROM {table} ORDER BY 1")]
            for table in (
                "documents",
                "document_reviews",
                "evidence_set_results",
                "evidence_set_reviews",
                "evidence_claim_audits",
            )
        }


def assert_unleased(library):
    with closing(library.connect()) as db:
        assert db.execute("SELECT COUNT(*) FROM document_runtime_lease").fetchone()[0] == 0


def test_reextracts_original_and_reproduces_refutations_without_rewriting_any_source(setup):
    client, library = setup
    set_id, audit_id, ids = prepare(client)
    original = stored(library)
    library.runtime = RuntimeControl(library.runtime)
    for source_id in ids:
        response = replay(client, set_id, audit_id, source_id)
        assert response.status_code == 200, response.text
        result = response.json()
        assert response.headers["cache-control"] == "no-store"
        assert result["status"] == "exact_snapshot_match"
        assert result["original_bytes_match"] and result["snapshot_matches"] and result["profile_matches"]
        assert result["extraction_reexecuted"] and result["numeric_checks_reproduced"] == 3
        assert result["numeric_counts"] == dict(supported=2, refuted=1, unresolved=0)
        assert not result["publisher_identity_verified"] and not result["release_authorized"]
        assert result["model_calls"] == 0 and result["runtime_image_id"] is None
        assert result["fingerprint"] == fingerprint({k: v for k, v in result.items() if k != "fingerprint"})
        assert "manifest" not in result and "previews" not in result and "original" not in result
    assert library.runtime.calls == 2 and stored(library) == original
    assert_unleased(library)


def forge_current_values(library, ids, value="999"):
    with closing(library.connect()) as db, db:
        for source_id in ids:
            doc = json.loads(
                db.execute("SELECT manifest FROM documents WHERE id=?", (source_id,)).fetchone()[0]
            )
            for page in doc["pages"]:
                for cell in page["regions"]:
                    if cell.get("row") == 1 and cell.get("column") == 2:
                        cell["text"] = value
                        cell["region_id"] = region_fingerprint(
                            doc["file_sha256"], page["page"], Region(**cell)
                        )
            doc["fingerprint"] = fingerprint({k: v for k, v in doc.items() if k != "fingerprint"})
            MultimodalDocument.model_validate_json(json.dumps(doc))
            db.execute("UPDATE documents SET manifest=? WHERE id=?", (json.dumps(doc), source_id))


def forged_audit(client, library, ids):
    forge_current_values(library, ids)
    set_id = create(client, ids, "Coherently changed snapshot control")
    current = fact(client, set_id).json()["id"]
    audit_id = start(client, set_id, [item(current, value="999")])
    advance(client, set_id, audit_id)
    result = client.post(
        f"{BASE}/{set_id}/audits/{audit_id}/replay-bundle",
        headers=headers(),
        json={
            "checkpoint_fingerprint": report(client, set_id, audit_id).json()["checkpoint_fingerprint"],
            "include_extraction_text": True,
        },
    )
    assert result.status_code == 200, result.text
    assert verify_claim_replay_bundle(
        load_claim_replay_bundle(result.content)
    ).all_numeric_assertions_supported
    return set_id, audit_id


def test_consistently_rewritten_snapshot_passes_plain_replay_but_not_original_reextraction(setup):
    client, library = setup
    ids = sources(client)
    set_id, audit_id = forged_audit(client, library, ids)
    original = stored(library)
    result = replay(client, set_id, audit_id, ids[0]).json()
    assert (
        result["status"] == "snapshot_changed"
        and result["original_bytes_match"]
        and result["profile_matches"]
    )
    assert result["extraction_reexecuted"] and not result["snapshot_matches"]
    assert result["numeric_checks_reproduced"] == 0 and result["numeric_counts"] is None
    assert stored(library) == original
    assert_unleased(library)


@pytest.mark.parametrize(
    "mode,status",
    [
        ("fail", "extraction_failed"),
        ("profile", "profile_changed"),
        ("binding", "runtime_binding_mismatch"),
        ("held", "extraction_held"),
    ],
)
def test_failures_profile_drift_and_holds_do_not_become_matches_and_always_release(setup, mode, status):
    client, library = setup
    set_id, audit_id, ids = prepare(client)
    original = stored(library)
    library.runtime = RuntimeControl(library.runtime, mode)
    response = replay(client, set_id, audit_id, ids[0])
    assert response.status_code == 200, response.text
    assert response.json()["status"] == status and not response.json()["snapshot_matches"]
    assert "PRIVATE_RUNTIME_ERROR" not in response.text and response.json()["numeric_checks_reproduced"] == 0
    assert stored(library) == original
    assert_unleased(library)


def test_changed_original_bytes_never_invoke_the_parser(setup):
    client, library = setup
    set_id, audit_id, ids = prepare(client)
    library.runtime = RuntimeControl(library.runtime)
    with closing(library.connect()) as db, db:
        db.execute("UPDATE documents SET original=? WHERE id=?", (b"%PDF-changed", ids[0]))
    result = replay(client, set_id, audit_id, ids[0]).json()
    assert result["status"] == "original_bytes_mismatch" and not result["extraction_reexecuted"]
    assert not result["original_bytes_match"] and library.runtime.calls == 0
    assert_unleased(library)


@pytest.mark.parametrize("raw", [b"", b"x" * 3_000_001], ids=["empty_original", "oversized_original"])
def test_original_byte_budget_is_checked_in_storage_before_reextraction(setup, raw):
    client, library = setup
    set_id, audit_id, ids = prepare(client)
    library.runtime = RuntimeControl(library.runtime)
    with closing(library.connect()) as db, db:
        db.execute("UPDATE documents SET original=? WHERE id=?", (raw, ids[0]))
    result = replay(client, set_id, audit_id, ids[0]).json()
    assert result["status"] == "original_outside_budget" and library.runtime.calls == 0
    assert not result["extraction_reexecuted"]
    assert_unleased(library)


def test_owner_scope_partial_stale_and_explicit_ocr_confirmation(setup):
    client, library = setup
    set_id, audit_id, ids = prepare(client)
    request = body(client, set_id, audit_id)
    path = endpoint(set_id, audit_id, ids[0])
    assert client.post(path, headers=headers("bob"), json=request).status_code == 404
    assert replay(client, set_id, audit_id, "f" * 32).status_code == 404
    assert replay(client, set_id, audit_id, ids[0], checkpoint_fingerprint="0" * 64).status_code == 409
    partial = start(client, set_id, [item(set_report(client, set_id).json()["results"][0]["id"])], "Partial")
    assert replay(client, set_id, partial, ids[0]).status_code == 409
    library.runtime = RuntimeControl(library.runtime)
    with closing(library.connect()) as db, db:
        db.execute("UPDATE documents SET ocr=1 WHERE id=?", (ids[0],))
    assert replay(client, set_id, audit_id, ids[0]).status_code == 409
    assert library.runtime.calls == 0
    assert replay(client, set_id, audit_id, ids[0], allow_original_ocr=True).status_code == 200


@pytest.mark.parametrize(
    "extra",
    [
        {"confirm_reextract": 1},
        {"confirm_reextract": False},
        {"allow_original_ocr": 1},
        {"image": "PRIVATE_IMAGE"},
        {"command": "PRIVATE_COMMAND"},
    ],
)
def test_strict_private_requests_do_not_accept_tools_or_implicit_consent(setup, extra):
    client, _ = setup
    set_id, audit_id, ids = prepare(client)
    response = replay(client, set_id, audit_id, ids[0], **extra)
    assert response.status_code == 422 and "PRIVATE_" not in response.text


def test_shared_lease_blocks_other_replays_and_new_uploads_but_not_reads(setup):
    client, library = setup
    set_id, audit_id, ids = prepare(client)
    runtime = library.runtime = RuntimeControl(library.runtime, "wait")
    request = body(client, set_id, audit_id)
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(client.post, endpoint(set_id, audit_id, ids[0]), headers=headers(), json=request)
        try:
            assert runtime.entered.wait(5)
            assert (
                client.post(endpoint(set_id, audit_id, ids[1]), headers=headers(), json=request).status_code
                == 429
            )
            assert upload(client, b"%PDF-new", name="new.pdf").status_code == 429
            assert report(client, set_id, audit_id).status_code == 200
        finally:
            runtime.release.set()
        response = future.result()
        assert response.status_code == 200 and response.json()["status"] == "exact_snapshot_match"
    assert_unleased(library)


def test_processing_upload_blocks_replay_and_expired_lease_can_recover(setup):
    client, library = setup
    set_id, audit_id, ids = prepare(client)
    with closing(library.connect()) as db, db:
        db.execute("UPDATE documents SET status='processing',created=? WHERE id=?", (time.time(), ids[1]))
    assert replay(client, set_id, audit_id, ids[0]).status_code == 429
    with closing(library.connect()) as db, db:
        db.execute("UPDATE documents SET status='ready' WHERE id=?", (ids[1],))
        db.execute("INSERT INTO document_runtime_lease VALUES (1,?,?)", ("expired", time.time() - 301))
    assert replay(client, set_id, audit_id, ids[0]).status_code == 200
    assert_unleased(library)


@pytest.mark.parametrize("change", ["original", "delete_audit", "replace_lease"])
def test_context_changes_during_execution_cannot_report_success_or_release_another_lease(setup, change):
    client, library = setup
    set_id, audit_id, ids = prepare(client)
    runtime = library.runtime = RuntimeControl(library.runtime, "wait")
    request = body(client, set_id, audit_id)
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(client.post, endpoint(set_id, audit_id, ids[0]), headers=headers(), json=request)
        try:
            assert runtime.entered.wait(5)
            with closing(library.connect()) as db, db:
                if change == "original":
                    db.execute("UPDATE documents SET original=? WHERE id=?", (b"%PDF-modified", ids[0]))
                elif change == "delete_audit":
                    db.execute("DELETE FROM evidence_claim_audits WHERE id=?", (audit_id,))
                else:
                    db.execute(
                        "UPDATE document_runtime_lease SET token=? WHERE slot=1", ("another-operation",)
                    )
        finally:
            runtime.release.set()
        assert future.result().status_code == (404 if change == "delete_audit" else 409)
    if change == "replace_lease":
        with closing(library.connect()) as db:
            assert db.execute("SELECT token FROM document_runtime_lease").fetchone()[0] == "another-operation"
    else:
        assert_unleased(library)


@pytest.mark.skipif(
    os.getenv("AF_RUN_DOCUMENT_DOCKER") != "1", reason="explicit isolated original-PDF replay opt-in"
)
def test_actual_pdf_source_replay_matches_native_pages_and_detects_coherent_snapshot_rewrite(
    tmp_path, monkeypatch
):
    from reportlab.pdfgen import canvas

    monkeypatch.setenv("WORKBENCH_DOCUMENTS_ENABLED", "true")
    library = DocumentLibrary(tmp_path / "original-replay.db", DocumentRuntime())
    with client_for(library) as client:
        ids = []
        for index in range(2):
            output = io.BytesIO()
            pdf = canvas.Canvas(output, pagesize=(600, 800))
            if index:
                pdf.drawString(60, 720, "Atlas cover; source metrics follow.")
                pdf.showPage()
            pdf.drawString(60, 720, f"Atlas source {index}: requests in raw counts.")
            pdf.grid([60, 240, 360, 480], [440, 480, 520, 560])
            for row, values in enumerate(
                [["Metric", "Previous", "Current"], ["Requests", "100", "125"], ["Errors", "10", "20"]]
            ):
                for col, text in enumerate(values):
                    pdf.drawString([70, 250, 370][col], 535 - row * 40, text)
            pdf.save()
            (tmp_path / f"original-{index}.pdf").write_bytes(output.getvalue())
            uploaded = upload(client, output.getvalue(), name=f"original-{index}.pdf").json()
            assert uploaded["status"] == "ready", uploaded
            ids.append(uploaded["id"])
        set_id = create(client, ids)
        current = fact(client, set_id).json()["id"]
        audit_id = start(client, set_id, [item(current), item(current, "wrong", "999")])
        advance(client, set_id, audit_id)
        original = stored(library)
        reports = []
        for source_id in ids:
            response = replay(client, set_id, audit_id, source_id)
            assert response.status_code == 200, response.text
            result = response.json()
            assert result["status"] == "exact_snapshot_match" and result["numeric_checks_reproduced"] == 2
            assert result["numeric_counts"] == dict(supported=1, refuted=1, unresolved=0)
            assert result["runtime_image_id"].startswith("sha256:")
            reports.append(result)
        assert stored(library) == original
        forged_set, forged_id = forged_audit(client, library, ids)
        original_forged = stored(library)
        changed = replay(client, forged_set, forged_id, ids[1]).json()
        assert changed["status"] == "snapshot_changed" and changed["original_bytes_match"]
        assert changed["profile_matches"] and changed["numeric_checks_reproduced"] == 0
        assert stored(library) == original_forged
        reports.append(changed)
        (tmp_path / "source-replay-reports.json").write_text(json.dumps(reports, indent=2), encoding="utf-8")
        image = client.get(f"/workbench/documents/{ids[1]}/pages/2", headers=headers()).json()["image"]
        (tmp_path / "original-page-two.png").write_bytes(base64.b64decode(image.split(",")[1]))
        assert_unleased(library)
