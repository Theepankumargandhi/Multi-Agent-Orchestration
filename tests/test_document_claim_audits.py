"""Resumable audit API, source bindings, private export and real PDF controls."""

import base64
import io
import json
import os
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing

import pytest
from test_document_workbench_reconciliation import (
    BASE,
    calculate,
    create,
    fact,
    sources,
)
from test_document_workbench_reconciliation import (
    report as set_report,
)
from test_document_workbench_reconciliation import setup as setup
from test_document_workbench_versions import client_for, headers, upload

from agentforge_sdk.models import fingerprint
from service.document_claim_audits import DocumentClaimAudits
from service.document_runtime import DocumentRuntime
from service.document_store import DocumentLibrary


def item(result_id, claim_id="count", value="125", unit="scalar"):
    return {
        "result_id": result_id,
        "claim": {
            "claim_id": claim_id,
            "label": f"Claim {claim_id}",
            "expected_value": value,
            "expected_unit": unit,
        },
    }


def seed(client, *, conflicting=False):
    set_id = create(client, sources(client, (b"%PDF-a", b"%PDF-v2" if conflicting else b"%PDF-b")))
    before = fact(client, set_id, "Previous").json()["id"]
    after = fact(client, set_id).json()["id"]
    calc = calculate(client, set_id, [before, after]).json()["id"]
    return set_id, after, calc


def body(client, set_id, claims, name="Quarterly claims"):
    return {
        "name": name,
        "collection_fingerprint": set_report(client, set_id).json()["collection_fingerprint"],
        "claims": claims,
    }


def start(client, set_id, claims, name="Quarterly claims"):
    response = client.post(
        f"{BASE}/{set_id}/audits", headers=headers(), json=body(client, set_id, claims, name)
    )
    assert response.status_code == 201, response.text
    return response.json()["id"]


def report(client, set_id, audit_id):
    return client.get(f"{BASE}/{set_id}/audits/{audit_id}", headers=headers())


def advance(client, set_id, audit_id, expected=None):
    return client.post(
        f"{BASE}/{set_id}/audits/{audit_id}/advance",
        headers=headers(),
        json={
            "checkpoint_fingerprint": expected
            or report(client, set_id, audit_id).json()["checkpoint_fingerprint"]
        },
    )


def test_frozen_order_explicit_steps_restart_and_partial_reports(setup):
    client, library = setup
    set_id, count, calc = seed(client)
    claims = [
        item(count),
        item(count, "wrong", "999"),
        item(calc, "growth", "25", "%"),
        item(count, "currency", "125", "USD"),
    ]
    audit_id = start(client, set_id, claims)
    assert start(client, set_id, claims) == audit_id
    initial = report(client, set_id, audit_id).json()
    assert initial["processed"] == 0 and not initial["all_claims_supported"]
    assert all(i["check"] is None for i in initial["items"])
    assert advance(client, set_id, audit_id, initial["checkpoint_fingerprint"]).status_code == 200
    partial = report(client, set_id, audit_id).json()
    assert partial["processed"] == 2 and not partial["complete"] and not partial["all_claims_supported"]
    assert partial["counts"] == {"supported": 1, "refuted": 1, "unresolved": 0}
    assert advance(client, set_id, audit_id, initial["checkpoint_fingerprint"]).status_code == 409
    reopened = DocumentLibrary(library.path, library.runtime)
    with client_for(reopened) as restart:
        assert advance(restart, set_id, audit_id).status_code == 200
        final = report(restart, set_id, audit_id).json()
    assert final["complete"] and final["processed"] == 4 and not final["all_claims_supported"]
    assert final["counts"] == {"supported": 2, "refuted": 1, "unresolved": 1}
    assert final["items"][-1]["check"]["reason"] == "unit_mismatch"
    assert advance(client, set_id, audit_id).status_code == 409
    assert final["model_calls"] == 0


def test_supported_summary_requires_every_claim_processed_and_supported(setup):
    client, _ = setup
    set_id, count, calc = seed(client)
    audit_id = start(
        client, set_id, [item(count), item(calc, "growth", "25.0", "%"), item(count, "same", "+125")]
    )
    advance(client, set_id, audit_id)
    assert not report(client, set_id, audit_id).json()["all_claims_supported"]
    advance(client, set_id, audit_id)
    assert report(client, set_id, audit_id).json()["all_claims_supported"]


def test_reviewing_a_conflict_does_not_make_a_claim_supported(setup):
    client, _ = setup
    set_id, count, _ = seed(client, conflicting=True)
    audit_id = start(client, set_id, [item(count)])
    saved = next(r for r in set_report(client, set_id).json()["results"] if r["id"] == count)
    assert (
        client.post(
            f"{BASE}/{set_id}/results/{count}/reviews",
            headers=headers(),
            json={"digest": saved["payload"]["digest"], "decision": "inspected", "note": "Still conflicting"},
        ).status_code
        == 200
    )
    advance(client, set_id, audit_id)
    check = report(client, set_id, audit_id).json()["items"][0]["check"]
    assert check["status"] == "unresolved" and check["observed_value"] is None


def test_owner_set_and_export_boundaries_and_cascading_deletion(setup):
    client, library = setup
    set_id, count, _ = seed(client)
    audit_id = start(client, set_id, [item(count)])
    for method, suffix, payload in [
        ("GET", "", None),
        ("GET", "/export", None),
        ("POST", "/advance", {"checkpoint_fingerprint": "a" * 64}),
        ("DELETE", "", None),
    ]:
        response = client.request(
            method, f"{BASE}/{set_id}/audits/{audit_id}{suffix}", headers=headers("bob"), json=payload
        )
        assert response.status_code == 404 and response.headers["cache-control"] == "no-store"
    assert client.get(f"{BASE}/{set_id}/audits", headers=headers("bob")).status_code == 404
    other = create(client, [s["id"] for s in set_report(client, set_id).json()["sources"]], "Other scope")
    assert (
        client.post(
            f"{BASE}/{other}/audits", headers=headers(), json=body(client, other, [item(count)])
        ).status_code
        == 404
    )
    assert client.get(f"{BASE}/{other}/audits/{audit_id}", headers=headers()).status_code == 404
    exported = client.get(f"{BASE}/{set_id}/audits/{audit_id}/export", headers=headers()).json()
    assert not exported["complete"] and not exported["all_claims_supported"]
    assert set(exported["evidence"]) == {count}
    assert exported["report_fingerprint"] == fingerprint(
        {k: v for k, v in exported.items() if k != "report_fingerprint"}
    )
    assert "original" not in exported and "manifest" not in exported and "signature" not in exported
    client.delete(f"{BASE}/{set_id}", headers=headers())
    with closing(library.connect()) as db:
        assert db.execute("SELECT COUNT(*) FROM evidence_claim_audits").fetchone()[0] == 0
        assert db.execute("SELECT COUNT(*) FROM documents").fetchone()[0] == 2


@pytest.mark.parametrize("change", ["duplicate", "over_limit", "extra", "float", "exponent", "stale"])
def test_strict_bounded_claim_requests_and_private_errors(setup, change):
    client, _ = setup
    set_id, count, _ = seed(client)
    request = body(client, set_id, [item(count)])
    if change == "duplicate":
        request["claims"] *= 2
    elif change == "over_limit":
        request["claims"] = [item(count, f"c{i}") for i in range(11)]
    elif change == "extra":
        request["claims"][0]["claim"]["tolerance"] = 0.1
    elif change == "float":
        request["claims"][0]["claim"]["expected_value"] = 125.0
    elif change == "exponent":
        request["claims"][0]["claim"]["expected_value"] = "PRIVATE_SENTINEL_1e3"
    else:
        request["collection_fingerprint"] = "0" * 64
    response = client.post(f"{BASE}/{set_id}/audits", headers=headers(), json=request)
    assert response.status_code == (409 if change == "stale" else 422), response.text
    assert "PRIVATE_SENTINEL" not in response.text


@pytest.mark.parametrize("target", ["checkpoint", "check", "parent"])
def test_tampered_state_never_becomes_a_valid_audit(setup, target):
    client, library = setup
    set_id, count, _ = seed(client)
    audit_id = start(client, set_id, [item(count)])
    advance(client, set_id, audit_id)
    with closing(library.connect()) as db, db:
        row = db.execute("SELECT * FROM evidence_claim_audits WHERE id=?", (audit_id,)).fetchone()
        if target == "checkpoint":
            db.execute(
                "UPDATE evidence_claim_audits SET checkpoint_fingerprint=? WHERE id=?", ("0" * 64, audit_id)
            )
        elif target == "check":
            checks = json.loads(row["checks"])
            checks[0]["status"] = "refuted"
            checks[0]["fingerprint"] = fingerprint({k: v for k, v in checks[0].items() if k != "fingerprint"})
            db.execute(
                "UPDATE evidence_claim_audits SET checks=?,checkpoint_fingerprint=? WHERE id=?",
                (json.dumps(checks), DocumentClaimAudits._checkpoint(row["digest"], checks), audit_id),
            )
        else:
            parent = json.loads(
                db.execute("SELECT payload FROM evidence_set_results WHERE id=?", (count,)).fetchone()[0]
            )
            parent["request"]["extra"] = "changed"
            parent["digest"] = fingerprint({k: v for k, v in parent.items() if k != "digest"})
            db.execute("UPDATE evidence_set_results SET payload=? WHERE id=?", (json.dumps(parent), count))
    assert report(client, set_id, audit_id).status_code == 409
    assert client.delete(f"{BASE}/{set_id}/audits/{audit_id}", headers=headers()).status_code == 200


def test_concurrent_step_exactly_one_commit_and_budget_rollback(setup, monkeypatch):
    client, _ = setup
    set_id, count, _ = seed(client)
    audit_id = start(client, set_id, [item(count, f"c{i}") for i in range(3)])
    expected = report(client, set_id, audit_id).json()["checkpoint_fingerprint"]
    with ThreadPoolExecutor(max_workers=2) as pool:
        responses = list(pool.map(lambda _: advance(client, set_id, audit_id, expected), range(2)))
    assert sorted(r.status_code for r in responses) == [200, 409]
    before = report(client, set_id, audit_id).json()
    with monkeypatch.context() as patch:
        patch.setattr(DocumentClaimAudits, "max_checkpoint_bytes", 1)
        assert advance(client, set_id, audit_id, before["checkpoint_fingerprint"]).status_code == 413
    after = report(client, set_id, audit_id).json()
    assert (
        after["processed"] == before["processed"]
        and after["checkpoint_fingerprint"] == before["checkpoint_fingerprint"]
    )


def test_owner_quota_dedup_at_capacity_and_no_orphaned_results(setup):
    client, library = setup
    set_id, count, _ = seed(client)
    first = start(client, set_id, [item(count)], "Audit 0")
    for index in range(1, 20):
        start(client, set_id, [item(count)], f"Audit {index}")
    assert start(client, set_id, [item(count)], "Audit 0") == first
    assert (
        client.post(
            f"{BASE}/{set_id}/audits",
            headers=headers(),
            json=body(client, set_id, [item(count)], "Over quota"),
        ).status_code
        == 429
    )
    with closing(library.connect()) as db:
        assert db.execute("SELECT COUNT(*) FROM evidence_claim_audits").fetchone()[0] == 20


@pytest.mark.skipif(
    os.getenv("AF_RUN_DOCUMENT_DOCKER") != "1", reason="explicit isolated local PDF integration opt-in"
)
def test_real_pdf_numeric_claim_audit_citations_partial_export_and_resume(tmp_path, monkeypatch):
    from reportlab.pdfgen import canvas

    monkeypatch.setenv("WORKBENCH_DOCUMENTS_ENABLED", "true")
    library = DocumentLibrary(tmp_path / "real-claims.db", DocumentRuntime())
    with client_for(library) as client:
        ids = []
        for index in range(2):
            output = io.BytesIO()
            pdf = canvas.Canvas(output, pagesize=(600, 800))
            if index:
                pdf.drawString(60, 720, "Atlas report cover; metrics follow.")
                pdf.showPage()
            pdf.drawString(60, 720, f"Atlas report {index}: 2026-Q1 requests in raw counts.")
            pdf.grid([60, 240, 360, 480], [440, 480, 520, 560])
            for row, values in enumerate(
                [["Metric", "Previous", "Current"], ["Requests", "100", "125"], ["Errors", "10", "20"]]
            ):
                for column, text in enumerate(values):
                    pdf.drawString([70, 250, 370][column], 535 - row * 40, text)
            pdf.save()
            raw = output.getvalue()
            (tmp_path / f"report-{index}.pdf").write_bytes(raw)
            doc = upload(client, raw, name=f"report-{index}.pdf").json()
            assert doc["status"] == "ready", doc
            ids.append(doc["id"])
        set_id = create(client, ids)
        before, after = [fact(client, set_id, column).json()["id"] for column in ("Previous", "Current")]
        growth = calculate(client, set_id, [before, after]).json()["id"]
        audit_id = start(
            client,
            set_id,
            [item(after), item(growth, "growth", "25", "%"), item(growth, "wrong-growth", "50", "%")],
        )
        advance(client, set_id, audit_id)
        assert not report(client, set_id, audit_id).json()["complete"]
        with client_for(DocumentLibrary(library.path, library.runtime)) as reopened:
            assert advance(reopened, set_id, audit_id).status_code == 200
            exported = reopened.get(f"{BASE}/{set_id}/audits/{audit_id}/export", headers=headers()).json()
        assert exported["counts"] == {"supported": 2, "refuted": 1, "unresolved": 0}
        assert exported["complete"] and not exported["all_claims_supported"] and exported["model_calls"] == 0
        assert any(c["page"] == 2 for o in exported["items"][0]["observations"] for c in o["citations"])
        from agentforge_sdk import load_claim_replay_bundle, verify_claim_replay_bundle

        baseline = report(client, set_id, audit_id).json()
        replay = client.post(
            f"{BASE}/{set_id}/audits/{audit_id}/replay-bundle",
            headers=headers(),
            json={
                "checkpoint_fingerprint": baseline["checkpoint_fingerprint"],
                "include_extraction_text": True,
            },
        )
        assert replay.status_code == 200, replay.text
        bundle = load_claim_replay_bundle(replay.content)
        verified = verify_claim_replay_bundle(bundle, expected_fingerprint=bundle.fingerprint)
        assert verified.supported == 2 and verified.refuted == 1 and verified.total == 3
        assert report(client, set_id, audit_id).json() == baseline
        assert any(len(s.document.pages) == 2 for s in bundle.sources)
        (tmp_path / "claim-replay-bundle.json").write_bytes(replay.content)
        (tmp_path / "claim-audit-report.json").write_text(json.dumps(exported, indent=2), encoding="utf-8")
        image = client.get(f"/workbench/documents/{ids[1]}/pages/2", headers=headers()).json()["image"]
        (tmp_path / "claim-source-preview.png").write_bytes(base64.b64decode(image.split(",")[1]))
