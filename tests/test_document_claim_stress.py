"""Source omission is a diagnostic, never a correction to the real audit."""

import base64
import io
import json
import os
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing

import pytest
from test_document_claim_audits import advance as audit_advance
from test_document_claim_audits import item, start
from test_document_claim_audits import report as audit_report
from test_document_workbench_reconciliation import BASE, calculate, create, fact, sources
from test_document_workbench_reconciliation import report as set_report
from test_document_workbench_reconciliation import setup as setup
from test_document_workbench_versions import client_for, headers, upload

from service.document_claim_stress import DocumentClaimStress
from service.document_runtime import DocumentRuntime
from service.document_store import DocumentLibrary


def prepare(client, contents=(b"%PDF-a", b"%PDF-b", b"%PDF-c")):
    ids = sources(client, contents)
    set_id = create(client, ids)
    before, after = [fact(client, set_id, c).json()["id"] for c in ("Previous", "Current")]
    growth = calculate(client, set_id, [before, after]).json()["id"]
    audit_id = start(client, set_id, [item(after), item(growth, "growth", "25", "%")])
    assert audit_advance(client, set_id, audit_id).status_code == 200
    return set_id, audit_id, ids


def base(set_id, audit_id):
    return f"{BASE}/{set_id}/audits/{audit_id}/stress-tests"


def body(client, set_id, audit_id, name="Evidence fragility"):
    return {
        "name": name,
        "source_checkpoint_fingerprint": audit_report(client, set_id, audit_id).json()[
            "checkpoint_fingerprint"
        ],
    }


def create_stress(client, set_id, audit_id, name="Evidence fragility"):
    response = client.post(
        base(set_id, audit_id), headers=headers(), json=body(client, set_id, audit_id, name)
    )
    assert response.status_code == 201, response.text
    return response.json()["id"]


def report(client, set_id, audit_id, run_id):
    return client.get(f"{base(set_id, audit_id)}/{run_id}", headers=headers())


def advance(client, set_id, audit_id, run_id, expected=None):
    return client.post(
        f"{base(set_id, audit_id)}/{run_id}/advance",
        headers=headers(),
        json={
            "checkpoint_fingerprint": expected
            or report(client, set_id, audit_id, run_id).json()["checkpoint_fingerprint"]
        },
    )


def finish(client, set_id, audit_id, run_id):
    while not report(client, set_id, audit_id, run_id).json()["complete"]:
        response = advance(client, set_id, audit_id, run_id)
        assert response.status_code == 200, response.text
    return report(client, set_id, audit_id, run_id).json()


def test_three_source_support_resumes_without_mutating_audits_or_saved_results(setup):
    client, library = setup
    set_id, audit_id, ids = prepare(client)
    baseline = audit_report(client, set_id, audit_id).json()
    source_results = set_report(client, set_id).json()["results"]
    run_id = create_stress(client, set_id, audit_id)
    assert create_stress(client, set_id, audit_id) == run_id
    initial = report(client, set_id, audit_id, run_id).json()
    assert initial["processed"] == 0 and initial["evaluated_checks"] == 0 and not initial["complete"]
    assert all(c["stable_numeric_support"] is None for c in initial["claims"])
    assert advance(client, set_id, audit_id, run_id, initial["checkpoint_fingerprint"]).status_code == 200
    partial = report(client, set_id, audit_id, run_id).json()
    assert partial["processed"] == 1 and partial["evaluated_checks"] == 2 and not partial["complete"]
    assert advance(client, set_id, audit_id, run_id, initial["checkpoint_fingerprint"]).status_code == 409
    with client_for(DocumentLibrary(library.path, library.runtime)) as reopened:
        final = finish(reopened, set_id, audit_id, run_id)
    assert final["complete"] and final["processed"] == 3 and final["evaluated_checks"] == 6
    assert final["support_lost_cases"] == final["support_appeared_cases"] == 0
    assert all(c["stable_numeric_support"] is True for c in final["claims"])
    for claim in final["claims"]:
        assert {v["omitted_source_id"] for v in claim["variants"]} == set(ids)
        for variant in claim["variants"]:
            assert variant["omitted_source_id"] not in {o["source_id"] for o in variant["observations"]}
    assert audit_report(client, set_id, audit_id).json() == baseline
    assert set_report(client, set_id).json()["results"] == source_results
    assert advance(client, set_id, audit_id, run_id).status_code == 409
    assert final["model_calls"] == 0 and final["diagnostic_only"] is True
    assert "all_claims_supported" not in final


def test_two_source_omissions_keep_original_two_artifact_requirement(setup):
    client, _ = setup
    set_id, audit_id, _ = prepare(client, (b"%PDF-a", b"%PDF-b"))
    result = finish(client, set_id, audit_id, create_stress(client, set_id, audit_id))
    assert result["support_lost_cases"] == 4
    assert result["policy"]["min_distinct_artifacts"] == 2 and result["policy"]["require_complete_coverage"]
    assert all(c["stable_numeric_support"] is False for c in result["claims"])
    assert all(
        v["check"]["status"] == "unresolved" and v["check"]["observed_value"] is None
        for c in result["claims"]
        for v in c["variants"]
    )


def test_dropping_dissent_exposes_selective_evidence_risk_not_conflict_resolution(setup):
    client, _ = setup
    set_id, audit_id, ids = prepare(client, (b"%PDF-a", b"%PDF-b", b"%PDF-v2"))
    baseline = audit_report(client, set_id, audit_id).json()
    result = finish(client, set_id, audit_id, create_stress(client, set_id, audit_id))
    assert result["support_appeared_cases"] == 2 and result["support_lost_cases"] == 0
    for claim in result["claims"]:
        assert claim["baseline"]["status"] == "unresolved" and claim["stable_numeric_support"] is None
        variant = next(v for v in claim["variants"] if v["omitted_source_id"] == ids[2])
        assert (
            variant["check"]["status"] == "supported"
            and variant["effect"] == "support_appeared_after_omission"
        )
    assert audit_report(client, set_id, audit_id).json() == baseline


def test_duplicate_raw_artifacts_do_not_inflate_stability(setup):
    client, _ = setup
    set_id, audit_id, ids = prepare(client, (b"%PDF-same", b"%PDF-same", b"%PDF-other"))
    result = finish(client, set_id, audit_id, create_stress(client, set_id, audit_id))
    assert result["support_lost_cases"] == 2
    assert all(
        next(v for v in c["variants"] if v["omitted_source_id"] == ids[2])["check"]["status"] == "unresolved"
        for c in result["claims"]
    )


@pytest.mark.parametrize(
    "extra",
    [
        {"excluded_sources": ["pick-a-winner"]},
        {"min_distinct_artifacts": 1},
        {"require_complete_coverage": False},
        {"max_steps": 5},
    ],
)
def test_clients_cannot_cherry_pick_scenarios_or_weaken_policy(setup, extra):
    client, _ = setup
    set_id, audit_id, _ = prepare(client)
    response = client.post(
        base(set_id, audit_id), headers=headers(), json={**body(client, set_id, audit_id), **extra}
    )
    assert response.status_code == 422 and "pick-a-winner" not in response.text
    assert response.headers["cache-control"] == "no-store"


def test_owner_isolation_partial_sources_and_stale_source_bindings(setup):
    client, _ = setup
    set_id, audit_id, _ = prepare(client)
    run_id = create_stress(client, set_id, audit_id)
    for method, path, payload in [
        ("GET", base(set_id, audit_id), None),
        ("GET", f"{base(set_id, audit_id)}/{run_id}", None),
        ("POST", base(set_id, audit_id), body(client, set_id, audit_id)),
        ("POST", f"{base(set_id, audit_id)}/{run_id}/advance", {"checkpoint_fingerprint": "0" * 64}),
        ("DELETE", f"{base(set_id, audit_id)}/{run_id}", None),
    ]:
        response = client.request(method, path, headers=headers("bob"), json=payload)
        assert response.status_code == 404 and response.headers["cache-control"] == "no-store"
    other = create(client, [s["id"] for s in set_report(client, set_id).json()["sources"]], "Other set")
    assert client.get(f"{base(other, audit_id)}/{run_id}", headers=headers()).status_code == 404
    stale = body(client, set_id, audit_id)
    stale["source_checkpoint_fingerprint"] = "0" * 64
    assert client.post(base(set_id, audit_id), headers=headers(), json=stale).status_code == 409
    unchecked = start(client, set_id, [item(fact(client, set_id).json()["id"])], "Unchecked source")
    assert (
        client.post(
            base(set_id, unchecked), headers=headers(), json=body(client, set_id, unchecked)
        ).status_code
        == 409
    )


def test_concurrent_advancement_and_checkpoint_budget_are_atomic(setup, monkeypatch):
    client, _ = setup
    set_id, audit_id, _ = prepare(client)
    run_id = create_stress(client, set_id, audit_id)
    initial = report(client, set_id, audit_id, run_id).json()
    with monkeypatch.context() as patch:
        patch.setattr(DocumentClaimStress, "max_checkpoint_bytes", 1)
        assert advance(client, set_id, audit_id, run_id, initial["checkpoint_fingerprint"]).status_code == 413
    assert (
        report(client, set_id, audit_id, run_id).json()["checkpoint_fingerprint"]
        == initial["checkpoint_fingerprint"]
    )
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(
            pool.map(
                lambda _: advance(client, set_id, audit_id, run_id, initial["checkpoint_fingerprint"]),
                range(2),
            )
        )
    assert sorted(r.status_code for r in results) == [200, 409]
    assert report(client, set_id, audit_id, run_id).json()["processed"] == 1


def test_resealed_forged_scenario_is_recomputed_and_rejected(setup):
    client, library = setup
    set_id, audit_id, _ = prepare(client)
    run_id = create_stress(client, set_id, audit_id)
    advance(client, set_id, audit_id, run_id)
    with closing(library.connect()) as db, db:
        row = db.execute("SELECT * FROM evidence_claim_stress WHERE id=?", (run_id,)).fetchone()
        scenarios = json.loads(row["scenarios"])
        scenarios[0]["checks"][0]["status"] = "refuted"
        db.execute(
            "UPDATE evidence_claim_stress SET scenarios=?,checkpoint_fingerprint=? WHERE id=?",
            (json.dumps(scenarios), DocumentClaimStress._checkpoint(row["digest"], scenarios), run_id),
        )
    assert report(client, set_id, audit_id, run_id).status_code == 409
    assert client.delete(f"{base(set_id, audit_id)}/{run_id}", headers=headers()).status_code == 200
    assert audit_report(client, set_id, audit_id).json()["complete"]


def test_stress_testing_does_not_consume_saved_result_capacity(setup):
    client, _ = setup
    set_id, audit_id, _ = prepare(client)
    for n in range(17):
        assert fact(client, set_id, row=f"Not present {n}").status_code == 200
    assert len(set_report(client, set_id).json()["results"]) == 20
    result = finish(client, set_id, audit_id, create_stress(client, set_id, audit_id))
    assert result["complete"] and len(set_report(client, set_id).json()["results"]) == 20


def test_report_quota_dedup_and_source_audit_cascade(setup):
    client, library = setup
    set_id, audit_id, _ = prepare(client)
    first = create_stress(client, set_id, audit_id, "Run 0")
    for n in range(1, 20):
        create_stress(client, set_id, audit_id, f"Run {n}")
    assert create_stress(client, set_id, audit_id, "Run 0") == first
    assert (
        client.post(
            base(set_id, audit_id), headers=headers(), json=body(client, set_id, audit_id, "Over quota")
        ).status_code
        == 429
    )
    assert client.delete(f"{BASE}/{set_id}/audits/{audit_id}", headers=headers()).status_code == 200
    with closing(library.connect()) as db:
        assert db.execute("SELECT COUNT(*) FROM evidence_claim_stress").fetchone()[0] == 0
        assert db.execute("SELECT COUNT(*) FROM documents").fetchone()[0] == 3
        assert db.execute("SELECT COUNT(*) FROM evidence_set_results").fetchone()[0] == 3


@pytest.mark.skipif(
    os.getenv("AF_RUN_DOCUMENT_DOCKER") != "1", reason="explicit isolated local PDF integration opt-in"
)
def test_real_pdf_stress_exposes_conflict_masking_and_keeps_original_evidence(tmp_path, monkeypatch):
    from reportlab.pdfgen import canvas

    class CheckedRuntime(DocumentRuntime):
        error = ""

        def extract(self, *args, **kwargs):
            try:
                return super().extract(*args, **kwargs)
            except Exception as exc:
                self.error = str(exc)
                raise

    monkeypatch.setenv("WORKBENCH_DOCUMENTS_ENABLED", "true")
    runtime = CheckedRuntime()
    library = DocumentLibrary(tmp_path / "real-stress.db", runtime)
    with client_for(library) as client:
        ids = []
        for index, value in enumerate(("125", "125", "125", "150")):
            output = io.BytesIO()
            pdf = canvas.Canvas(output, pagesize=(600, 800))
            if index == 3:
                pdf.drawString(60, 720, "Atlas separate report; metrics follow.")
                pdf.showPage()
            pdf.drawString(60, 720, f"Atlas source {index}: 2026-Q1 requests in raw counts.")
            pdf.grid([60, 240, 360, 480], [440, 480, 520, 560])
            for row, values in enumerate(
                [["Metric", "Previous", "Current"], ["Requests", "100", value], ["Errors", "10", "20"]]
            ):
                for column, text in enumerate(values):
                    pdf.drawString([70, 250, 370][column], 535 - row * 40, text)
            pdf.save()
            raw = output.getvalue()
            (tmp_path / f"source-{index}.pdf").write_bytes(raw)
            doc = upload(client, raw, name=f"source-{index}.pdf").json()
            assert doc["status"] == "ready", (doc, runtime.error)
            ids.append(doc["id"])
        reports = []
        for label, selected in [("Agreement", ids[:3]), ("Conflict", [ids[0], ids[1], ids[3]])]:
            set_id = create(client, selected, label)
            before, after = [fact(client, set_id, c).json()["id"] for c in ("Previous", "Current")]
            growth = calculate(client, set_id, [before, after]).json()["id"]
            audit_id = start(
                client, set_id, [item(after), item(growth, "growth", "25", "%")], label + " audit"
            )
            audit_advance(client, set_id, audit_id)
            baseline = audit_report(client, set_id, audit_id).json()
            original_results = set_report(client, set_id).json()["results"]
            result = finish(
                client, set_id, audit_id, create_stress(client, set_id, audit_id, label + " stress")
            )
            assert audit_report(client, set_id, audit_id).json() == baseline
            assert set_report(client, set_id).json()["results"] == original_results
            assert len(original_results) == 3
            if label == "Agreement":
                assert all(c["stable_numeric_support"] for c in result["claims"])
            else:
                assert result["support_appeared_cases"] == 2
                assert all(c["baseline"]["status"] == "unresolved" for c in result["claims"])
                assert any(
                    c["page"] == 2
                    for claim in result["claims"]
                    for variant in claim["variants"]
                    for observation in variant["observations"]
                    if observation["source_id"] == ids[3]
                    for c in observation["citations"]
                )
            reports.append(result)
        (tmp_path / "stress-reports.json").write_text(json.dumps(reports, indent=2), encoding="utf-8")
        image = client.get(f"/workbench/documents/{ids[3]}/pages/2", headers=headers()).json()["image"]
        (tmp_path / "dissenting-source-preview.png").write_bytes(base64.b64decode(image.split(",")[1]))
