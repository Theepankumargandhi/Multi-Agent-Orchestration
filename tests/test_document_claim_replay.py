"""Fresh evidence replay preserves assertions and never rewrites the source audit."""

import base64
import io
import json
import os
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from types import SimpleNamespace

import pytest
from test_document_claim_audits import advance as audit_advance
from test_document_claim_audits import item, seed, start
from test_document_claim_audits import report as audit_report
from test_document_workbench_reconciliation import BASE, binding, create, fact, sources
from test_document_workbench_reconciliation import report as set_report
from test_document_workbench_reconciliation import setup as setup
from test_document_workbench_versions import client_for, headers, upload

from agentforge_sdk.claim_audit import NumericClaim, check_numeric_claim
from service.document_claim_replay import DocumentClaimReplay, compare_checks
from service.document_runtime import DocumentRuntime
from service.document_store import DocumentLibrary


def prepared(client, contents=(b"%PDF-v2-a", b"%PDF-v2-b")):
    old, count, growth = seed(client)
    errors = fact(client, old, row="Errors").json()["id"]
    audit_id = start(
        client, old, [item(count), item(growth, "growth", "25", "%"), item(errors, "errors", "20")]
    )
    while not audit_report(client, old, audit_id).json()["complete"]:
        assert audit_advance(client, old, audit_id).status_code == 200
    new = create(client, sources(client, contents), "Updated evidence")
    return old, audit_id, new


def request(client, old, audit_id, new, name="Recheck the same assertions"):
    return {
        "name": name,
        "target_set_id": new,
        "source_checkpoint_fingerprint": audit_report(client, old, audit_id).json()["checkpoint_fingerprint"],
        "target_collection_fingerprint": set_report(client, new).json()["collection_fingerprint"],
        "scope_confirmed": True,
    }


def base(old, audit_id):
    return f"{BASE}/{old}/audits/{audit_id}/replays"


def create_replay(client, old, audit_id, new, name="Recheck the same assertions"):
    response = client.post(
        base(old, audit_id), headers=headers(), json=request(client, old, audit_id, new, name)
    )
    assert response.status_code == 201, response.text
    return response.json()["id"]


def report(client, old, audit_id, replay_id):
    return client.get(f"{base(old, audit_id)}/{replay_id}", headers=headers())


def advance(client, old, audit_id, replay_id, expected=None):
    return client.post(
        f"{base(old, audit_id)}/{replay_id}/advance",
        headers=headers(),
        json={
            "checkpoint_fingerprint": expected
            or report(client, old, audit_id, replay_id).json()["checkpoint_fingerprint"]
        },
    )


def test_changed_report_replays_identical_assertions_and_preserves_old_checks(setup):
    client, library = setup
    old, audit_id, new = prepared(client)
    original = audit_report(client, old, audit_id).json()
    replay_id = create_replay(client, old, audit_id, new)
    assert create_replay(client, old, audit_id, new) == replay_id
    assert not set_report(client, new).json()["results"]
    first = report(client, old, audit_id, replay_id).json()
    assert first["processed"] == 0 and not first["complete"] and not first["all_claims_supported"]
    assert all(i["after"] is None for i in first["items"])
    assert advance(client, old, audit_id, replay_id, first["checkpoint_fingerprint"]).status_code == 200
    partial = report(client, old, audit_id, replay_id).json()
    assert partial["processed"] == 2 and not partial["complete"] and partial["support_lost"] == 2
    assert [i["after"]["observed_value"] for i in partial["items"][:2]] == ["150", "50"]
    assert len(set_report(client, new).json()["results"]) == 3
    assert advance(client, old, audit_id, replay_id, first["checkpoint_fingerprint"]).status_code == 409
    with client_for(DocumentLibrary(library.path, library.runtime)) as reopened:
        assert advance(reopened, old, audit_id, replay_id).status_code == 200
        final = report(reopened, old, audit_id, replay_id).json()
    assert final["complete"] and final["processed"] == 3 and not final["all_claims_supported"]
    assert final["items"][2]["comparison"] == {
        "transition": "unchanged_verdict",
        "observed_value_changed": False,
    }
    assert all(i["claim"] == i["before"]["claim"] == i["after"]["claim"] for i in final["items"])
    assert audit_report(client, old, audit_id).json() == original
    assert all(r["review"] is None for r in set_report(client, new).json()["results"])
    assert advance(client, old, audit_id, replay_id).status_code == 409 and final["model_calls"] == 0


@pytest.mark.parametrize(
    "contents",
    [(b"%PDF-a-new", b"%PDF-v2-b"), (b"%PDF-missing-a", b"%PDF-missing-b"), (b"%PDF-held-a", b"%PDF-held-b")],
)
def test_conflict_missing_and_held_targets_lose_support_without_releasing_values(setup, contents):
    client, _ = setup
    old, audit_id, new = prepared(client, contents)
    replay_id = create_replay(client, old, audit_id, new)
    assert advance(client, old, audit_id, replay_id).status_code == 200
    result = report(client, old, audit_id, replay_id).json()["items"][0]
    assert result["after"]["status"] == "unresolved" and result["after"]["observed_value"] is None
    assert result["comparison"] == {"transition": "support_lost", "observed_value_changed": None}


def test_new_support_is_a_relative_verdict_not_a_quality_or_review_override(setup):
    client, _ = setup
    old, count, _ = seed(client, conflicting=True)
    audit_id = start(client, old, [item(count)])
    audit_advance(client, old, audit_id)
    new = create(client, sources(client, (b"%PDF-clean-a", b"%PDF-clean-b")), "Agreeing reports")
    replay_id = create_replay(client, old, audit_id, new)
    advance(client, old, audit_id, replay_id)
    result = report(client, old, audit_id, replay_id).json()
    assert result["all_claims_supported"] and result["items"][0]["comparison"]["transition"] == "new_support"


@pytest.mark.parametrize(
    "before,after,transition",
    [
        ("supported", "supported", "unchanged_verdict"),
        ("supported", "refuted", "support_lost"),
        ("supported", "unresolved", "support_lost"),
        ("refuted", "supported", "new_support"),
        ("refuted", "refuted", "unchanged_verdict"),
        ("refuted", "unresolved", "became_unresolved"),
        ("unresolved", "supported", "new_support"),
        ("unresolved", "refuted", "new_refutation"),
        ("unresolved", "unresolved", "unchanged_verdict"),
    ],
)
def test_internal_transition_matrix_does_not_conflate_refutation_with_missing_evidence(
    before, after, transition
):
    # Only classification logic is authored here; API tests exercise actual receipt reverification.
    left = SimpleNamespace(
        claim="same frozen assertion", status=before, observed_value=None, observed_unit=None
    )
    right = SimpleNamespace(claim=left.claim, status=after, observed_value=None, observed_unit=None)
    assert compare_checks(left, right) == {"transition": transition, "observed_value_changed": None}


def test_replay_preserves_noncommutative_operand_order(setup):
    from test_document_workbench_reconciliation import calculate

    client, _ = setup
    old, count, _ = seed(client)
    before = fact(client, old, "Previous").json()["id"]
    difference = calculate(client, old, [before, count], "difference").json()["id"]
    audit_id = start(client, old, [item(difference, "difference", "-25")])
    audit_advance(client, old, audit_id)
    new = create(
        client, sources(client, (b"%PDF-v2-difference-a", b"%PDF-v2-difference-b")), "Changed difference"
    )
    replay_id = create_replay(client, old, audit_id, new)
    assert advance(client, old, audit_id, replay_id).status_code == 200
    check = report(client, old, audit_id, replay_id).json()["items"][0]["after"]
    assert check["observed_value"] == "-50" and check["status"] == "refuted"


@pytest.mark.parametrize("change", ["scope", "source", "target", "false", "integer", "extra", "same"])
def test_explicit_comparability_and_frozen_context_checks(setup, change):
    client, _ = setup
    old, audit_id, new = prepared(client)
    payload = request(client, old, audit_id, new)
    if change == "scope":
        new_binding = binding(client, [s["id"] for s in set_report(client, new).json()["sources"]])
        new_binding["scope"]["period"] = "2026-Q2"
        different = client.post(BASE, headers=headers(), json=new_binding).json()["id"]
        payload = request(client, old, audit_id, different)
    elif change == "source":
        payload["source_checkpoint_fingerprint"] = "0" * 64
    elif change == "target":
        payload["target_collection_fingerprint"] = "0" * 64
    elif change in {"false", "integer"}:
        payload["scope_confirmed"] = False if change == "false" else 1
    elif change == "same":
        payload = request(client, old, audit_id, old)
    else:
        payload["claim_overrides"] = "PRIVATE_SENTINEL"
    response = client.post(base(old, audit_id), headers=headers(), json=payload)
    assert response.status_code == (409 if change in {"scope", "source", "target"} else 422), response.text
    assert "PRIVATE_SENTINEL" not in response.text and response.headers["cache-control"] == "no-store"


def test_partial_source_and_foreign_owners_or_audits_cannot_replay(setup):
    client, _ = setup
    old, audit_id, new = prepared(client)
    payload = request(client, old, audit_id, new)
    assert client.post(base(old, audit_id), headers=headers("bob"), json=payload).status_code == 404
    partial = start(client, old, [item(fact(client, old).json()["id"])], "Unchecked source")
    assert (
        client.post(
            base(old, partial),
            headers=headers(),
            json={
                **payload,
                "source_checkpoint_fingerprint": audit_report(client, old, partial).json()[
                    "checkpoint_fingerprint"
                ],
            },
        ).status_code
        == 409
    )
    replay_id = create_replay(client, old, audit_id, new)
    assert client.get(f"{base(new, audit_id)}/{replay_id}", headers=headers()).status_code == 404
    for method, suffix, body in [
        ("GET", "", None),
        ("POST", "/advance", {"checkpoint_fingerprint": "0" * 64}),
        ("DELETE", "", None),
    ]:
        response = client.request(
            method, f"{base(old, audit_id)}/{replay_id}{suffix}", headers=headers("bob"), json=body
        )
        assert response.status_code == 404


def test_checkpoint_budget_and_target_result_quota_roll_back_whole_step(setup, monkeypatch):
    client, library = setup
    old, audit_id, new = prepared(client)
    replay_id = create_replay(client, old, audit_id, new)
    checkpoint = report(client, old, audit_id, replay_id).json()["checkpoint_fingerprint"]
    with monkeypatch.context() as patch:
        patch.setattr(DocumentClaimReplay, "max_checkpoint_bytes", 1)
        assert advance(client, old, audit_id, replay_id, checkpoint).status_code == 413
    assert not set_report(client, new).json()["results"]
    assert report(client, old, audit_id, replay_id).json()["checkpoint_fingerprint"] == checkpoint
    for index in range(19):
        assert fact(client, new, row=f"Missing {index}").status_code == 200
    assert advance(client, old, audit_id, replay_id, checkpoint).status_code == 429
    with closing(library.connect()) as db:
        assert (
            db.execute("SELECT COUNT(*) FROM evidence_set_results WHERE set_id=?", (new,)).fetchone()[0] == 19
        )
    assert report(client, old, audit_id, replay_id).json()["processed"] == 0


def test_concurrent_advances_commit_only_one_step(setup):
    client, _ = setup
    old, audit_id, new = prepared(client)
    replay_id = create_replay(client, old, audit_id, new)
    checkpoint = report(client, old, audit_id, replay_id).json()["checkpoint_fingerprint"]
    with ThreadPoolExecutor(max_workers=2) as pool:
        responses = list(pool.map(lambda _: advance(client, old, audit_id, replay_id, checkpoint), range(2)))
    assert sorted(r.status_code for r in responses) == [200, 409]
    assert report(client, old, audit_id, replay_id).json()["processed"] == 2


def test_relinking_to_a_different_valid_fact_cannot_forge_the_replay_recipe(setup):
    client, library = setup
    old, audit_id, new = prepared(client)
    replay_id = create_replay(client, old, audit_id, new)
    advance(client, old, audit_id, replay_id)
    wrong_id = fact(client, new, row="Errors").json()["id"]
    controller = DocumentClaimReplay(library)
    with closing(library.connect()) as db, db:
        row = db.execute("SELECT * FROM evidence_claim_replays WHERE id=?", (replay_id,)).fetchone()
        entries = json.loads(row["entries"])
        _, _, evidence, _, _ = controller.sets._load(db, "alice", new)
        payload, receipt, parents = controller.audits._input(db, "alice", new, wrong_id, evidence)
        spec = NumericClaim(**entries[0]["check"]["claim"])
        entries[0] = {
            "result_id": wrong_id,
            "digest": payload["digest"],
            "check": check_numeric_claim(evidence, spec, receipt, facts=parents).model_dump(mode="json"),
        }
        db.execute(
            "UPDATE evidence_claim_replays SET entries=?,checkpoint_fingerprint=? WHERE id=?",
            (json.dumps(entries), controller._checkpoint(row["digest"], entries), replay_id),
        )
    assert report(client, old, audit_id, replay_id).status_code == 409
    assert client.delete(f"{base(old, audit_id)}/{replay_id}", headers=headers()).status_code == 200


@pytest.mark.parametrize("deleted", ["audit", "target"])
def test_deletion_cascades_reports_but_does_not_delete_unrelated_evidence(setup, deleted):
    client, library = setup
    old, audit_id, new = prepared(client)
    replay_id = create_replay(client, old, audit_id, new)
    advance(client, old, audit_id, replay_id)
    path = f"{BASE}/{old}/audits/{audit_id}" if deleted == "audit" else f"{BASE}/{new}"
    assert client.delete(path, headers=headers()).status_code == 200
    with closing(library.connect()) as db:
        assert db.execute("SELECT COUNT(*) FROM evidence_claim_replays").fetchone()[0] == 0
        assert db.execute("SELECT COUNT(*) FROM documents").fetchone()[0] == 4
    if deleted == "audit":
        assert len(set_report(client, new).json()["results"]) == 3
    else:
        assert audit_report(client, old, audit_id).json()["complete"]


def test_replay_quota_allows_idempotent_retries_at_capacity(setup):
    client, _ = setup
    old, audit_id, new = prepared(client)
    original = create_replay(client, old, audit_id, new, "Run 0")
    for index in range(1, 20):
        create_replay(client, old, audit_id, new, f"Run {index}")
    assert create_replay(client, old, audit_id, new, "Run 0") == original
    assert (
        client.post(
            base(old, audit_id), headers=headers(), json=request(client, old, audit_id, new, "Over quota")
        ).status_code
        == 429
    )


@pytest.mark.skipif(
    os.getenv("AF_RUN_DOCUMENT_DOCKER") != "1", reason="explicit isolated PDF integration opt-in"
)
def test_real_pdf_revision_replays_growth_and_keeps_page_two_lineage(tmp_path, monkeypatch):
    from reportlab.pdfgen import canvas
    from test_document_workbench_reconciliation import calculate

    monkeypatch.setenv("WORKBENCH_DOCUMENTS_ENABLED", "true")

    class CheckedRuntime(DocumentRuntime):
        error = ""

        def extract(self, *args, **kwargs):
            try:
                return super().extract(*args, **kwargs)
            except Exception as exc:
                self.error = str(exc)
                raise

    runtime = CheckedRuntime()
    library = DocumentLibrary(tmp_path / "real-replay.db", runtime)
    with client_for(library) as client:
        old_ids, new_ids = [], []
        for revision in range(2):
            for index in range(2):
                output = io.BytesIO()
                pdf = canvas.Canvas(output, pagesize=(600, 800))
                if revision:
                    pdf.drawString(60, 720, "Atlas corrected report; metrics follow.")
                    pdf.showPage()
                pdf.drawString(60, 720, f"Atlas report {index}, revision {revision}: 2026-Q1 raw counts.")
                pdf.grid([60, 240, 360, 480], [440, 480, 520, 560])
                for row, values in enumerate(
                    [
                        ["Metric", "Previous", "Current"],
                        ["Requests", "100", "150" if revision else "125"],
                        ["Errors", "10", "20"],
                    ]
                ):
                    for column, text in enumerate(values):
                        pdf.drawString([70, 250, 370][column], 535 - row * 40, text)
                pdf.save()
                raw = output.getvalue()
                (tmp_path / f"report-{index}-v{revision}.pdf").write_bytes(raw)
                doc = upload(
                    client,
                    raw,
                    parent=old_ids[index] if revision else None,
                    name=f"report-{index}-v{revision}.pdf",
                ).json()
                assert doc["status"] == "ready", (doc, runtime.error)
                (new_ids if revision else old_ids).append(doc["id"])
        old, new = create(client, old_ids), create(client, new_ids, "Corrected evidence")
        before, after = [fact(client, old, column).json()["id"] for column in ("Previous", "Current")]
        calc = calculate(client, old, [before, after]).json()["id"]
        audit_id = start(client, old, [item(calc, "growth", "25", "%")])
        audit_advance(client, old, audit_id)
        replay_id = create_replay(client, old, audit_id, new)
        assert advance(client, old, audit_id, replay_id).status_code == 200
        result = report(client, old, audit_id, replay_id).json()
        assert result["complete"] and result["support_lost"] == 1 and result["model_calls"] == 0
        assert result["items"][0]["before"]["status"] == "supported"
        assert result["items"][0]["after"]["status"] == "refuted"
        assert result["items"][0]["after"]["observed_value"] == "50"
        observation = next(o for o in result["items"][0]["observations"] if o["source_id"] == new_ids[0])
        assert all(c["page"] == 2 and c["document_id"] == old_ids[0] for c in observation["citations"])
        (tmp_path / "replay-report.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        image = client.get(f"/workbench/documents/{new_ids[0]}/pages/2", headers=headers()).json()["image"]
        (tmp_path / "replay-source-preview.png").write_bytes(base64.b64decode(image.split(",")[1]))
