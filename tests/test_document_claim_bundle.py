"""Explicit private replay export, unchanged originals and offline verification."""

from contextlib import closing

import pytest
from test_document_claim_audits import advance, item, report, seed, start
from test_document_workbench_reconciliation import BASE, create
from test_document_workbench_reconciliation import report as set_report
from test_document_workbench_reconciliation import setup as setup
from test_document_workbench_versions import headers

from agentforge_sdk import load_claim_replay_bundle, verify_claim_replay_bundle


def endpoint(set_id, audit_id):
    return f"{BASE}/{set_id}/audits/{audit_id}/replay-bundle"


def request(client, set_id, audit_id):
    return {
        "checkpoint_fingerprint": report(client, set_id, audit_id).json()["checkpoint_fingerprint"],
        "include_extraction_text": True,
    }


def completed(client, *, conflicting=False):
    set_id, count, calc = seed(client, conflicting=conflicting)
    audit_id = start(
        client, set_id, [item(count), item(calc, "growth", "25", "%"), item(count, "wrong", "999")]
    )
    advance(client, set_id, audit_id)
    advance(client, set_id, audit_id)
    return set_id, audit_id


@pytest.mark.parametrize("conflicting", [False, True])
def test_explicit_export_keeps_every_claim_and_replays_after_source_deletion(setup, conflicting):
    client, library = setup
    set_id, audit_id = completed(client, conflicting=conflicting)
    before = report(client, set_id, audit_id).json()
    results = set_report(client, set_id).json()["results"]
    response = client.post(
        endpoint(set_id, audit_id), headers=headers(), json=request(client, set_id, audit_id)
    )
    assert response.status_code == 200, response.text
    assert response.headers["cache-control"] == "no-store"
    bundle = load_claim_replay_bundle(response.content)
    assert [e.recipe.claim.claim_id for e in bundle.entries] == ["count", "growth", "wrong"]
    assert [e.check.model_dump(mode="json") for e in bundle.entries] == [i["check"] for i in before["items"]]
    assert report(client, set_id, audit_id).json() == before
    assert set_report(client, set_id).json()["results"] == results
    assert (
        client.post(
            endpoint(set_id, audit_id), headers=headers(), json=request(client, set_id, audit_id)
        ).json()
        == response.json()
    )
    assert not {"owner", "reviews", "original", "images", "notes"} & response.json().keys()
    client.delete(f"{BASE}/{set_id}", headers=headers())
    with closing(library.connect()) as db:
        ids = [r[0] for r in db.execute("SELECT id FROM documents")]
    for doc_id in ids:
        assert client.delete(f"/workbench/documents/{doc_id}", headers=headers()).status_code == 200
    verified = verify_claim_replay_bundle(bundle, expected_fingerprint=bundle.fingerprint)
    assert verified.replay_verified and verified.total == 3
    assert (verified.supported, verified.refuted, verified.unresolved) == (
        (0, 0, 3) if conflicting else (2, 1, 0)
    )


def test_partial_stale_cross_owner_and_wrong_set_exports_are_refused(setup):
    client, _ = setup
    set_id, count, _ = seed(client)
    audit_id = start(client, set_id, [item(count)])
    pending = request(client, set_id, audit_id)
    assert client.post(endpoint(set_id, audit_id), headers=headers(), json=pending).status_code == 409
    advance(client, set_id, audit_id)
    assert client.post(endpoint(set_id, audit_id), headers=headers(), json=pending).status_code == 409
    current = request(client, set_id, audit_id)
    assert client.post(endpoint(set_id, audit_id), headers=headers("bob"), json=current).status_code == 404
    other = create(client, [s["id"] for s in set_report(client, set_id).json()["sources"]], "Other set")
    assert client.post(endpoint(other, audit_id), headers=headers(), json=current).status_code == 404
    assert client.get(endpoint(set_id, audit_id), headers=headers()).status_code == 405


@pytest.mark.parametrize("acknowledgement", [False, 1, "true", None])
def test_export_requires_boolean_acknowledgement_without_private_echo(setup, acknowledgement):
    client, _ = setup
    set_id, audit_id = completed(client)
    body = request(client, set_id, audit_id)
    body["include_extraction_text"] = acknowledgement
    response = client.post(endpoint(set_id, audit_id), headers=headers(), json=body)
    assert response.status_code == 422 and response.headers["cache-control"] == "no-store"


def test_export_does_not_accept_subset_or_policy_overrides_and_rechecks_tampered_parents(setup):
    client, library = setup
    set_id, audit_id = completed(client)
    body = request(client, set_id, audit_id)
    response = client.post(
        endpoint(set_id, audit_id), headers=headers(), json={**body, "claim_ids": ["PRIVATE_SUBSET"]}
    )
    assert response.status_code == 422 and "PRIVATE_SUBSET" not in response.text
    with closing(library.connect()) as db, db:
        db.execute(
            "UPDATE evidence_claim_audits SET checkpoint_fingerprint=? WHERE id=?", ("0" * 64, audit_id)
        )
    assert client.post(endpoint(set_id, audit_id), headers=headers(), json=body).status_code == 409


def test_budget_rejection_leaves_no_new_persistence(setup, monkeypatch):
    client, library = setup
    set_id, audit_id = completed(client)
    before = report(client, set_id, audit_id).json()
    monkeypatch.setattr("agentforge_sdk.claim_bundle.MAX_BUNDLE_BYTES", 1)
    response = client.post(
        endpoint(set_id, audit_id), headers=headers(), json=request(client, set_id, audit_id)
    )
    assert response.status_code == 409
    assert report(client, set_id, audit_id).json() == before
    with closing(library.connect()) as db:
        assert db.execute("SELECT COUNT(*) FROM evidence_claim_audits").fetchone()[0] == 1
        assert db.execute("SELECT COUNT(*) FROM evidence_set_results").fetchone()[0] == 3
