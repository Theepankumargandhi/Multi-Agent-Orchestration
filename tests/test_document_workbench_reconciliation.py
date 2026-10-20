"""Owner-private reconciliation controls; the opt-in test also extracts actual PDF bytes."""

import base64
import io
import json
import os
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing

import pytest
from test_document_workbench_versions import VersionExtraction, client_for, detail, headers, upload

from agentforge_sdk.models import fingerprint
from agentforge_sdk.spatial import MultimodalDocument, Region, region_fingerprint
from service.document_runtime import DocumentRuntime
from service.document_store import DocumentLibrary

BASE = "/workbench/documents/reconciliations"
SCOPE = {"entity": "Atlas", "period": "2026-Q1", "basis": "Requests; raw counts"}


class ReconciliationExtraction(VersionExtraction):
    def extract(self, raw, **meta):
        result = super().extract(raw, **meta)
        doc = result["manifest"]
        for page in doc["pages"]:
            for cell in page["regions"]:
                if cell.get("row") == 1 and cell.get("column") == 2:
                    if b"unsupported" in raw:
                        cell["text"] = "1,000"
                    elif b"usd" in raw:
                        cell["text"] = "USD 125"
                    elif b"eur" in raw:
                        cell["text"] = "EUR 125"
                if b"missing" in raw and cell.get("row") == 1 and cell.get("column") == 0:
                    cell["text"] = "Other"
                cell["region_id"] = region_fingerprint(doc["file_sha256"], page["page"], Region(**cell))
        doc["fingerprint"] = fingerprint({k: v for k, v in doc.items() if k != "fingerprint"})
        MultimodalDocument.model_validate_json(json.dumps(doc))
        return result


@pytest.fixture
def setup(tmp_path, monkeypatch):
    monkeypatch.setenv("WORKBENCH_DOCUMENTS_ENABLED", "true")
    library = DocumentLibrary(tmp_path / "facts.db", ReconciliationExtraction())
    with client_for(library) as client:
        yield client, library


def sources(client, contents=(b"%PDF-a", b"%PDF-b")):
    return [
        upload(client, content, name=f"report-{index}.pdf").json()["id"]
        for index, content in enumerate(contents)
    ]


def binding(client, ids, name="Quarterly request check"):
    return {
        "name": name,
        "scope": dict(SCOPE),
        "scope_confirmed": True,
        "sources": [
            {"document_id": i, "fingerprint": detail(client, i)["manifest"]["fingerprint"]} for i in ids
        ],
    }


def create(client, ids, name="Quarterly request check"):
    response = client.post(BASE, headers=headers(), json=binding(client, ids, name))
    assert response.status_code == 201, response.text
    return response.json()["id"]


def report(client, set_id, owner="alice"):
    return client.get(f"{BASE}/{set_id}", headers=headers(owner))


def fact(client, set_id, column="Current", row="Requests", expected=None):
    context = expected or report(client, set_id).json()["collection_fingerprint"]
    return client.post(
        f"{BASE}/{set_id}/facts",
        headers=headers(),
        json={"collection_fingerprint": context, "row_label": row, "column_label": column},
    )


def calculate(client, set_id, ids, operation="percent_change"):
    return client.post(
        f"{BASE}/{set_id}/calculations",
        headers=headers(),
        json={
            "collection_fingerprint": report(client, set_id).json()["collection_fingerprint"],
            "operation": operation,
            "fact_ids": ids,
        },
    )


def saved(client, set_id, result_id):
    return next(r for r in report(client, set_id).json()["results"] if r["id"] == result_id)


def exercise_conflict(client, ids):
    set_id = create(client, ids)
    before = fact(client, set_id, "Previous").json()["id"]
    current = fact(client, set_id).json()["id"]
    conflict = saved(client, set_id, current)
    assert conflict["payload"]["receipt"]["status"] == "conflict"
    assert conflict["payload"]["receipt"]["value"] is None
    assert conflict["attention_reason"] == "unresolved_evidence"
    assert sorted(len(g["source_ids"]) for g in conflict["payload"]["receipt"]["groups"]) == [1, 2]
    calc = calculate(client, set_id, [before, current])
    assert calc.status_code == 200, calc.text
    blocked = saved(client, set_id, calc.json()["id"])["payload"]["receipt"]
    assert (
        blocked["status"] == "abstained"
        and blocked["reason"] == "unresolved_fact"
        and blocked["value"] is None
    )
    review = {
        "digest": conflict["payload"]["digest"],
        "decision": "inspected",
        "note": "Inspected both values; no winner selected.",
    }
    path = f"{BASE}/{set_id}/results/{current}/reviews"
    assert client.post(path, headers=headers(), json=review).status_code == 200
    assert client.post(path, headers=headers(), json=review).status_code == 200
    assert client.post(path, headers=headers(), json={**review, "decision": "flagged"}).status_code == 409
    inspected = saved(client, set_id, current)
    assert (
        inspected["payload"] == conflict["payload"] and inspected["attention_reason"] == "unresolved_evidence"
    )
    assert calculate(client, set_id, [before, current]).json()["id"] == calc.json()["id"]
    return set_id


def test_conflicting_minority_never_loses_to_majority_or_owner_review(setup):
    client, _ = setup
    exercise_conflict(client, sources(client, (b"%PDF-a", b"%PDF-b", b"%PDF-v2")))


def test_agreement_exact_decimal_calculation_and_three_citations_per_source(setup):
    client, _ = setup
    ids = sources(client)
    set_id = create(client, ids)
    assert create(client, list(reversed(ids))) == set_id
    before = fact(client, set_id, "Previous").json()["id"]
    after = fact(client, set_id).json()["id"]
    assert fact(client, set_id).json()["id"] == after
    result = saved(client, set_id, after)
    receipt = result["payload"]["receipt"]
    assert (receipt["status"], receipt["value"], receipt["unit"]) == ("agreed", "125", "scalar")
    assert receipt["distinct_artifacts"] == 2
    assert {o["source_id"] for o in receipt["observations"]} == set(ids)
    assert all(
        [c["quote"] for c in o["citations"]] == ["Requests", "Current", "125"]
        for o in receipt["observations"]
    )
    calc = calculate(client, set_id, [before, after]).json()["id"]
    assert saved(client, set_id, calc)["payload"]["receipt"]["value"] == "25"
    assert report(client, set_id).json()["model_calls"] == 0
    assert (
        client.post(
            f"{BASE}/{set_id}/results/{after}/reviews",
            headers=headers(),
            json={"digest": result["payload"]["digest"], "decision": "inspected", "note": "Read source"},
        ).status_code
        == 200
    )
    assert saved(client, set_id, after)["attention_reason"] is None
    catalog = report(client, set_id).json()["catalog"]
    assert any(q["row_label"] == "Requests" and q["column_label"] == "Current" for q in catalog)
    listed = client.get("/workbench/documents", headers=headers()).json()["documents"]
    assert all(d["manifest_fingerprint"] and "original" not in d for d in listed)


@pytest.mark.parametrize(
    "contents,reason",
    [
        ((b"%PDF-same", b"%PDF-same"), "insufficient_distinct_artifacts"),
        ((b"%PDF-a", b"%PDF-missing"), "incomplete_coverage"),
        ((b"%PDF-a", b"%PDF-held"), "held"),
        ((b"%PDF-a", b"%PDF-duplicate"), "ambiguous"),
        ((b"%PDF-a", b"%PDF-unsupported"), "unsupported_number"),
        ((b"%PDF-usd", b"%PDF-eur"), "incompatible_units"),
    ],
)
def test_incomplete_or_incomparable_sources_cannot_release_a_value(setup, contents, reason):
    client, _ = setup
    set_id = create(client, sources(client, contents))
    result = fact(client, set_id)
    assert result.status_code == 200, result.text
    receipt = saved(client, set_id, result.json()["id"])["payload"]["receipt"]
    assert receipt["status"] == "abstained" and receipt["value"] is None
    assert reason in receipt["reasons"] or any(o["status"] == reason for o in receipt["observations"])


def test_owner_isolation_private_errors_and_source_lifecycle(setup, monkeypatch):
    client, library = setup
    ids = sources(client)
    set_id = create(client, ids)
    result_id = fact(client, set_id).json()["id"]
    assert client.get(BASE, headers=headers("bob")).json() == {"sets": []}
    for method, path, body in [
        ("GET", f"{BASE}/{set_id}", None),
        ("DELETE", f"{BASE}/{set_id}", None),
        ("POST", BASE, binding(client, ids)),
        (
            "POST",
            f"{BASE}/{set_id}/facts",
            {"collection_fingerprint": "a" * 64, "row_label": "Requests", "column_label": "Current"},
        ),
        (
            "POST",
            f"{BASE}/{set_id}/results/{result_id}/reviews",
            {"digest": "a" * 64, "decision": "inspected", "note": ""},
        ),
    ]:
        response = client.request(method, path, headers=headers("bob"), json=body)
        assert response.status_code == 404, response.text
        assert response.headers["cache-control"] == "no-store"
        assert "Atlas" not in response.text
    assert client.get(BASE).status_code == 401
    assert detail(client, ids[0])["evidence_set_count"] == 1
    assert client.delete(f"/workbench/documents/{ids[0]}", headers=headers()).status_code == 409
    monkeypatch.setenv("WORKBENCH_DOCUMENTS_ENABLED", "false")
    assert client.get(BASE, headers=headers()).status_code == 404
    monkeypatch.setenv("WORKBENCH_DOCUMENTS_ENABLED", "true")
    assert client.delete(f"{BASE}/{set_id}", headers=headers()).status_code == 200
    assert detail(client, ids[0])["evidence_set_count"] == 0
    with closing(library.connect()) as db:
        assert db.execute("SELECT COUNT(*) FROM evidence_set_results").fetchone()[0] == 0
        assert db.execute("SELECT COUNT(*) FROM documents").fetchone()[0] == 2
    assert client.delete(f"/workbench/documents/{ids[0]}", headers=headers()).status_code == 200


def test_linked_version_citations_keep_application_row_identity_and_frozen_membership(setup):
    client, _ = setup
    root = upload(client, b"%PDF-original").json()["id"]
    child = upload(client, b"%PDF-revised-same-values", parent=root).json()["id"]
    other = upload(client, b"%PDF-other", name="other.pdf").json()["id"]
    set_id = create(client, [child, other])
    context = report(client, set_id).json()["collection_fingerprint"]
    result = fact(client, set_id).json()["id"]
    observation = next(
        o
        for o in saved(client, set_id, result)["payload"]["receipt"]["observations"]
        if o["source_id"] == child
    )
    assert observation["source_id"] != root
    assert all(c["document_id"] == root for c in observation["citations"])
    new = upload(client, b"%PDF-v2", parent=child).json()["id"]
    current = report(client, set_id).json()
    assert current["collection_fingerprint"] == context
    assert new not in {s["id"] for s in current["sources"]}
    assert saved(client, set_id, result)["payload"]["receipt"]["value"] == "125"


@pytest.mark.parametrize(
    "change", ["false", "integer", "duplicate", "single", "policy", "multiline", "stale"]
)
def test_scope_snapshot_and_strict_request_controls(setup, change):
    client, _ = setup
    body = binding(client, sources(client))
    if change in {"false", "integer"}:
        body["scope_confirmed"] = False if change == "false" else 1
    elif change == "duplicate":
        body["sources"][1] = body["sources"][0]
    elif change == "single":
        body["sources"].pop()
    elif change == "policy":
        body["min_distinct_artifacts"] = 1
    elif change == "multiline":
        body["scope"]["entity"] = "PRIVATE\nSENTINEL"
    else:
        body["sources"][0]["fingerprint"] = "0" * 64
    response = client.post(BASE, headers=headers(), json=body)
    assert response.status_code == (409 if change == "stale" else 422), response.text
    assert "SENTINEL" not in response.text and response.headers["cache-control"] == "no-store"
    assert client.get(BASE, headers=headers()).json() == {"sets": []}


def test_operands_are_saved_facts_from_this_set_not_client_values(setup):
    client, _ = setup
    ids = sources(client)
    left, right = create(client, ids), create(client, ids, "Another scoped check")
    a = fact(client, left).json()["id"]
    assert calculate(client, right, [a], "sum").status_code == 404
    assert calculate(client, left, [a, a]).status_code == 422
    calc = calculate(client, left, [a], "sum").json()["id"]
    assert calculate(client, left, [calc], "sum").status_code == 422
    assert fact(client, left, expected="0" * 64).status_code == 409
    response = client.post(
        f"{BASE}/{left}/facts",
        headers=headers(),
        json={
            "collection_fingerprint": "0" * 64,
            "row_label": "PRIVATE_SENTINEL",
            "column_label": "Current",
            "value": "999",
        },
    )
    assert response.status_code == 422 and "PRIVATE_SENTINEL" not in response.text


@pytest.mark.parametrize("target", ["result", "review", "binding"])
def test_saved_payload_or_review_tampering_is_not_silently_trusted(setup, target):
    client, library = setup
    set_id = create(client, sources(client))
    result_id = fact(client, set_id).json()["id"]
    result = saved(client, set_id, result_id)
    client.post(
        f"{BASE}/{set_id}/results/{result_id}/reviews",
        headers=headers(),
        json={"digest": result["payload"]["digest"], "decision": "inspected", "note": ""},
    )
    with closing(library.connect()) as db, db:
        if target == "result":
            result["payload"]["receipt"]["value"] = "999"
            db.execute(
                "UPDATE evidence_set_results SET payload=? WHERE id=?",
                (json.dumps(result["payload"]), result_id),
            )
        elif target == "review":
            db.execute("UPDATE evidence_set_reviews SET digest=? WHERE result_id=?", ("0" * 64, result_id))
        else:
            db.execute("UPDATE evidence_sets SET digest=? WHERE id=?", ("0" * 64, set_id))
    assert report(client, set_id).status_code == 409


def test_parallel_identical_writes_are_idempotent_and_quotas_are_atomic(setup):
    client, library = setup
    ids = sources(client)
    body = binding(client, ids)
    with ThreadPoolExecutor(max_workers=2) as pool:
        responses = list(pool.map(lambda _: client.post(BASE, headers=headers(), json=body), range(2)))
    assert all(r.status_code == 201 for r in responses)
    set_id = responses[0].json()["id"]
    assert responses[1].json()["id"] == set_id
    context = report(client, set_id).json()["collection_fingerprint"]
    with ThreadPoolExecutor(max_workers=2) as pool:
        responses = list(pool.map(lambda _: fact(client, set_id, expected=context), range(2)))
    assert responses[0].json() == responses[1].json()
    for n in range(19):
        assert fact(client, set_id, row=f"Absent {n}").status_code == 200
    assert fact(client, set_id, row="Over quota").status_code == 429
    assert fact(client, set_id).status_code == 200  # Identical retry is still safe at the quota.
    for n in range(19):
        assert (
            client.post(BASE, headers=headers(), json=binding(client, ids, f"Set {n:02}")).status_code == 201
        )
    assert client.post(BASE, headers=headers(), json=binding(client, ids, "Over quota")).status_code == 429
    with closing(library.connect()) as db:
        assert db.execute("SELECT COUNT(*) FROM evidence_sets").fetchone()[0] == 20
        assert db.execute("SELECT COUNT(*) FROM evidence_set_sources").fetchone()[0] == 40
        assert db.execute("SELECT COUNT(*) FROM evidence_set_results").fetchone()[0] == 20


@pytest.mark.skipif(
    os.getenv("AF_RUN_DOCUMENT_DOCKER") != "1", reason="explicit isolated local PDF integration opt-in"
)
def test_actual_pdf_conflicts_moved_page_citations_agreement_and_guarded_calculation(tmp_path, monkeypatch):
    from reportlab.pdfgen import canvas

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
    library = DocumentLibrary(tmp_path / "real-facts.db", runtime)
    with client_for(library) as client:
        ids = []
        for index, number in enumerate(("125", "125", "150")):
            output = io.BytesIO()
            pdf = canvas.Canvas(output, pagesize=(600, 800))
            if index == 2:
                pdf.drawString(60, 720, "Atlas correction: metrics on the next page.")
                pdf.showPage()
            pdf.drawString(60, 720, f"Atlas report {index}: 2026-Q1 requests in raw counts.")
            pdf.grid([60, 240, 360, 480], [440, 480, 520, 560])
            for row, values in enumerate(
                [["Metric", "Previous", "Current"], ["Requests", "100", number], ["Errors", "10", "20"]]
            ):
                for column, text in enumerate(values):
                    pdf.drawString([70, 250, 370][column], 535 - row * 40, text)
            pdf.save()
            raw = output.getvalue()
            (tmp_path / f"report-{index}.pdf").write_bytes(raw)
            doc = upload(client, raw, name=f"report-{index}.pdf").json()
            assert doc["status"] == "ready", (doc, runtime.error)
            ids.append(doc["id"])
        conflict_id = exercise_conflict(client, ids)
        conflict = report(client, conflict_id).json()
        current = next(
            r
            for r in conflict["results"]
            if r["payload"]["kind"] == "fact"
            and r["payload"]["receipt"]["query"]["column_label"] == "Current"
        )
        moved = next(o for o in current["payload"]["receipt"]["observations"] if o["source_id"] == ids[2])
        assert [c["page"] for c in moved["citations"]] == [2, 2, 2]
        agreed_id = create(client, ids[:2], "Explicit agreeing subset")
        operands = [fact(client, agreed_id, column).json()["id"] for column in ("Previous", "Current")]
        calc_id = calculate(client, agreed_id, operands).json()["id"]
        assert saved(client, agreed_id, calc_id)["payload"]["receipt"]["value"] == "25"
        (tmp_path / "reconciliation-report.json").write_text(
            json.dumps(
                {
                    "conflicting_set": conflict,
                    "agreeing_set": report(client, agreed_id).json(),
                    "provider_calls": 0,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        for doc_id, page, name in [
            (ids[0], 1, "agreeing-preview.png"),
            (ids[2], 2, "conflicting-preview.png"),
        ]:
            image = client.get(f"/workbench/documents/{doc_id}/pages/{page}", headers=headers()).json()[
                "image"
            ]
            (tmp_path / name).write_bytes(base64.b64decode(image.split(",")[1]))
