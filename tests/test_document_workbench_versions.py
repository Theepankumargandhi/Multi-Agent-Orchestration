"""Authored manifests for lifecycle controls; the opt-in integration uses real PDF containers."""

import base64
import hashlib
import io
import json
import os
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from agentforge_sdk.models import fingerprint
from agentforge_sdk.spatial import (
    Box,
    EvidencePage,
    ExtractionConfig,
    MultimodalDocument,
    Region,
    region_fingerprint,
)
from service.document_api import create_document_router
from service.document_runtime import DocumentRuntime
from service.document_store import DocumentLibrary


class VersionExtraction:
    def extract(self, raw, **meta):
        digest = hashlib.sha256(raw).hexdigest()
        updated = b"v2" in raw
        grid = [
            ["Metric", "Previous", "Current"],
            ["Requests", "100", "150" if updated else "125"],
            ["Errors", "10", "20"],
        ]
        if b"duplicate" in raw:
            grid[2][0] = "Requests"
        if updated:
            grid[1:] = list(reversed(grid[1:]))
        regions = []
        for row, values in enumerate(grid):
            for column, text in enumerate(values):
                draft = Region(
                    region_id="0" * 64,
                    kind="table_cell",
                    text=text,
                    table_id="p1:t0",
                    row=row,
                    column=column,
                    box=Box(
                        x0=0.1 + column * 0.25,
                        x1=0.3 + column * 0.25,
                        y0=0.2 + row * 0.15,
                        y1=0.3 + row * 0.15,
                    ),
                )
                regions.append(draft.model_copy(update={"region_id": region_fingerprint(digest, 1, draft)}))
        for index, text in enumerate(
            ["Atlas monthly metrics.", "Errors remain within the reviewed operating range."]
        ):
            draft = Region(
                region_id="0" * 64,
                kind="text",
                text=text,
                box=Box(x0=0.1, x1=0.9, y0=0.7 + index * 0.1, y1=0.75 + index * 0.1),
            )
            regions.append(draft.model_copy(update={"region_id": region_fingerprint(digest, 1, draft)}))
        held = b"held" in raw
        page = EvidencePage(
            page=1,
            width=600,
            height=800,
            method="none" if held else "native",
            status="held" if held else "ready",
            reason="ocr_required" if held else "extracted",
            regions=() if held else tuple(regions),
        )
        payload = dict(
            protocol="agentforge-spatial-evidence-v1",
            document_id=meta["document_id"],
            source=meta["name"],
            version=meta["version"],
            file_sha256=digest,
            media_type="application/pdf",
            config=ExtractionConfig().model_dump(),
            extractor="changed-profile" if b"drift" in raw else "authored-versions",
            ocr_backend="none",
            pages=(page.model_dump(),),
        )
        document = MultimodalDocument(**payload, fingerprint=fingerprint(payload))
        return {
            "manifest": document.model_dump(mode="json"),
            "previews": [base64.b64encode(b"authored preview").decode()],
        }


def client_for(library):
    app = FastAPI()

    @app.middleware("http")
    async def identity(request, call_next):
        token = request.headers.get("authorization")
        if token in {"Bearer alice", "Bearer bob"}:
            request.state.user_id = token[7:]
        return await call_next(request)

    app.include_router(create_document_router(library))
    return TestClient(app)


def headers(owner="alice"):
    return {"Authorization": "Bearer " + owner}


@pytest.fixture
def setup(tmp_path, monkeypatch):
    monkeypatch.setenv("WORKBENCH_DOCUMENTS_ENABLED", "true")
    library = DocumentLibrary(tmp_path / "versions.db", VersionExtraction())
    with client_for(library) as client:
        yield client, library


def upload(client, content=b"%PDF-v1", parent=None, owner="alice", name="atlas.pdf"):
    return client.post(
        "/workbench/documents",
        params={"name": name, **({"parent_id": parent} if parent else {})},
        headers={**headers(owner), "Content-Type": "application/pdf"},
        content=content,
    )


def detail(client, document_id):
    return client.get(f"/workbench/documents/{document_id}", headers=headers()).json()


def seed(client):
    old = upload(client).json()["id"]
    manifest = detail(client, old)["manifest"]
    cells = manifest["pages"][0]["regions"]
    ids = [next(r["region_id"] for r in cells if r["text"] == number) for number in ("100", "125")]
    saved = []
    requests = [
        {"kind": "calculation", "operation": "percent_change", "region_ids": ids},
        {"kind": "question", "question": "Atlas monthly metrics"},
        {"kind": "question", "question": "Errors operating range"},
    ]
    for request in requests:
        response = client.post(
            f"/workbench/documents/{old}/queries",
            headers=headers(),
            json={**request, "document_fingerprint": manifest["fingerprint"]},
        )
        assert response.status_code == 200, response.text
        saved.append(response.json()["id"])
    new = upload(client, b"%PDF-v2", old).json()["id"]
    return old, new, saved


def create(client, old, new, saved, **options):
    return client.post(
        f"/workbench/documents/{new}/comparisons",
        headers=headers(),
        json={
            "before_id": old,
            "before_fingerprint": detail(client, old)["manifest"]["fingerprint"],
            "after_fingerprint": detail(client, new)["manifest"]["fingerprint"],
            "result_ids": saved,
            **options,
        },
    )


def report(client, new, comparison):
    return client.get(f"/workbench/documents/{new}/comparisons/{comparison}", headers=headers())


def advance(client, new, comparison, expected="initial"):
    return client.post(
        f"/workbench/documents/{new}/comparisons/{comparison}/advance",
        headers=headers(),
        json={"checkpoint_fingerprint": expected},
    )


def test_explicit_lineage_does_not_rewrite_legacy_receipts_or_independent_uploads(setup):
    client, library = setup
    old, new, saved = seed(client)
    left, right = detail(client, old), detail(client, new)
    assert left["manifest"]["document_id"] == right["manifest"]["document_id"] == old
    assert right["document"]["parent_id"] == old and right["document"]["family_id"] == old
    assert len(right["versions"]) == 2
    independent = upload(client, b"%PDF-independent").json()["id"]
    assert detail(client, independent)["document"]["family_id"] == independent
    assert upload(client, b"%PDF-independent", old).status_code == 409
    assert upload(client, b"%PDF-stolen", old, owner="bob").status_code == 404
    assert client.delete(f"/workbench/documents/{old}", headers=headers()).status_code == 409
    assert create(client, old, independent, saved).status_code == 409
    with closing(library.connect()) as db:
        receipt = json.loads(
            db.execute("SELECT payload FROM document_results WHERE id=?", (saved[0],)).fetchone()[0]
        )
        assert receipt["receipt"]["value"] == "25"


def test_assessment_only_and_bounded_resumable_refresh_do_not_transfer_reviews(setup):
    client, library = setup
    old, new, saved = seed(client)
    source = next(r for r in detail(client, old)["results"] if r["id"] == saved[0])
    assert (
        client.post(
            f"/workbench/documents/{old}/results/{saved[0]}/reviews",
            headers=headers(),
            json={
                "digest": source["payload"]["digest"],
                "decision": "supported",
                "note": "Old version inspected",
            },
        ).status_code
        == 200
    )
    assessment = create(client, old, new, saved).json()["id"]
    assert report(client, new, assessment).json()["processed"] == 0
    assert advance(client, new, assessment).status_code == 200
    initial = report(client, new, assessment).json()
    assert initial["processed"] == 2 and not initial["complete"]
    assert all(e["status"] == "refresh_required" for e in initial["entries"])
    assert not detail(client, new)["results"]
    comparison = create(client, old, new, saved, refresh_local=True).json()["id"]
    assert create(client, old, new, saved, refresh_local=True).json()["id"] == comparison
    assert advance(client, new, comparison).status_code == 200
    partial = report(client, new, comparison).json()
    assert partial["processed"] == 2 and not partial["complete"] and partial["total"] == 3
    assert partial["entries"][0]["replay"]["result"]["value"] == "50"
    assert partial["entries"][0]["result_changed"] is True
    assert all(i["review"] is None for i in partial["items"])
    assert advance(client, new, comparison).status_code == 409
    # Reconstruct the application instance over the same protected DB, not a browser-supplied checkpoint.
    restarted = DocumentLibrary(library.path, VersionExtraction())
    with client_for(restarted) as fresh_client:
        assert advance(fresh_client, new, comparison, partial["checkpoint_fingerprint"]).status_code == 200
        complete = report(fresh_client, new, comparison).json()
        assert complete["complete"] and complete["processed"] == 3 and complete["model_calls"] == 0
        assert (
            len(complete["review_queue"]) == 3
        )  # Includes unchanged fresh answers needing a new owner review.
        target_id = complete["items"][0]["target_result_id"]
        result = next(r for r in detail(fresh_client, new)["results"] if r["id"] == target_id)
        assert (
            result["payload"]["receipt"]["document_fingerprint"]
            == detail(fresh_client, new)["manifest"]["fingerprint"]
        )
        assert (
            client.post(
                f"/workbench/documents/{new}/results/{target_id}/reviews",
                headers=headers(),
                json={
                    "digest": result["payload"]["digest"],
                    "decision": "supported",
                    "note": "New version inspected",
                },
            ).status_code
            == 200
        )
        assert len(report(fresh_client, new, comparison).json()["review_queue"]) == 2


@pytest.mark.parametrize("case", ["held", "drift", "duplicate"])
def test_uncertain_or_changed_profiles_block_calculation_replay(setup, case):
    client, _ = setup
    old, _, saved = seed(client)
    new = upload(client, b"%PDF-v2-" + case.encode(), old).json()["id"]
    comparison = create(client, old, new, saved[:1], refresh_local=True).json()["id"]
    assert advance(client, new, comparison).status_code == 200
    data = report(client, new, comparison).json()
    assert data["entries"][0]["status"] == "blocked"
    assert data["items"][0]["target_result_id"] is None
    assert not detail(client, new)["results"]


def test_budget_zero_still_replays_tools_and_does_not_repeat_steps_under_concurrency(setup):
    client, _ = setup
    old, new, saved = seed(client)
    comparison = create(client, old, new, saved, refresh_local=True, max_answer_attempts=0).json()["id"]
    with ThreadPoolExecutor(max_workers=2) as pool:
        statuses = list(pool.map(lambda _: advance(client, new, comparison).status_code, range(2)))
    assert sorted(statuses) == [200, 409]
    data = report(client, new, comparison).json()
    assert data["entries"][0]["status"] == "refreshed"
    assert data["entries"][1]["reason"] == "answer_budget_exhausted"
    assert data["processed"] == 2


def test_owner_version_checkpoint_and_validation_boundaries(setup):
    client, library = setup
    old, new, saved = seed(client)
    comparison = create(client, old, new, saved, refresh_local=True).json()["id"]
    sibling = upload(client, b"%PDF-v2-sibling", old).json()["id"]
    assert create(client, new, sibling, saved).status_code == 409
    base = f"/workbench/documents/{new}/comparisons/{comparison}"
    assert client.get(base, headers=headers("bob")).status_code == 404
    assert (
        client.post(
            base + "/advance", headers=headers("bob"), json={"checkpoint_fingerprint": "initial"}
        ).status_code
        == 404
    )
    assert (
        client.post(
            base + "/advance",
            headers=headers(),
            json={"checkpoint_fingerprint": "initial", "checkpoint": "PRIVATE"},
        ).status_code
        == 422
    )
    request = {
        "before_id": old,
        "before_fingerprint": "0" * 64,
        "after_fingerprint": detail(client, new)["manifest"]["fingerprint"],
        "result_ids": saved,
    }
    assert (
        client.post(f"/workbench/documents/{new}/comparisons", headers=headers(), json=request).status_code
        == 409
    )
    assert create(client, old, new, [saved[0], saved[0]]).status_code == 422
    assert advance(client, new, comparison).status_code == 200
    with closing(library.connect()) as db, db:
        checkpoint = json.loads(
            db.execute("SELECT checkpoint FROM document_comparisons WHERE id=?", (comparison,)).fetchone()[0]
        )
        checkpoint["entries"][0]["status"] = "current"
        db.execute(
            "UPDATE document_comparisons SET checkpoint=? WHERE id=?", (json.dumps(checkpoint), comparison)
        )
    response = report(client, new, comparison)
    assert response.status_code == 409 and "PRIVATE" not in response.text


def test_target_quota_failure_rolls_back_the_checkpoint_and_fresh_results(setup):
    client, _ = setup
    old, new, saved = seed(client)
    fp = detail(client, new)["manifest"]["fingerprint"]
    for index in range(20):
        assert (
            client.post(
                f"/workbench/documents/{new}/queries",
                headers=headers(),
                json={"document_fingerprint": fp, "kind": "question", "question": f"Atlas {index}"},
            ).status_code
            == 200
        )
    comparison = create(client, old, new, saved, refresh_local=True).json()["id"]
    assert advance(client, new, comparison).status_code == 429
    data = report(client, new, comparison).json()
    assert data["processed"] == 0 and data["checkpoint_fingerprint"] == "initial"
    assert len(detail(client, new)["results"]) == 20


def test_migration_backfills_existing_documents_without_changing_saved_bytes(setup):
    client, library = setup
    old, _, _ = seed(client)
    with closing(library.connect()) as db, db:
        before = db.execute("SELECT manifest,original FROM documents WHERE id=?", (old,)).fetchone()
        # Simulate the 0.2 schema with one root document and no lineage metadata.
        db.execute("DELETE FROM documents WHERE id != ?", (old,))
        db.execute("DROP TABLE document_lineage")
    DocumentLibrary(library.path, VersionExtraction())
    with closing(library.connect()) as db:
        after = db.execute("SELECT manifest,original FROM documents WHERE id=?", (old,)).fetchone()
        lineage = db.execute(
            "SELECT family_id,parent_id FROM document_lineage WHERE document_id=?", (old,)
        ).fetchone()
        assert tuple(before) == tuple(after) and tuple(lineage) == (old, None)


@pytest.mark.skipif(
    os.getenv("AF_RUN_DOCUMENT_DOCKER") != "1", reason="explicit local Docker integration opt-in"
)
def test_real_pdf_revision_moves_page_reorders_rows_and_recalculates_without_reusing_review(
    tmp_path, monkeypatch
):
    from reportlab.pdfgen import canvas

    def pdf_bytes(updated):
        output = io.BytesIO()
        pdf = canvas.Canvas(output, pagesize=(600, 800))
        if updated:
            pdf.drawString(60, 720, "Atlas revision notes. Metrics moved to the next page.")
            pdf.showPage()
        pdf.drawString(60, 720, "Atlas monthly metrics.")
        pdf.grid([60, 240, 360, 480], [440, 480, 520, 560])
        data = [
            ["Metric", "Previous", "Current"],
            ["Requests", "100", "150" if updated else "125"],
            ["Errors", "10", "20"],
        ]
        if updated:
            data[1:] = list(reversed(data[1:]))
        for row, values in enumerate(data):
            for column, text in enumerate(values):
                pdf.drawString([70, 250, 370][column], 535 - row * 40, text)
        pdf.save()
        return output.getvalue()

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
    library = DocumentLibrary(tmp_path / "real-versions.db", runtime)
    with client_for(library) as client:
        old_bytes, new_bytes = pdf_bytes(False), pdf_bytes(True)
        (tmp_path / "atlas-v1.pdf").write_bytes(old_bytes)
        (tmp_path / "atlas-v2.pdf").write_bytes(new_bytes)
        old = upload(client, old_bytes, name="atlas-v1.pdf").json()
        assert old["status"] == "ready", (old, runtime.error)
        left = detail(client, old["id"])
        cells = left["manifest"]["pages"][0]["regions"]
        operands = [next(r["region_id"] for r in cells if r["text"] == value) for value in ("100", "125")]
        saved = client.post(
            f"/workbench/documents/{old['id']}/queries",
            headers=headers(),
            json={
                "kind": "calculation",
                "operation": "percent_change",
                "region_ids": operands,
                "document_fingerprint": left["manifest"]["fingerprint"],
            },
        ).json()["id"]
        result = detail(client, old["id"])["results"][0]
        assert result["payload"]["receipt"]["value"] == "25"
        assert (
            client.post(
                f"/workbench/documents/{old['id']}/results/{saved}/reviews",
                headers=headers(),
                json={"digest": result["payload"]["digest"], "decision": "supported", "note": "Inspected v1"},
            ).status_code
            == 200
        )
        new = upload(client, new_bytes, old["id"], name="atlas-v2.pdf").json()
        assert new["status"] == "ready", (new, runtime.error)
        comparison = create(client, old["id"], new["id"], [saved], refresh_local=True).json()["id"]
        assert advance(client, new["id"], comparison).status_code == 200
        data = report(client, new["id"], comparison).json()
        assert data["complete"] and data["entries"][0]["status"] == "refreshed"
        assert data["entries"][0]["replay"]["result"]["value"] == "50"
        assert data["entries"][0]["replay"]["result"]["citations"][0]["page"] == 2
        assert data["items"][0]["review"] is None and len(data["review_queue"]) == 1
        assert data["entries"][0]["result_changed"] is True and data["model_calls"] == 0
        assert any("numeric" in r["flags"] and "page" in r["flags"] for r in data["changes"]["regions"])
        (tmp_path / "revision-report.json").write_text(json.dumps(data, indent=2), encoding="utf-8")
        for doc, page, filename in [(old, 1, "v1-preview.png"), (new, 2, "v2-preview.png")]:
            preview = client.get(f"/workbench/documents/{doc['id']}/pages/{page}", headers=headers()).json()[
                "image"
            ]
            (tmp_path / filename).write_bytes(base64.b64decode(preview.split(",")[1]))
