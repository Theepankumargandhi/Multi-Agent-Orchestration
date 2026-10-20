"""Owner/version/budget controls around the SDK. Fake extraction unless explicitly marked Docker."""

import base64
import hashlib
import io
import os
import time
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

PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+aG1sAAAAASUVORK5CYII="
)
RAW = b"%PDF-authored-private-fixture"


class AuthoredExtraction:
    calls = 0
    held = False
    fail = False
    error = ""

    def extract(self, raw, **meta):
        self.calls += 1
        if self.fail:
            raise RuntimeError("private raw provider/parser error must not escape")
        digest = hashlib.sha256(raw).hexdigest()
        cells = []
        for index, text in enumerate(["10", "20"]):
            draft = Region(
                region_id="0" * 64,
                kind="table_cell",
                text=text,
                box=Box(x0=0.1, y0=0.1 + index * 0.2, x1=0.3, y1=0.2 + index * 0.2),
                table_id="authored-table",
                row=index,
                column=0,
            )
            cells.append(draft.model_copy(update={"region_id": region_fingerprint(digest, 1, draft)}))
        draft = Region(
            region_id="0" * 64,
            kind="text",
            text="Annual revenue increased after the launch.",
            box=Box(x0=0.1, y0=0.6, x1=0.9, y1=0.7),
        )
        cells.append(draft.model_copy(update={"region_id": region_fingerprint(digest, 1, draft)}))
        page = EvidencePage(
            page=1,
            width=600,
            height=800,
            method="none" if self.held else "native",
            status="held" if self.held else "ready",
            reason="ocr_required" if self.held else "extracted",
            regions=() if self.held else tuple(cells),
        )
        payload = dict(
            document_id=meta["document_id"],
            source=meta["name"],
            version=meta["version"],
            file_sha256=digest,
            media_type="application/pdf",
            config=ExtractionConfig().model_dump(),
            extractor="authored-extraction",
            ocr_backend="none",
            pages=(page.model_dump(),),
            protocol="agentforge-spatial-evidence-v1",
        )
        document = MultimodalDocument(**payload, fingerprint=fingerprint(payload))
        return {"manifest": document.model_dump(mode="json"), "previews": [base64.b64encode(PNG).decode()]}


def make_client(library):
    app = FastAPI()

    @app.middleware("http")
    async def identity(request, call_next):
        # Authored identities only. Production uses the service's verified bearer middleware.
        token = request.headers.get("authorization", "")
        if token in {"Bearer alice", "Bearer bob"}:
            request.state.user_id = token[7:]
        return await call_next(request)

    app.include_router(create_document_router(library))
    return TestClient(app)


def headers(owner="alice", media=None):
    return {"Authorization": "Bearer " + owner, **({"Content-Type": media} if media else {})}


@pytest.fixture
def setup(tmp_path, monkeypatch):
    monkeypatch.setenv("WORKBENCH_DOCUMENTS_ENABLED", "true")
    runtime = AuthoredExtraction()
    library = DocumentLibrary(tmp_path / "library.db", runtime)
    with make_client(library) as client:
        yield client, library, runtime


def upload(client, raw=RAW, name="annual.pdf", owner="alice", ocr=False):
    return client.post(
        "/workbench/documents",
        params={"name": name, "ocr": str(ocr).lower()},
        content=raw,
        headers=headers(owner, "application/pdf"),
    )


def detail(client, doc, owner="alice"):
    return client.get("/workbench/documents/" + doc, headers=headers(owner))


def test_disabled_and_spoofed_identity_never_parse(setup, monkeypatch):
    client, _, runtime = setup
    assert client.get("/workbench/documents", headers={"X-User-ID": "alice"}).status_code == 401
    monkeypatch.setenv("WORKBENCH_DOCUMENTS_ENABLED", "false")
    assert upload(client).status_code == 404
    assert runtime.calls == 0


def test_upload_ownership_minimal_library_and_content_idempotence(setup):
    client, _, runtime = setup
    response = upload(client)
    assert response.status_code == 201 and response.headers["cache-control"] == "no-store"
    assert response.json()["status"] == "ready"
    doc = response.json()["id"]
    assert upload(client).json()["id"] == doc and runtime.calls == 1
    assert upload(client, raw=RAW + b"new").json()["id"] != doc
    assert detail(client, doc, "bob").status_code == 404
    assert client.get(f"/workbench/documents/{doc}/pages/1", headers=headers("bob")).status_code == 404
    assert client.get("/workbench/documents", headers=headers("bob")).json() == {"documents": []}
    listing = client.get("/workbench/documents", headers=headers()).json()
    assert all(
        not {"original", "manifest", "previews", "owner"} & item.keys() for item in listing["documents"]
    )
    assert (
        client.get(f"/workbench/documents/{doc}/pages/1", headers=headers())
        .json()["image"]
        .startswith("data:image/png;base64,")
    )


@pytest.mark.parametrize(
    "name,raw,media,expected",
    [
        ("../private.pdf", RAW, "application/pdf", 422),
        ("x.svg", b"<svg/>", "image/svg+xml", 415),
        ("valid.pdf", b"not a pdf", "application/pdf", 415),
        ("valid.pdf", RAW, "text/html", 415),
        ("huge.pdf", b"%PDF-" + b"x" * 3_000_000, "application/pdf", 413),
    ],
    ids=["path", "type", "signature", "content-type", "stream-budget"],
)
def test_upload_stream_and_signature_limits(setup, name, raw, media, expected):
    client, _, runtime = setup
    response = client.post(
        "/workbench/documents", params={"name": name}, headers=headers(media=media), content=raw
    )
    assert response.status_code == expected and runtime.calls == 0


def test_queries_cells_reviews_and_version_binding(setup):
    client, library, _ = setup
    doc = upload(client).json()["id"]
    manifest = detail(client, doc).json()["manifest"]
    query = {
        "kind": "question",
        "question": "What happened to annual revenue?",
        "document_fingerprint": manifest["fingerprint"],
    }
    path = f"/workbench/documents/{doc}/queries"
    result = client.post(path, json=query, headers=headers())
    assert result.status_code == 200
    assert client.post(path, json=query, headers=headers()).json() == result.json()
    assert (
        client.post(path, json={**query, "document_fingerprint": "0" * 64}, headers=headers()).status_code
        == 409
    )
    assert client.post(path, json={**query, "user_id": "bob"}, headers=headers()).status_code == 422
    assert client.post(path, json=query, headers=headers("bob")).status_code == 404
    regions = [r["region_id"] for r in manifest["pages"][0]["regions"][:2]]
    calculation = {
        "kind": "calculation",
        "operation": "sum",
        "region_ids": regions,
        "document_fingerprint": manifest["fingerprint"],
    }
    assert client.post(path, json=calculation, headers=headers()).status_code == 200
    saved = detail(client, doc).json()["results"]
    assert all(r["payload"]["lineage_verified"] and r["payload"]["model_calls"] == 0 for r in saved)
    assert any(r["payload"]["receipt"].get("value") == "30" for r in saved)
    answer = next(r for r in saved if r["id"] == result.json()["id"])
    assert answer["payload"]["receipt"]["status"] == "evidence_found"
    review_path = f"/workbench/documents/{doc}/results/{answer['id']}/reviews"
    review = {"digest": answer["payload"]["digest"], "decision": "supported", "note": "Inspected the source."}
    assert client.post(review_path, json={**review, "digest": "0" * 64}, headers=headers()).status_code == 409
    assert client.post(review_path, json=review, headers=headers("bob")).status_code == 404
    assert client.post(review_path, json=review, headers=headers()).status_code == 200
    assert client.post(review_path, json=review, headers=headers()).status_code == 200
    assert (
        client.post(review_path, json={**review, "decision": "unsupported"}, headers=headers()).status_code
        == 409
    )
    assert client.delete(f"/workbench/documents/{doc}", headers=headers("bob")).status_code == 404
    assert client.delete(f"/workbench/documents/{doc}", headers=headers()).status_code == 200
    assert detail(client, doc).status_code == 404
    with closing(library.connect()) as db:
        assert db.execute("SELECT COUNT(*) FROM document_results").fetchone()[0] == 0
        assert db.execute("SELECT COUNT(*) FROM document_reviews").fetchone()[0] == 0


def test_held_page_abstains_and_parser_errors_are_private(setup):
    client, _, runtime = setup
    runtime.held = True
    doc = upload(client).json()
    assert doc["status"] == "held"
    manifest = detail(client, doc["id"]).json()["manifest"]
    response = client.post(
        f"/workbench/documents/{doc['id']}/queries",
        headers=headers(),
        json={"kind": "question", "question": "Revenue?", "document_fingerprint": manifest["fingerprint"]},
    )
    assert response.status_code == 200
    receipt = detail(client, doc["id"]).json()["results"][0]["payload"]["receipt"]
    assert receipt["status"] == "abstained" and receipt["reason"] == "incomplete_extraction"
    runtime.fail = True
    failed = upload(client, raw=RAW + b"broken")
    assert failed.json()["status"] == "failed" and "private raw" not in failed.text


def test_saved_result_budget_rejects_new_work_but_allows_identical_retry(setup):
    client, _, _ = setup
    doc = upload(client).json()["id"]
    manifest = detail(client, doc).json()["manifest"]
    path = f"/workbench/documents/{doc}/queries"
    first = None
    for index in range(20):
        body = {
            "kind": "question",
            "question": f"Annual revenue {index}?",
            "document_fingerprint": manifest["fingerprint"],
        }
        result = client.post(path, json=body, headers=headers())
        assert result.status_code == 200
        if index == 0:
            first = result.json()
    assert (
        client.post(
            path, json={**body, "question": "Another revenue question?"}, headers=headers()
        ).status_code
        == 429
    )
    assert (
        client.post(path, json={**body, "question": "Annual revenue 0?"}, headers=headers()).json() == first
    )


def test_busy_quota_and_crash_recovery(setup):
    client, library, _ = setup
    doc = upload(client).json()["id"]
    with closing(library.connect()) as db, db:
        db.execute("UPDATE documents SET status='processing' WHERE id=?", (doc,))
    assert upload(client, raw=RAW + b"busy").status_code == 429
    with closing(library.connect()) as db, db:
        db.execute("UPDATE documents SET created=? WHERE id=?", (time.time() - 400, doc))
    assert detail(client, doc).json()["document"]["reason"] == "extraction_interrupted"
    assert upload(client, raw=RAW + b"next").status_code == 201
    for index in range(8):
        assert upload(client, raw=RAW + str(index).encode()).status_code == 201
    assert upload(client, raw=RAW + b"over-limit").status_code == 429


def test_runtime_timeout_always_removes_only_the_owned_container(monkeypatch):
    from types import SimpleNamespace

    from service import document_runtime

    calls = []

    def run(argv, **kwargs):
        calls.append((argv, kwargs))
        return SimpleNamespace(
            returncode=124 if argv[1] == "exec" else 0,
            timed_out=argv[1] == "exec",
            stdout="sha256:" + "a" * 64 if argv[1] == "image" else "",
            stdout_bytes=71 if argv[1] == "image" else 0,
            stderr="PRIVATE content must not escape",
        )

    monkeypatch.setattr(document_runtime, "run_bounded", run)
    with pytest.raises(RuntimeError, match="isolated_document_exec_failed") as error:
        DocumentRuntime().extract(
            RAW, name="fixture.pdf", suffix=".pdf", document_id="doc", version="v1", ocr=False
        )
    assert "PRIVATE" not in str(error.value)
    create = next(argv for argv, _ in calls if argv[1] == "create")
    assert {
        "--network=none",
        "--read-only",
        "--cap-drop=ALL",
        "--memory=512m",
        "--pids-limit=64",
        "--pull=never",
        "--rm",
        "--user=10001:10001",
    } <= set(create)
    assert not any("mount" in arg or "env-file" in arg for arg in create)
    assert calls[-1][0][:3] == ["docker", "rm", "-f"]
    assert calls[-1][0][3].startswith("agentforge-doc-")
    assert next(kwargs for argv, kwargs in calls if argv[1] == "exec")["stdin_file"].closed


@pytest.mark.skipif(
    os.getenv("AF_RUN_DOCUMENT_DOCKER") != "1", reason="explicit local Docker integration opt-in"
)
def test_real_container_pdf_image_ocr_and_cleanup(tmp_path, monkeypatch):
    from PIL import Image, ImageDraw
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
    library = DocumentLibrary(tmp_path / "real.db", runtime)
    output = io.BytesIO()
    pdf = canvas.Canvas(output, pagesize=(600, 800))
    pdf.drawString(60, 720, "Annual revenue increased after the launch.")
    pdf.grid([60, 260, 460], [500, 540, 580])
    for x, y, text in [(70, 555, "Old"), (270, 555, "New"), (70, 515, "10"), (270, 515, "20")]:
        pdf.drawString(x, y, text)
    pdf.save()
    (tmp_path / "native.pdf").write_bytes(output.getvalue())
    with make_client(library) as client:
        native = upload(client, raw=output.getvalue(), name="native.pdf").json()
        assert native["status"] == "ready", (native, runtime.error)
        manifest = detail(client, native["id"]).json()["manifest"]
        assert manifest["pages"][0]["method"] == "native"
        cells = [r["region_id"] for r in manifest["pages"][0]["regions"] if r["text"] in {"10", "20"}]
        assert len(cells) == 2
        assert (
            client.post(
                f"/workbench/documents/{native['id']}/queries",
                headers=headers(),
                json={
                    "kind": "calculation",
                    "operation": "sum",
                    "region_ids": cells,
                    "document_fingerprint": manifest["fingerprint"],
                },
            ).status_code
            == 200
        )
        assert detail(client, native["id"]).json()["results"][0]["payload"]["receipt"]["value"] == "30"
        assert (
            client.post(
                f"/workbench/documents/{native['id']}/queries",
                headers=headers(),
                json={
                    "kind": "question",
                    "question": "Annual revenue?",
                    "document_fingerprint": manifest["fingerprint"],
                },
            ).status_code
            == 200
        )
        preview = client.get(f"/workbench/documents/{native['id']}/pages/1", headers=headers()).json()[
            "image"
        ]
        preview_bytes = base64.b64decode(preview.split(",")[1])
        (tmp_path / "native-preview.png").write_bytes(preview_bytes)
        assert Image.open(io.BytesIO(preview_bytes)).size == (900, 1200)
        image = Image.new("RGB", (900, 200), "white")
        ImageDraw.Draw(image).text((20, 60), "Revenue increased after launch", fill="black", font_size=36)
        raster = io.BytesIO()
        image.save(raster, format="PNG")
        for ocr, expected in [(False, "held"), (True, "ready")]:
            response = client.post(
                "/workbench/documents",
                params={"name": "scan.png", "ocr": str(ocr).lower()},
                content=raster.getvalue(),
                headers=headers(media="image/png"),
            )
            assert response.json()["status"] == expected, response.text
        invalid = upload(client, raw=b"%PDF-this is invalid", name="broken.pdf").json()
        assert invalid["status"] == "failed"
    from code_agent.bounded_process import run_bounded

    remaining = run_bounded(
        ["docker", "ps", "-a", "--filter", "name=agentforge-doc-", "--format", "{{.Names}}"],
        timeout=10,
        max_chars=1000,
    )
    assert remaining.returncode == 0 and not remaining.stdout.strip()
