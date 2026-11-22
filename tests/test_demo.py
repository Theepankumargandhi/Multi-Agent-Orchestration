import importlib
from unittest.mock import AsyncMock
from uuid import uuid4

from fastapi import FastAPI
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage

from service.demo import QuestionQuota, SinglePortApp


class Clock:
    def __init__(self, now: float):
        self.now = now

    def __call__(self) -> float:
        return self.now


def test_quota_limits_each_account_per_hour_and_frees_slots_as_they_age():
    clock = Clock(1_800_000_000)  # 2027-01-15, mid-day UTC
    quota = QuestionQuota(per_hour=2, per_day=100, clock=clock)
    assert quota.spend("alice") is None
    clock.now += 600
    assert quota.spend("alice") is None
    refused = quota.spend("alice")
    assert refused and "2 questions an hour" in refused and "50 minutes" in refused
    # Another account has its own allowance, and a refused question is not counted.
    assert quota.spend("bob") is None
    clock.now += 3000  # the first question is now an hour old
    assert quota.spend("alice") is None
    assert quota.spend("alice") is not None


def test_quota_stops_everyone_after_the_daily_total_until_midnight_utc():
    clock = Clock(1_800_000_000)
    quota = QuestionQuota(per_hour=10, per_day=3, clock=clock)
    assert [quota.spend(name) for name in ("a", "b", "c")] == [None, None, None]
    refused = quota.spend("d")
    assert refused and "resets at midnight UTC" in refused
    clock.now += 86_400
    assert quota.spend("d") is None


def test_single_port_app_serves_the_ui_and_the_api_under_api(tmp_path):
    (tmp_path / "index.html").write_text("<!doctype html><title>CiteRight</title>", encoding="utf-8")
    (tmp_path / "assets").mkdir()
    (tmp_path / "assets" / "app-1a2b.js").write_text("console.log(1)", encoding="utf-8")
    api = FastAPI()

    @api.get("/healthz")
    async def healthz():
        return {"status": "ok"}

    @api.post("/stream")
    async def stream():
        return {"path": "/stream"}

    with TestClient(SinglePortApp(api, tmp_path)) as c:
        page = c.get("/")
        assert page.status_code == 200 and "CiteRight" in page.text
        assert "frame-ancestors 'self' https://huggingface.co" in page.headers["content-security-policy"]
        assert page.headers["cache-control"] == "no-store"
        asset = c.get("/assets/app-1a2b.js")
        assert asset.status_code == 200 and "immutable" in asset.headers["cache-control"]
        assert c.get("/api/healthz").json() == {"status": "ok"}
        assert c.post("/api/stream").json() == {"path": "/stream"}
        assert c.get("/healthz").status_code == 404  # the API is only under /api
        assert c.get("/missing.js").status_code == 404


def test_demo_mode_limits_questions_and_documents_and_says_so(monkeypatch):
    from tests.test_document_uploads import FakeStore

    service_module = importlib.import_module("service.service")
    demo = importlib.import_module("service.demo")
    local_rag = importlib.import_module("agent.local_rag")
    from service.test_service import _auth_headers, client

    monkeypatch.setattr(demo, "DEMO_MODE", True)
    monkeypatch.setattr(demo, "DEMO_MAX_DOCUMENTS", 1)
    monkeypatch.setattr(service_module, "QUESTION_QUOTA", QuestionQuota(per_hour=1, per_day=50))
    monkeypatch.setenv("USE_CHROMA_RAG", "true")
    monkeypatch.setattr(local_rag, "USE_CHROMA_RAG", True)
    uploads = FakeStore()
    monkeypatch.setattr(local_rag, "_get_uploads_store", lambda: uploads)
    monkeypatch.setattr(local_rag, "_get_chroma_store", lambda: FakeStore())

    agent = type("MockAgent", (), {})()
    agent.ainvoke = AsyncMock(return_value={"messages": [AIMessage(content="Answer.")]})
    with client as c:
        c.app.state.agent = agent
        headers = _auth_headers(c, user_id=f"demo-{uuid4().hex[:8]}")
        capabilities = c.get("/capabilities").json()
        assert capabilities["demo"]["enabled"] is True and capabilities["demo"]["max_documents"] == 1

        assert c.post("/invoke", json={"message": "First?"}, headers=headers).status_code == 200
        limited = c.post("/stream", json={"message": "Second?"}, headers=headers)
        assert limited.status_code == 429
        assert "1 question an hour" in limited.json()["detail"]
        assert agent.ainvoke.await_count == 1

        first = c.post("/documents", params={"file_name": "a.md"}, content=b"# A\nAlpha.\n", headers=headers)
        assert first.status_code == 200
        # Uploading the same file again replaces it; a second file is refused.
        assert c.post("/documents", params={"file_name": "a.md"}, content=b"# A\nBeta.\n", headers=headers).status_code == 200
        second = c.post("/documents", params={"file_name": "b.md"}, content=b"# B\nGamma.\n", headers=headers)
        assert second.status_code == 422
        assert second.json()["detail"] == "The demo keeps up to 1 document per account. Remove one first."
