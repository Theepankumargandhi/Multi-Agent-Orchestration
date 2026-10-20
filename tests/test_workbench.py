"""Actual Workbench router + SQLite controls; no provider or sandbox execution."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from code_agent.execution import LocalArtifactStore, SQLiteJobStore
from code_agent.job_models import CodeTaskRequest
from code_agent.workbench import create_workbench_router


@pytest.fixture
def setup(tmp_path, monkeypatch):
    monkeypatch.setenv("WORKBENCH_REPOSITORIES", "example,team/approved")
    monkeypatch.setenv("OPENAI_API_KEY", "authored-offline-key")
    monkeypatch.setenv("OPENAI_CHAT_MODEL", "authored-model")
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    store = SQLiteJobStore(tmp_path / "jobs.db", LocalArtifactStore(tmp_path / "artifacts"))
    manager = SimpleNamespace(store=store, workflow="verified_pr", submit=AsyncMock())
    app = FastAPI()

    @app.middleware("http")
    async def authored_identity(request, call_next):
        # Deliberately authored identities for router tests, NOT the production auth implementation.
        token = request.headers.get("Authorization")
        if token in {"Bearer alice", "Bearer bob"}:
            request.state.user_id = token.removeprefix("Bearer ")
        return await call_next(request)

    app.include_router(create_workbench_router(manager, code_enabled=True))
    with TestClient(app) as client:
        yield client, store, manager


def headers(owner="alice", key="authored-submit-1"):
    return {"Authorization": "Bearer " + owner, "Idempotency-Key": key}


@pytest.mark.parametrize("path", ["/workbench/config", "/workbench/code/tasks"])
def test_workbench_never_trusts_a_body_or_header_user_id(setup, path):
    client, _, manager = setup
    assert client.get(path, headers={"X-User-ID": "alice"}).status_code == 401
    manager.submit.assert_not_awaited()


def test_configuration_is_explicit_not_filesystem_discovery_or_provider_check(setup):
    client, _, manager = setup
    response = client.get("/workbench/config", headers=headers())
    assert response.status_code == 200 and response.headers["cache-control"] == "no-store"
    payload = response.json()
    assert payload["repositories"] == ["example", "team/approved"]
    assert payload["coding_models"] == ["authored-model"] and payload["identity"] == "alice"
    assert payload["integrations"]["documents"] == "disabled"
    assert "authored-offline-key" not in response.text and "repository_root" not in response.text
    manager.submit.assert_not_awaited()


@pytest.mark.parametrize("value", ["../outside", "/absolute", "C:/private", "example,example", "a/../b", "a//b"])
def test_invalid_operator_repository_configuration_fails_closed(setup, monkeypatch, value):
    client, _, _ = setup
    monkeypatch.setenv("WORKBENCH_REPOSITORIES", value)
    assert client.get("/workbench/config", headers=headers()).status_code == 503


def test_history_is_owner_scoped_bounded_and_does_not_release_artifacts(setup):
    client, store, manager = setup
    request = CodeTaskRequest(repository="example", issue="Fix rounding in invoice totals", model="authored-model")
    alice, _ = store.enqueue("alice", request, "alice-authored-key", 3)
    bob, _ = store.enqueue("bob", request, "bob-authored-key", 3)
    response = client.get("/workbench/code/tasks?limit=1", headers=headers())
    assert response.status_code == 200
    rows = response.json()["tasks"]
    assert len(rows) == 1 and rows[0]["task_id"] == alice.task_id
    assert bob.task_id not in response.text
    assert not {"user_id", "result", "patch", "lease_owner", "idempotency_key"} & rows[0].keys()
    assert client.get("/workbench/code/tasks?limit=51", headers=headers()).status_code == 422
    manager.submit.assert_not_awaited()


def test_submission_preserves_owner_and_idempotency_and_fixes_test_command(setup):
    client, store, manager = setup
    request = CodeTaskRequest(repository="example", issue="Fix rounding in invoice totals", model="authored-model")
    record, _ = store.enqueue("alice", request, "authored-submit-1", 3)
    manager.submit.return_value = record
    response = client.post("/workbench/code/tasks", headers=headers(), json=request.model_dump(exclude={"test_command"}))
    assert response.status_code == 202 and response.json()["task_id"] == record.task_id
    owner, passed, key = manager.submit.await_args.args
    assert owner == "alice" and key == "authored-submit-1"
    assert passed.test_command == ["python", "-m", "pytest", "-q"]


@pytest.mark.parametrize("change,status", [({"repository": "other"}, 403), ({"model": "unconfigured"}, 400),
                                         ({"test_command": ["sh", "-c", "not executed"]}, 422),
                                         ({"policy": {"network_enabled": True}}, 422), ({"user_id": "bob"}, 422),
                                         ({"issue": " " * 20}, 422)])
def test_browser_cannot_invent_repositories_models_commands_policies_or_identity(setup, change, status):
    client, _, manager = setup
    payload = dict(repository="example", issue="Fix rounding in invoice totals", model="authored-model") | change
    response = client.post("/workbench/code/tasks", headers=headers(), json=payload)
    assert response.status_code == status
    manager.submit.assert_not_awaited()


def test_validation_does_not_echo_private_payload_and_missing_key_is_rejected(setup):
    client, _, manager = setup
    payload = dict(repository="example", issue="PRIVATE submitted issue", model="authored-model")
    response = client.post("/workbench/code/tasks", headers=headers(), json={**payload, "secret": "PRIVATE_CREDENTIAL"})
    assert response.status_code == 422 and "PRIVATE" not in response.text
    response = client.post("/workbench/code/tasks", headers={"Authorization": "Bearer alice"}, json=payload)
    assert response.status_code == 400
    manager.submit.assert_not_awaited()


def test_recent_store_rejects_unbounded_or_ownerless_queries(setup):
    _, store, _ = setup
    for owner, limit in [("", 5), ("alice", 0), ("alice", 51), ("alice", True)]:
        with pytest.raises(ValueError):
            store.recent(owner, limit)


def test_disabled_code_workflow_cannot_be_activated_from_browser(setup):
    _, _, manager = setup
    app = FastAPI()

    @app.middleware("http")
    async def identity(request, call_next):
        request.state.user_id = "alice"
        return await call_next(request)

    app.include_router(create_workbench_router(manager, code_enabled=False))
    with TestClient(app) as client:
        assert client.get("/workbench/config").json()["repositories"] == []
        assert client.get("/workbench/code/tasks").status_code == 404
        assert client.post("/workbench/code/tasks", json=dict(repository="example", issue="Fix invoice totals", model="authored-model")).status_code == 404


def test_frontend_test_labels_are_not_code_retrieval_targets(tmp_path):
    from code_agent.context_evaluation import DirectoryWorkspace

    for name in ("frontend/src/App.tsx", "frontend/src/test/App.test.tsx", "frontend/src/__tests__/api.ts",
                 "frontend/src/api.spec.ts", "frontend/src/tests/labels.json"):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("authored source-policy control", encoding="utf-8")
    assert DirectoryWorkspace(tmp_path).list_files() == ["frontend/src/App.tsx"]


def test_real_service_bearer_middleware_protects_workbench_configuration(monkeypatch):
    from service import service

    monkeypatch.setattr(service, "ENABLE_USER_AUTH", True)
    monkeypatch.setattr(service, "USER_AUTH_SECRET", "authored-workbench-signing-control-" * 2)
    # No lifespan startup, worker launch, provider invocation or private store is needed for this route.
    client = TestClient(service.app)
    assert client.get("/workbench/config", headers={"X-User-ID": "alice"}).status_code == 401
    for owner in ("alice", "bob"):
        token = service._create_access_token(owner)
        response = client.get("/workbench/config", headers={"Authorization": "Bearer " + token})
        assert response.status_code == 200 and response.json()["identity"] == owner
