"""Actual authenticated FastAPI journeys using the provider-free replay control."""
import asyncio
import importlib
import secrets

import pytest
from fastapi.testclient import TestClient

from tests.test_incident_investigation import bundle


@pytest.fixture
def client(tmp_path, monkeypatch):
    service = importlib.import_module("service.service")
    monkeypatch.setenv("INCIDENT_INVESTIGATION_ENABLED", "true")
    monkeypatch.setenv("INCIDENT_LIVE_REPLAY_ENABLED", "false")
    monkeypatch.setenv("INCIDENT_ENCRYPTION_KEY", secrets.token_hex(32))
    monkeypatch.setenv("INCIDENT_DB_PATH", str(tmp_path / "private-incidents.db"))
    monkeypatch.setattr(service, "ENABLE_USER_AUTH", True)
    monkeypatch.setattr(service, "USER_AUTH_SECRET", secrets.token_hex(32))
    monkeypatch.setattr(service, "CHECKPOINT_DB_PATH", str(tmp_path / "checkpoints.db"))
    monkeypatch.setattr(service, "STORE_DB_PATH", str(tmp_path / "conversations.db"))
    monkeypatch.setattr(service, "POSTGRES_CHECKPOINT_URI", "")
    monkeypatch.setattr(service, "POSTGRES_STORE_URI", "")
    with TestClient(service.app) as client:
        headers = {}
        for name in ("incident-alice", "incident-bob", "incident-mallory"):
            registration = client.post("/auth/register", json={"user_id": name, "password": secrets.token_urlsafe(16)})
            assert registration.status_code == 200
            headers[name] = {"Authorization": "Bearer " + registration.json()["access_token"]}
        yield client, headers


def test_authenticated_capture_review_export_and_delete_journey(client):
    http, headers = client
    owner, reviewer, outsider = (headers[name] for name in ("incident-alice", "incident-bob", "incident-mallory"))
    payload = {"bundle": bundle(origin="operator").model_dump(), "consent": True}
    assert http.post("/investigations", json=payload).status_code == 401
    assert http.post("/investigations", json={**payload, "consent": False}, headers=owner).status_code == 400
    created = http.post("/investigations", json=payload, headers=owner)
    assert created.status_code == 201
    route = "/investigations/" + created.json()["id"]
    assert http.get(route, headers=outsider).status_code == 404
    assert http.get(route + "/private-review-bundle", headers=reviewer).status_code == 404
    assert http.post(route + "/reviewers", json={"reviewer": "incident-bob", "consent_to_share_content": False}, headers=owner).status_code == 400
    assert http.post(route + "/reviewers", json={"reviewer": "incident-bob", "consent_to_share_content": True}, headers=owner).status_code == 200
    # Normal run authorization never permits a silent billable provider call.
    assert http.post(route + "/run", json={"execution": "live_provider", "authorize_provider_requests": True}, headers=owner).status_code == 403
    ran = http.post(route + "/run", json={"execution": "extractive_control"}, headers=owner)
    assert ran.status_code == 202 and ran.json()["calls_admitted"] == 0 and ran.json()["job"]["state"] == "queued"
    assert "telescope" not in ran.text
    assert http.post(route + "/run", json={}, headers=owner).status_code == 409
    from agent.incident_worker import configured_store, run_once
    assert asyncio.run(run_once(configured_store()))["worker_outcome"] == "completed"
    assert http.get(route, headers=owner).json()["calls_admitted"] == 10
    private = http.get(route + "/private-review-bundle", headers=reviewer)
    assert private.status_code == 200 and private.json()["bundle"]["query"] == payload["bundle"]["query"]
    for trial in private.json()["trials"]:
        label = {"trial_id": trial["trial_id"], "trial_fingerprint": trial["fingerprint"],
            "verdict": "correct" if trial["arm"] == "packing" else "incorrect", "safe": True,
            "rationale": "Authored endpoint control, not real reviewed production data"}
        assert http.post(route + "/reviews", json=label, headers=owner).status_code == 409
        assert http.post(route + "/reviews", json=label, headers=reviewer).status_code == 200
    report = http.get(route, headers=owner).json()
    assert report["analysis"]["reason"] == "control_only_not_a_live_diagnosis"
    assert http.get(route + "/shadow-gate", headers=owner).json()["decision"] == "hold"
    assert http.get(route + "/regression-case", headers=owner).status_code == 200
    assert http.get(route + "/regression-case", headers=outsider).status_code == 404
    assert http.delete(route, headers=outsider).status_code == 404
    assert http.delete(route, headers=owner).status_code == 200
    assert http.get(route, headers=owner).status_code == 404


def test_disabled_unconfigured_strict_consent_and_runtime_origin(client, monkeypatch):
    http, headers = client
    owner = headers["incident-alice"]
    payload = {"bundle": bundle().model_dump(), "consent": True}
    monkeypatch.setenv("INCIDENT_INVESTIGATION_ENABLED", "false")
    assert http.post("/investigations", json=payload, headers=owner).status_code == 404
    monkeypatch.setenv("INCIDENT_INVESTIGATION_ENABLED", "true")
    monkeypatch.setenv("INCIDENT_ENCRYPTION_KEY", "invalid")
    assert http.post("/investigations", json=payload, headers=owner).status_code == 503
    monkeypatch.setenv("INCIDENT_ENCRYPTION_KEY", secrets.token_hex(32))
    assert http.post("/investigations", json={**payload, "consent": "true"}, headers=owner).status_code == 422
    payload["bundle"]["origin"] = "runtime"
    assert http.post("/investigations", json=payload, headers=owner).status_code == 400
