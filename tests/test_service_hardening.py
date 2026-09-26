from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI
from langgraph.types import Interrupt

from evals.adaptive_router import CostAwareRouter
from service.service import _resolve_native_resume, _select_runtime_model


@pytest.mark.asyncio
async def test_native_resume_uses_durable_checkpoint_when_memory_is_empty():
    app = FastAPI()
    app.state.web_hitl_pending_cache = {}
    app.state.agent = SimpleNamespace(
        aget_state=AsyncMock(
            return_value=SimpleNamespace(
                interrupts=[],
                tasks=[SimpleNamespace(interrupts=[Interrupt(value={"kind": "web_search_approval"})])],
                values={},
            )
        )
    )
    result = await _resolve_native_resume(app, "user-a", "thread-a", "reject: too old", "gpt-4o-mini")
    assert result == {"action": "reject", "reason": "too old"}


@pytest.mark.asyncio
async def test_non_decision_never_resumes_checkpoint():
    app = FastAPI()
    app.state.web_hitl_pending_cache = {}
    app.state.agent = SimpleNamespace(aget_state=AsyncMock())
    result = await _resolve_native_resume(app, "user-a", "thread-a", "new question", "gpt-4o-mini")
    assert result is None
    app.state.agent.aget_state.assert_not_awaited()


def test_adaptive_model_router_is_opt_in_and_risk_aware(tmp_path, monkeypatch):
    from service import service

    artifact = tmp_path / "router.json"
    CostAwareRouter(
        weights=[-4.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 8.0, 0.0],
        threshold=0.5,
        training_fingerprint="test-fingerprint",
    ).save(artifact)
    monkeypatch.setattr(service, "ADAPTIVE_MODEL_ROUTER_ENABLED", True)
    monkeypatch.setattr(service, "ADAPTIVE_MODEL_ROUTER_PATH", artifact)
    monkeypatch.setattr(service, "ADAPTIVE_SMALL_MODEL", "small-model")
    monkeypatch.setattr(service, "ADAPTIVE_STRONG_MODEL", "strong-model")
    monkeypatch.setattr(service, "_adaptive_router_cache", None)

    selected, metadata = _select_runtime_model("adaptive", "say hello")
    assert selected == "small-model"
    assert metadata["selected_tier"] == "small"

    selected, metadata = _select_runtime_model("adaptive", "delete production credentials")
    assert selected == "strong-model"
    assert metadata["high_risk_override"] is True

    explicit, metadata = _select_runtime_model("chosen-model", "anything")
    assert explicit == "chosen-model"
    assert metadata["policy"] == "explicit"
