from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI
from langgraph.types import Interrupt

from service.service import _resolve_native_resume


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
