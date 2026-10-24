"""Thin browser integration: explicit repositories, owner-scoped history, no model calls on reads."""

from __future__ import annotations

import asyncio
import os
import re
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.routing import APIRoute
from pydantic import BaseModel, ConfigDict, Field, field_validator

from code_agent.job_models import CodeTaskRequest
from code_agent.repository_access import RepositoryAccess


def _owner(request: Request) -> str:
    # Identity must come from the service's verified bearer middleware, never a body/header user ID.
    user = getattr(request.state, "user_id", None)
    if not isinstance(user, str) or not re.fullmatch(r"[A-Za-z0-9._-]{3,64}", user):
        raise HTTPException(status_code=401, detail="Workbench requires per-user authentication")
    return user


Owner = Annotated[str, Depends(_owner)]


class _PrivateValidationRoute(APIRoute):
    def get_route_handler(self):
        handler = super().get_route_handler()

        async def bounded_handler(request: Request):
            try:
                response = await handler(request)
                response.headers['Cache-Control'] = 'no-store'
                response.headers['X-Content-Type-Options'] = 'nosniff'
                return response
            except RequestValidationError:
                return JSONResponse(status_code=422, content={"detail": "invalid_workbench_request"},
                                    headers={"Cache-Control": "no-store"})
            except HTTPException as exc:
                exc.headers = {**(exc.headers or {}), 'Cache-Control': 'no-store', 'X-Content-Type-Options': 'nosniff'}
                raise

        return bounded_handler


def configured_repositories() -> tuple[str, ...]:
    raw = os.getenv("WORKBENCH_REPOSITORIES", "")
    entries = tuple(item.strip() for item in raw.split(",") if item.strip())
    if len(raw) > 8000 or len(entries) > 40 or len(set(entries)) != len(entries) or any(
        len(item) > 200 or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*(?:/[A-Za-z0-9][A-Za-z0-9._-]*)*", item)
        or any(part in {".", ".."} for part in item.split("/")) for item in entries
    ):
        raise HTTPException(status_code=503, detail="Workbench repository configuration is invalid")
    return entries


def configured_coding_models() -> tuple[str, ...]:
    models = []
    for key, setting, default in (("OPENAI_API_KEY", "OPENAI_CHAT_MODEL", "gpt-4o-mini"),
                                  ("GROQ_API_KEY", "GROQ_CHAT_MODEL", "llama-3.3-70b-versatile")):
        if os.getenv(key):
            model = os.getenv(setting, default).strip()
            if not re.fullmatch(r"[A-Za-z0-9_./:-]{1,100}", model):
                raise HTTPException(status_code=503, detail="Workbench model configuration is invalid")
            models.append(model)
    return tuple(dict.fromkeys(models))


class WorkbenchTask(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, hide_input_in_errors=True)
    repository: str = Field(min_length=1, max_length=200)
    issue: str = Field(min_length=10, max_length=12000)
    model: str = Field(min_length=1, max_length=100)

    @field_validator("issue")
    @classmethod
    def meaningful_issue(cls, value: str) -> str:
        if len(value.strip()) < 10:
            raise ValueError("issue must contain at least ten meaningful characters")
        return value.strip()


def create_workbench_router(manager: Any, *, code_enabled: bool) -> APIRouter:
    router = APIRouter(prefix="/workbench", tags=["Workbench"], route_class=_PrivateValidationRoute)

    @router.get("/config")
    async def configuration(user: Owner, response: Response):
        response.headers["Cache-Control"] = "no-store"
        documents = os.getenv('WORKBENCH_DOCUMENTS_ENABLED', 'false').lower() == 'true'
        return {"version": "0.11.0", "evidence_mode": "configured_backend", "identity": user, "code_enabled": code_enabled,
                "repositories": [name for name in configured_repositories()
                                 if getattr(manager, "repository_access", RepositoryAccess()).allows(user, name)] if code_enabled else [],
                "coding_models": configured_coding_models() if code_enabled else [],
                "workflow": getattr(manager, "workflow", "unavailable"),
                "test_command": ["python", "-m", "pytest", "-q"],
                "integrations": {"documents": "local_extractive" if documents else "disabled", "evaluations": "separate_dashboard",
                                 "investigations": "api_only"}}

    @router.get("/code/tasks")
    async def history(user: Owner, response: Response, limit: int = Query(default=30, ge=1, le=50)):
        response.headers["Cache-Control"] = "no-store"
        if not code_enabled:
            raise HTTPException(status_code=404, detail="Sandboxed coding agent is disabled")
        store = getattr(manager, "store", None)
        if store is None or not callable(getattr(store, "recent", None)):
            raise HTTPException(status_code=503, detail="Durable coding history is unavailable")
        return {"tasks": await asyncio.to_thread(store.recent, user, limit)}

    @router.post("/code/tasks", status_code=202)
    async def submit(payload: WorkbenchTask, request: Request, user: Owner, response: Response):
        response.headers["Cache-Control"] = "no-store"
        if not code_enabled:
            raise HTTPException(status_code=404, detail="Sandboxed coding agent is disabled")
        if payload.repository not in configured_repositories():
            raise HTTPException(status_code=403, detail="Repository is not enabled for Workbench submission")
        if payload.model not in configured_coding_models():
            raise HTTPException(status_code=400, detail="Choose a configured coding model")
        key = request.headers.get("Idempotency-Key", "")
        if not re.fullmatch(r"[A-Za-z0-9._:-]{8,128}", key):
            raise HTTPException(status_code=400, detail="Workbench submission requires a bounded Idempotency-Key")
        # Browser clients cannot supply shell arguments, policies, images, credentials or host paths.
        record = await manager.submit(user, CodeTaskRequest(repository=payload.repository,
            issue=payload.issue, model=payload.model, test_command=["python", "-m", "pytest", "-q"]), key)
        return record.public()

    return router
