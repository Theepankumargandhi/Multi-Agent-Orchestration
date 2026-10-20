"""Opt-in authenticated investigation API; no automatic production activation."""
from __future__ import annotations

import os
from functools import lru_cache
from typing import Literal

from fastapi import APIRouter, HTTPException, Request
from pydantic import Field

from agent.incident_models import IncidentBundle, InvestigationPlan, StrictModel, TrialReview

router = APIRouter(prefix="/investigations", tags=["AI failure investigations"])


class CaptureRequest(StrictModel):
    bundle: IncidentBundle
    consent: bool
    ttl_seconds: int = Field(default=86400, ge=60, le=604800)
    plan: InvestigationPlan = Field(default_factory=InvestigationPlan)
    parent_id: str | None = Field(default=None, max_length=64)


class ReviewerRequest(StrictModel):
    reviewer: str = Field(min_length=1, max_length=200)


class GrantRequest(ReviewerRequest):
    consent_to_share_content: bool


class RunRequest(StrictModel):
    execution: Literal["extractive_control", "live_provider"] = "extractive_control"
    authorize_provider_requests: bool = False


class ComparisonRequest(StrictModel):
    case_ids: list[str] = Field(min_length=3, max_length=50)


@lru_cache(maxsize=1)
def _configured_store(path, key):
    from agent.incident_store import IncidentStore
    return IncidentStore(path, bytes.fromhex(key))


def context(request: Request):
    if os.getenv("INCIDENT_INVESTIGATION_ENABLED", "false").lower() != "true":
        raise HTTPException(404, "Incident investigation is disabled")
    actor = str(getattr(request.state, "user_id", "") or "")
    if not actor:
        raise HTTPException(401, "Authenticated user mode is required")
    try:
        store = _configured_store(os.getenv("INCIDENT_DB_PATH", "data/incidents/private.db"), os.getenv("INCIDENT_ENCRYPTION_KEY", ""))
    except (ValueError, OSError):
        raise HTTPException(503, "Private incident storage is not configured") from None
    return store, actor


def checked(call):
    from agent.incident_store import IncidentConflict
    try:
        return call()
    except KeyError:
        raise HTTPException(404, "Investigation is unavailable") from None
    except IncidentConflict:
        raise HTTPException(409, "Investigation state, review or integrity conflict") from None
    except ValueError:
        raise HTTPException(400, "Investigation input or consent is invalid") from None


@router.post("", status_code=201)
async def capture(payload: CaptureRequest, request: Request):
    store, actor = context(request)
    if payload.bundle.origin == "runtime":
        raise HTTPException(400, "Client-supplied snapshots must use operator or synthetic origin")
    identifier = checked(lambda: store.create(actor, payload.bundle, payload.plan, consent=payload.consent,
        ttl_seconds=payload.ttl_seconds, parent_id=payload.parent_id))
    return store.report(identifier, actor)


@router.get("/{identifier}")
async def report(identifier: str, request: Request):
    store, actor = context(request)
    return checked(lambda: store.report(identifier, actor))


@router.get("/{identifier}/private-review-bundle")
async def private_review_bundle(identifier: str, request: Request):
    store, actor = context(request)
    return checked(lambda: store.review_bundle(identifier, actor))


@router.post("/{identifier}/reviewers")
async def grant(identifier: str, payload: GrantRequest, request: Request):
    store, actor = context(request)
    if payload.consent_to_share_content is not True:
        raise HTTPException(400, "Sharing requires explicit reproduction-content consent")
    checked(lambda: store.grant(identifier, actor, payload.reviewer))
    return {"granted": True}


@router.post("/{identifier}/run", status_code=202)
async def run(identifier: str, payload: RunRequest, request: Request):
    store, actor = context(request)
    checked(lambda: store.get(identifier, actor))
    if payload.execution == "live_provider":
        if not payload.authorize_provider_requests or os.getenv("INCIDENT_LIVE_REPLAY_ENABLED", "false").lower() != "true":
            raise HTTPException(403, "Live replay requires operator enablement and explicit request authorization")
    return checked(lambda: store.enqueue(identifier, actor, payload.execution, authorized_live=payload.authorize_provider_requests))


@router.post("/{identifier}/cancel")
async def cancel(identifier: str, request: Request):
    store, actor = context(request)
    return checked(lambda: store.cancel(identifier, actor))


@router.post("/{identifier}/comparison", status_code=201)
async def register_comparison(identifier: str, payload: ComparisonRequest, request: Request):
    from agent.incident_comparison import register_comparison
    store, actor = context(request)
    return checked(lambda: register_comparison(store, identifier, actor, payload.case_ids))


@router.get("/{identifier}/comparison")
async def comparison(identifier: str, request: Request):
    from agent.incident_comparison import comparison_report
    store, actor = context(request)
    return checked(lambda: comparison_report(store, identifier, actor))


@router.post("/{identifier}/reviewers/revoke")
async def revoke(identifier: str, payload: ReviewerRequest, request: Request):
    store, actor = context(request)
    checked(lambda: store.revoke(identifier, actor, payload.reviewer))
    return {"revoked": True}


@router.post("/{identifier}/reviews")
async def review(identifier: str, payload: TrialReview, request: Request):
    store, actor = context(request)
    checked(lambda: store.review(identifier, actor, payload))
    return {"review_recorded": True}


@router.get("/{identifier}/shadow-gate")
async def shadow_gate(identifier: str, request: Request):
    store, actor = context(request)
    return checked(lambda: store.shadow_gate(identifier, actor))


@router.get("/{identifier}/regression-case")
async def regression_case(identifier: str, request: Request):
    store, actor = context(request)
    data = checked(lambda: store.get(identifier, actor))
    return {"kind": "private_reviewed_case_not_an_automatic_correctness_oracle", "bundle": data["bundle"],
            "bundle_fingerprint": data["bundle_fingerprint"], "reviews": data["reviews"],
            "plan": data["plan"], "analysis": store.report(identifier, actor)["analysis"]}


@router.delete("/{identifier}")
async def delete(identifier: str, request: Request):
    store, actor = context(request)
    checked(lambda: store.delete(identifier, actor))
    return {"deleted": True, "note": "Logical deletion; externally retained backups require separate retention controls."}
