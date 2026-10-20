"""Durable admission, fencing, cancellation and unknown-spend recovery controls."""
import asyncio

import pytest

from agent.incident_investigation import run_investigation
from agent.incident_models import InvestigationPlan
from agent.incident_pipeline import ExtractiveControlReplay
from agent.incident_store import IncidentConflict
from agent.incident_worker import replay_for, run_once
from tests.test_incident_investigation import bundle, store


def queued(vault):
    identifier = vault.create("alice", bundle(), InvestigationPlan(), consent=True)
    vault.enqueue(identifier, "alice", "extractive_control")
    return identifier


@pytest.mark.asyncio
async def test_worker_reopens_store_completes_and_submission_is_not_repeatable(tmp_path):
    vault = store(tmp_path)
    identifier = queued(vault)
    with pytest.raises(IncidentConflict):
        vault.enqueue(identifier, "alice", "extractive_control")
    result = await run_once(store(tmp_path))
    assert result["worker_outcome"] == "completed"
    report = vault.report(identifier, "alice")
    assert report["calls_admitted"] == 10 and report["job"]["state"] == "completed"
    assert "owner" not in report["job"] and "lease_token" not in report["job"]
    assert await run_once(vault) is None


def test_expired_unstarted_lease_is_reclaimed_but_started_work_never_retries(tmp_path):
    now = [1000.0]
    vault = store(tmp_path, clock=lambda: now[0])
    identifier = queued(vault)
    first = vault.claim_next(lease_seconds=5)
    assert vault.claim_next(lease_seconds=5) is None
    now[0] += 6
    second = vault.claim_next(lease_seconds=5)
    assert second["id"] == identifier and first["token"] != second["token"]
    with pytest.raises(IncidentConflict):
        vault.claim(identifier, "alice", "extractive_control", lease_token=first["token"])
    vault.claim(identifier, "alice", "extractive_control", lease_token=second["token"])
    now[0] += 6
    assert vault.claim_next() is None
    assert vault.report(identifier, "alice")["job"]["state"] == "interrupted_unknown_usage"
    with pytest.raises(IncidentConflict):
        vault.finish_job(identifier, "alice", second["token"], "completed")


@pytest.mark.asyncio
async def test_queued_cancel_and_inflight_cancel_preserve_unknown_usage(tmp_path):
    vault = store(tmp_path)
    identifier = queued(vault)
    assert vault.cancel(identifier, "alice")["job"]["state"] == "cancelled"
    assert await run_once(vault) is None
    identifier = queued(vault)
    started = asyncio.Event()
    class Slow(ExtractiveControlReplay):
        async def answer(self, prepared, config):
            started.set()
            await asyncio.sleep(30)
    task = asyncio.create_task(run_once(vault, replay_factory=lambda *_: Slow(), lease_seconds=5))
    await asyncio.wait_for(started.wait(), 10)
    vault.cancel(identifier, "alice")
    await asyncio.wait_for(task, 10)
    report = vault.report(identifier, "alice")
    assert report["job"]["state"] == "cancelled" and report["calls_admitted"] == 1 and not report["usage_complete"]
    assert vault.get(identifier, "alice")["trials"][0]["status"] == "started"
    assert await run_once(vault) is None


def test_live_worker_requires_separate_operator_enablement(tmp_path, monkeypatch):
    monkeypatch.setenv("INCIDENT_WORKER_LIVE_ENABLED", "false")
    with pytest.raises(ValueError):
        replay_for({"bundle": bundle().model_dump()}, "live_provider")


@pytest.mark.asyncio
async def test_queued_job_cannot_bypass_lease_and_budget_hold_makes_zero_calls(tmp_path):
    vault = store(tmp_path)
    identifier = queued(vault)
    with pytest.raises(IncidentConflict):
        await run_investigation(vault, identifier, "alice", ExtractiveControlReplay())
    other = vault.create("alice", bundle(family="budget"), InvestigationPlan(max_calls=1), consent=True)
    vault.enqueue(other, "alice", "extractive_control")
    vault.cancel(identifier, "alice")
    assert (await run_once(vault))["worker_outcome"] == "budget_hold"
    assert vault.report(other, "alice")["calls_admitted"] == 0


@pytest.mark.asyncio
async def test_external_worker_interruption_does_not_requeue_started_call(tmp_path):
    vault = store(tmp_path)
    identifier = queued(vault)
    started = asyncio.Event()
    class Slow(ExtractiveControlReplay):
        async def answer(self, prepared, config):
            started.set()
            await asyncio.sleep(30)
    task = asyncio.create_task(run_once(vault, replay_factory=lambda *_: Slow()))
    await asyncio.wait_for(started.wait(), 10)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    report = vault.report(identifier, "alice")
    assert report["job"]["state"] == "interrupted_unknown_usage" and report["calls_admitted"] == 1
    assert await run_once(vault) is None


@pytest.mark.asyncio
async def test_worker_factory_failure_is_terminal_and_serve_once_exits(tmp_path):
    from agent.incident_worker import serve
    vault = store(tmp_path)
    identifier = queued(vault)
    def unavailable(*_):
        raise ValueError("private provider failure must not be logged")
    result = await run_once(vault, replay_factory=unavailable)
    assert result["worker_outcome"] == "incomplete"
    assert vault.report(identifier, "alice")["calls_admitted"] == 0
    assert await serve(vault, once=True) is None
