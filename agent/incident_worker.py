"""Leased, single-host incident workers; never retry an admitted model call after interruption."""
from __future__ import annotations

import argparse
import asyncio
import json
import os

from agent.incident_investigation import run_investigation
from agent.incident_models import IncidentBundle, InvestigationPlan
from agent.incident_pipeline import ExtractiveControlReplay, LiveIncidentReplay, RuntimeTextReplay
from agent.incident_store import IncidentConflict, IncidentStore


def configured_store():
    if os.getenv("INCIDENT_INVESTIGATION_ENABLED", "false").lower() != "true":
        raise ValueError("private investigation is disabled")
    return IncidentStore(os.getenv("INCIDENT_DB_PATH", "data/incidents/private.db"),
                         bytes.fromhex(os.getenv("INCIDENT_ENCRYPTION_KEY", "")))


def replay_for(data, kind):
    if kind == "extractive_control":
        return ExtractiveControlReplay()
    if any(os.getenv(flag, "false").lower() != "true" for flag in
           ("INCIDENT_LIVE_REPLAY_ENABLED", "INCIDENT_WORKER_LIVE_ENABLED")):
        raise ValueError("live worker inference is not operator-enabled")
    bundle = IncidentBundle.model_validate(data["bundle"])
    plan = InvestigationPlan.model_validate(data["plan"])
    return RuntimeTextReplay(bundle, plan) if bundle.runtime_rag else LiveIncidentReplay(bundle.model, plan)


async def run_once(store, *, replay_factory=replay_for, lease_seconds=60):
    lease = await asyncio.to_thread(store.claim_next, lease_seconds=lease_seconds)
    if not lease:
        return None
    identifier, owner, token = lease["id"], lease["owner"], lease["token"]
    heartbeat_error = []
    task = None
    async def heartbeat():
        while True:
            await asyncio.sleep(min(5, lease_seconds / 3))
            try:
                await asyncio.to_thread(store.heartbeat, identifier, owner, token, lease_seconds=lease_seconds)
            except (IncidentConflict, KeyError):
                heartbeat_error.append(True)
                if task:
                    task.cancel()
                return
    pulse = None
    state = "held_before_execution"
    try:
        data = await asyncio.to_thread(store.get, identifier, owner)
        replay = replay_factory(data, lease["kind"])
        task = asyncio.create_task(run_investigation(store, identifier, owner, replay, lease_token=token))
        pulse = asyncio.create_task(heartbeat())
        result = await task
        state = "budget_hold" if result.get("status") == "budget_hold" else (
            "completed" if result.get("state") == "awaiting_reviews" else "incomplete")
    except asyncio.CancelledError:
        state = "interrupted_unknown_usage"
        if not heartbeat_error:
            raise
    except Exception:
        # Provider exception text and private payloads must never reach worker logs.
        state = "incomplete"
    finally:
        if task and not task.done():
            task.cancel()
        if pulse:
            pulse.cancel()
            await asyncio.gather(pulse, return_exceptions=True)
        try:
            await asyncio.to_thread(store.finish_job, identifier, owner, token, state)
        except (IncidentConflict, KeyError):
            state = "lease_lost_or_case_unavailable"
    return {"id": identifier, "worker_outcome": state, "provider_messages_logged": False}


async def serve(store, once=False):
    while True:
        result = await run_once(store)
        if result:
            print(json.dumps(result), flush=True)
        if once:
            return
        if result is None:
            await asyncio.sleep(1)


def main():
    from dotenv import load_dotenv
    load_dotenv()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    asyncio.run(serve(configured_store(), args.once))


if __name__ == "__main__":
    main()
