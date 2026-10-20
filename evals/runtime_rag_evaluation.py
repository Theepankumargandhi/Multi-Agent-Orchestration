"""Authored actual-SQLite-RAG and durable-worker controls; zero live provider calls or human labels."""
from __future__ import annotations

import argparse
import asyncio
import json
import secrets
import tempfile
from pathlib import Path

from langchain_core.messages import AIMessage

from agent import local_rag
from agent.incident_models import TrialReview
from agent.incident_runtime import RuntimeCapture
from agent.incident_store import IncidentStore
from agent.incident_worker import run_once


async def evaluate():
    outcomes = []
    saved = local_rag.USE_CHROMA_RAG, local_rag._redis_disabled, local_rag._local_rag_cache
    local_rag.USE_CHROMA_RAG, local_rag._redis_disabled, local_rag._local_rag_cache = False, True, {}
    try:
        with tempfile.TemporaryDirectory(prefix="runtime-rag-controls-") as temporary:
            root = Path(temporary)
            key = secrets.token_bytes(32)  # Ephemeral control key; never emitted.
            vault = IncidentStore(root / "incidents.db", key)
            for case, product, text, reason in (
                ("packing-loss", "telescope", "Background information. " * 40 + "\nThe telescope request limit for authenticated users is 42.", "control_only_not_a_live_diagnosis"),
                ("already-correct", "cache", "The cache request limit for authenticated users is 42.", "baseline_failure_not_reproduced"),
                ("missing-evidence", "billing", "Billing request limits for authenticated users are not specified.", "no_supported_intervention"),
            ):
                path = str(root / (case + ".db"))
                local_rag.add_local_document("Request limits", text, source="authored://" + case, db_path=path)
                query = f"{product} request limit authenticated users"
                capture = RuntimeCapture("control-owner", case, "authored-no-provider", vault)
                with capture.scope():
                    notes = local_rag.search_local_knowledge(query, limit=1, db_path=path)
                    user = "Question: " + query + "\nLocal RAG evidence:\n" + notes
                    capture.before_generation("Answer from evidence.", user, {"route": "rag"})
                    capture.model_response("Answer from evidence.", user, AIMessage(content="Not measured: original generation is a control."), "groq:authored-not-a-model")
                receipt = capture.finish({"route": "rag"}, "Not measured: original generation is a control.")
                identifier = receipt["id"]
                vault.enqueue(identifier, "control-owner", "extractive_control")
                # Reopen storage to exercise persistence across the enqueue/worker boundary.
                completed = await run_once(IncidentStore(root / "incidents.db", key))
                vault.grant(identifier, "control-owner", "simulated-labels")
                for trial in vault.get(identifier, "control-owner")["trials"]:
                    correct = "42" in trial["answer"]
                    vault.review(identifier, "simulated-labels", TrialReview(trial_id=trial["trial_id"],
                        trial_fingerprint=trial["fingerprint"], verdict="correct" if correct else "incorrect", safe=True,
                        abstained=None, rationale="Authored exact-fact assertion, not an independent human review."))
                report = vault.report(identifier, "control-owner")
                outcomes.append({"case": case, "backend": "sqlite_overlap", "calls_admitted": report["calls_admitted"],
                    "diagnosis_reason": report["analysis"]["reason"], "worker_outcome": completed["worker_outcome"],
                    "passed": report["analysis"]["decision"] == "abstain" and report["analysis"]["reason"] == reason
                              and completed["worker_outcome"] == "completed" and report["calls_admitted"] == 10})
    finally:
        local_rag.USE_CHROMA_RAG, local_rag._redis_disabled, local_rag._local_rag_cache = saved
    return {"evidence_kind": "authored_real_sqlite_rag_and_durable_worker_controls", "live_provider_calls": 0,
        "human_reviews": 0, "live_accuracy_measured": False, "production_impact_measured": False,
        "chroma_embedding_service_exercised": False, "total": len(outcomes), "passed": sum(row["passed"] for row in outcomes),
        "outcomes": outcomes}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = asyncio.run(evaluate())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if report["passed"] != report["total"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
