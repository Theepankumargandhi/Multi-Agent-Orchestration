"""Provider-free controlled RAG incidents; not live diagnosis accuracy or production lift."""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import tempfile
from pathlib import Path

from agent.incident_investigation import run_investigation
from agent.incident_models import (
    IncidentBundle,
    IncidentDocument,
    IncidentPipeline,
    InvestigationPlan,
    TrialReview,
)
from agent.incident_pipeline import ExtractiveControlReplay
from agent.incident_store import IncidentStore


def document(identifier: str, text: str) -> IncidentDocument:
    return IncidentDocument(document_id=identifier, text=text, version_sha256=hashlib.sha256(text.encode()).hexdigest())


def cases():
    query = "What is the telescope request limit for authenticated users?"
    good = document("manual", "The telescope request limit for authenticated users is 42.")
    bad = document("background", "Background information. " * 20)
    common = dict(origin="synthetic", query=query, observed_answer="Background information.",
                  model="groq:authored-control-not-a-live-model", prompt="Answer from supplied documents.", application_revision="controlled-v1")
    return [
        ("retrieval-miss", "retrieval", IncidentBundle(**common, family="retrieval-miss", documents=[bad, good],
            retrieved_ids=["background"], ranked_ids=["background"], pipeline=IncidentPipeline(top_k=1, context_chars=180))),
        ("reranking-loss", "reranking", IncidentBundle(**common, family="reranking-loss", documents=[bad, good],
            retrieved_ids=["background", "manual"], ranked_ids=["background", "manual"],
            pipeline=IncidentPipeline(retrieval="lexical", top_k=2, context_chars=180))),
        ("packing-loss", "packing", IncidentBundle(**common, family="packing-loss",
            documents=[document("manual", bad.text + "\n" + good.text)], retrieved_ids=["manual"], ranked_ids=["manual"],
            pipeline=IncidentPipeline(context_chars=180))),
        ("baseline-already-correct", None, IncidentBundle(**common, family="baseline-correct", documents=[good],
            retrieved_ids=["manual"], ranked_ids=["manual"], pipeline=IncidentPipeline(context_chars=180))),
        ("no-answer-in-snapshot", None, IncidentBundle(**common, family="missing-answer", documents=[bad],
            retrieved_ids=["background"], ranked_ids=["background"], pipeline=IncidentPipeline(context_chars=180))),
    ]


async def evaluate():
    rows = []
    with tempfile.TemporaryDirectory(prefix="agentforge-incident-controls-") as temporary:
        store = IncidentStore(Path(temporary) / "private.db", b"synthetic-control-key-only-123456"[:32].ljust(32, b"0"))
        for name, expected, bundle in cases():
            identifier = store.create("control-owner", bundle, InvestigationPlan(), consent=True)
            await run_investigation(store, identifier, "control-owner", ExtractiveControlReplay())
            store.grant(identifier, "control-owner", "authored-label-simulator")
            for trial in store.get(identifier, "control-owner")["trials"]:
                store.review(identifier, "authored-label-simulator", TrialReview(trial_id=trial["trial_id"],
                    trial_fingerprint=trial["fingerprint"], verdict="correct" if "42" in trial["answer"] else "incorrect",
                    safe=True, rationale="Controlled exact-fact assertion; simulated label, not an independent human review."))
            report = store.report(identifier, "control-owner")
            analysis = report["analysis"]
            rows.append({"case": name, "expected_component": expected, "observed_component": analysis["supported_component"],
                "reason": analysis["reason"], "calls_admitted": report["calls_admitted"],
                "passed": analysis["supported_component"] == expected and analysis["decision"] == "abstain"
                          and report["calls_admitted"] == 10 and not report["serving_changed"]})
    return {"evidence_kind": "authored_snapshot_and_extractive_controls", "live_provider_calls": 0,
        "human_reviews": 0, "live_diagnosis_accuracy_measured": False, "production_impact_measured": False,
        "total": len(rows), "passed": sum(row["passed"] for row in rows), "outcomes": rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = asyncio.run(evaluate())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    if result["passed"] != result["total"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
