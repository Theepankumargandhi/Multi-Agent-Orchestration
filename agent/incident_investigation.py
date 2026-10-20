"""Fixed-budget interventional replays and review-only repair decisions."""
from __future__ import annotations

import asyncio
import random
import time

from agent.incident_models import ARMS, IncidentBundle, InvestigationPlan, configuration_for, fingerprint
from agent.incident_pipeline import prepare_replay


async def run_investigation(store, identifier: str, owner: str, replay, *, lease_token=None) -> dict:
    data = store.get(identifier, owner)
    bundle = IncidentBundle.model_validate(data["bundle"])
    plan = InvestigationPlan.model_validate(data["plan"])
    arms = ("baseline", "candidate") if data.get("parent_id") else ARMS
    rng = random.Random(plan.randomization_seed)
    # Randomized complete blocks reduce the confounding of every baseline running first.
    jobs = [(arm, repeat, configuration_for(data, arm)) for repeat in range(plan.repeats)
            for arm in rng.sample(list(arms), len(arms))]
    prepared = [(arm, repeat, config, prepare_replay(bundle, config, plan.max_completion_tokens)) for arm, repeat, config in jobs]
    reservations = sum(item.reserved_tokens for _, _, _, item in prepared)
    if len(jobs) > plan.max_calls or reservations > plan.max_reserved_tokens:
        return {"status": "budget_hold", "calls_admitted": 0, "required_calls": len(jobs), "reserved_tokens": reservations}
    # Claim before invoking the provider. Interrupted/unknown-usage runs cannot be silently replayed.
    store.claim(identifier, owner, replay.execution_kind, required_seconds=len(jobs) * (plan.timeout_seconds + 1) + 5, lease_token=lease_token)
    for arm, repeat, config, inputs in prepared:
        trial = {"trial_id": f"{arm}-{repeat}", "arm": arm, "repeat": repeat,
            "config": config.model_dump(), "config_fingerprint": inputs.config_fingerprint,
            "context_fingerprint": fingerprint(inputs.context), "document_ids": inputs.document_ids,
            "reserved_tokens": inputs.reserved_tokens, "execution_kind": replay.execution_kind,
            "preparation_ms": inputs.preparation_ms,
            "status": "started", "answer": "", "citations": [], "input_tokens": None, "output_tokens": None}
        store.start_trial(identifier, owner, trial, lease_token=lease_token)
        started = time.perf_counter()
        try:
            response = await asyncio.wait_for(replay.answer(inputs, config), plan.timeout_seconds + 1)
            if not isinstance(response.get("answer"), str) or not 0 < len(response["answer"]) <= 4000:
                raise ValueError("invalid bounded answer")
            if not set(response.get("citations", [])) <= set(inputs.document_ids):
                raise ValueError("invalid citation scope")
            usage = {key: response.get(key) for key in ("input_tokens", "output_tokens")}
            if any(value is not None and (type(value) is not int or value < 0) for value in usage.values()):
                raise ValueError("invalid provider usage")
            trial.update(answer=response["answer"], citations=response.get("citations", []), **usage, status="completed")
        except Exception as exc:
            # Stop every remaining branch; never retain provider messages or assume zero spend.
            trial.update(status="failed", error_type=type(exc).__name__)
        trial["duration_ms"] = round((time.perf_counter() - started) * 1000, 3)
        trial["fingerprint"] = fingerprint(trial)
        store.finish_trial(identifier, owner, trial, lease_token=lease_token)
        if trial["status"] != "completed":
            store.finish(identifier, owner, "incomplete", lease_token=lease_token)
            return store.report(identifier, owner)
    store.finish(identifier, owner, "awaiting_reviews", lease_token=lease_token)
    return store.report(identifier, owner)


def analyze(data: dict) -> dict:
    trials, reviews = data.get("trials", []), data.get("reviews", {})
    repeats = data["plan"]["repeats"]
    result = {"decision": "abstain", "reason": "incomplete_or_unreviewed_experiment", "supported_component": None,
              "candidate_config": None, "interpretation": "Conditional replay evidence, not a general causal proof.",
              "serving_changed": False}
    arms = ("baseline", "candidate") if data.get("parent_id") else ARMS
    if len(trials) != repeats * len(arms) or any(row["status"] != "completed" or row["trial_id"] not in reviews for row in trials):
        return result
    if any(reviews[row["trial_id"]]["trial_fingerprint"] != row["fingerprint"] for row in trials):
        return result
    if any(reviews[row["trial_id"]]["reviewer"] not in data["reviewers"] for row in trials):
        result["reason"] = "reviewer_access_revoked"
        return result
    if any(reviews[row["trial_id"]]["verdict"] == "ambiguous" for row in trials):
        result["reason"] = "ambiguous_independent_reviews"
        return result
    grouped = {arm: [reviews[row["trial_id"]] for row in trials if row["arm"] == arm] for arm in arms}
    if data.get("parent_id"):
        result["reason"] = "shadow_regression_check"
        result["candidate_safe_correct"] = all(row["safe"] and row["verdict"] == "correct" for row in grouped["candidate"])
        result["regression"] = any(base["verdict"] == "correct" and (candidate["verdict"] != "correct" or not candidate["safe"])
            for base, candidate in zip(grouped["baseline"], grouped["candidate"], strict=True))
        return result
    if not all(row["verdict"] == "incorrect" for row in grouped["baseline"]):
        result["reason"] = "baseline_failure_not_reproduced"
        return result
    improved = [arm for arm in ARMS[1:] if all(row["safe"] and row["verdict"] == "correct" for row in grouped[arm])]
    result["improving_components"] = improved
    if len(improved) != 1:
        result["reason"] = "multiple_explanations" if improved else "no_supported_intervention"
        return result
    component = improved[0]
    candidate = configuration_for(data, component).model_dump()
    base = data["bundle"]["pipeline"]
    if [key for key in base if base[key] != candidate[key]] != [component]:
        result["reason"] = "invalid_one_factor_comparison"
        return result
    result.update(supported_component=component, candidate_config=candidate,
        candidate_fingerprint=fingerprint({"base": base, "candidate": candidate, "bundle": data["bundle_fingerprint"]}))
    if data["bundle"]["origin"] == "synthetic" or data.get("execution_kind") != "live_provider":
        result["reason"] = "control_only_not_a_live_diagnosis"
    else:
        result.update(decision="supported_under_replay", reason="unique_reviewed_safe_intervention")
    return result
