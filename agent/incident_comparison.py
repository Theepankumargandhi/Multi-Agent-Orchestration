"""Preregistered paired outcome comparisons. No serving activation or automatic correctness oracle."""
from __future__ import annotations

import random
import statistics

from agent.incident_investigation import analyze
from agent.incident_models import fingerprint
from agent.incident_store import IncidentConflict


def register_comparison(store, identifier, owner, case_ids):
    if not 3 <= len(case_ids) <= 50 or len(set(case_ids)) != len(case_ids):
        raise ValueError("three to fifty unique held-out families are required")
    with store.db() as db:
        parent_row, parent = store.load(db, identifier, owner)
        proposal = analyze(parent)
        if proposal["decision"] != "supported_under_replay" or parent.get("comparison"):
            raise IncidentConflict("a supported diagnosis and an unused comparison registration are required")
        members, families = [], set()
        for case_id in case_ids:
            row, data = store.load(db, case_id, owner)
            if (data["state"] != "captured" or data.get("job") or data["parent_id"] != identifier
                    or data["candidate_fingerprint"] != proposal["candidate_fingerprint"] or row["family"] in families
                    or data["bundle"]["origin"] == "synthetic"):
                raise IncidentConflict("held-out cases must be fresh unexecuted shadow families")
            members.append({"id": case_id, "bundle_fingerprint": data["bundle_fingerprint"], "family": row["family"]})
            families.add(row["family"])
        # Every exposed child is retained; a failed/deleted registration cannot be omitted.
        exposures = db.execute("SELECT family FROM incident_exposures WHERE tenant=? AND parent=?",
                               (parent_row["tenant"], identifier)).fetchall()
        if families != {row["family"] for row in exposures}:
            raise IncidentConflict("comparison must include every already exposed shadow family")
        protocol = {"protocol": "paired-rag-repair-v1", "candidate_fingerprint": proposal["candidate_fingerprint"],
                    "members": members, "registered_at": store.clock(), "bootstrap_seed": 71, "bootstrap_samples": 1000}
        parent["comparison"] = {**protocol, "fingerprint": fingerprint(protocol)}
        store.save(db, parent_row, parent)
    return comparison_report(store, identifier, owner)


def _percentile(values, fraction):
    if not values:
        return None
    ordered = sorted(values)
    return ordered[round((len(ordered) - 1) * fraction)]


def comparison_report(store, identifier, owner):
    with store.db() as db:
        _, parent = store.load(db, identifier, owner)
        study = parent.get("comparison")
        if not study:
            raise IncidentConflict("no frozen comparison is registered")
        protocol = {key: value for key, value in study.items() if key != "fingerprint"}
        if fingerprint(protocol) != study["fingerprint"]:
            raise IncidentConflict("comparison lineage changed")
        proposal = analyze(parent)
        holds, families, trials = [], [], []
        if proposal.get("candidate_fingerprint") != study["candidate_fingerprint"] or proposal["decision"] != "supported_under_replay":
            holds.append("diagnostic_reviews_or_candidate_unavailable")
        for member in study["members"]:
            try:
                _, data = store.load(db, member["id"], owner)
            except KeyError:
                data = None
            if (not data or data["bundle_fingerprint"] != member["bundle_fingerprint"]
                    or data.get("candidate_fingerprint") != study["candidate_fingerprint"]):
                holds.append("missing_expired_or_changed_member")
                families.append(None)
                continue
            trials.extend(data["trials"])
            rows = {arm: [] for arm in ("baseline", "candidate")}
            complete = data["state"] == "awaiting_reviews" and len(data["trials"]) == data["plan"]["repeats"] * 2
            for trial in data["trials"]:
                review = data["reviews"].get(trial["trial_id"])
                if (trial["status"] != "completed" or not review or review["reviewer"] not in data["reviewers"]
                        or review["trial_fingerprint"] != trial["fingerprint"] or review["verdict"] == "ambiguous"):
                    complete = False
                    continue
                rows[trial["arm"]].append(review)
            if (not complete or any(len(rows[arm]) != data["plan"]["repeats"] for arm in rows)):
                holds.append("missing_ambiguous_or_revoked_independent_reviews")
                families.append(None)
                continue
            if data["execution_kind"] != "live_provider":
                holds.append("control_execution_not_live_evidence")
            baseline = statistics.mean(row["verdict"] == "correct" for row in rows["baseline"])
            candidate = statistics.mean(row["verdict"] == "correct" for row in rows["candidate"])
            regressions = statistics.mean(base["verdict"] == "correct" and (other["verdict"] != "correct" or not other["safe"])
                for base, other in zip(rows["baseline"], rows["candidate"], strict=True))
            unsafe = any(not row["safe"] for row in rows["candidate"])
            if unsafe:
                holds.append("reviewed_unsafe_candidate")
            abstention = {arm: statistics.mean(row["abstained"] for row in reviews)
                if all(row.get("abstained") is not None for row in reviews) else None for arm, reviews in rows.items()}
            families.append({"baseline": baseline, "candidate": candidate, "regression": regressions, "abstention": abstention})
    total = len(families)
    judged = [row for row in families if row is not None]
    all_judged = len(judged) == total
    metrics = {}
    for arm in ("baseline", "candidate"):
        lower = sum(row[arm] for row in judged) / total
        metrics[arm] = {"correctness": lower if all_judged else None, "correctness_lower": lower,
                        "correctness_upper": lower + (total - len(judged)) / total,
                        "abstention_rate": statistics.mean(row["abstention"][arm] for row in judged)
                        if all_judged and all(row["abstention"][arm] is not None for row in judged) else None}
        arm_trials = [row for row in trials if row["arm"] == arm]
        times = [row["duration_ms"] for row in arm_trials if row.get("duration_ms") is not None]
        preparations = [row["preparation_ms"] for row in arm_trials if row.get("preparation_ms") is not None]
        known_usage = sum(row.get("input_tokens") is not None and row.get("output_tokens") is not None for row in arm_trials)
        metrics[arm]["generation_latency_ms"] = {"p50": _percentile(times, .5), "p95": _percentile(times, .95), "observations": len(times)}
        metrics[arm]["snapshot_preparation_ms"] = {"p50": _percentile(preparations, .5), "p95": _percentile(preparations, .95)}
        metrics[arm]["usage"] = {"admitted_calls": len(arm_trials), "calls_with_reported_usage": known_usage,
            "complete": known_usage == total * parent["plan"]["repeats"],
            "reported_input_tokens": sum(row.get("input_tokens") or 0 for row in arm_trials),
            "reported_output_tokens": sum(row.get("output_tokens") or 0 for row in arm_trials)}
    deltas = [row["candidate"] - row["baseline"] for row in judged]
    interval = None
    if all_judged:
        rng = random.Random(study["bootstrap_seed"])
        resamples = [statistics.mean(rng.choices(deltas, k=total)) for _ in range(study["bootstrap_samples"])]
        interval = [_percentile(resamples, .025), _percentile(resamples, .975)]
    latencies = [row["duration_ms"] for row in trials if row.get("duration_ms") is not None]
    expected_calls = sum(parent["plan"]["repeats"] * 2 for _ in families)
    reported = sum(row.get("input_tokens") is not None and row.get("output_tokens") is not None for row in trials)
    return {"protocol": study["protocol"], "study_fingerprint": study["fingerprint"], "candidate_fingerprint": study["candidate_fingerprint"],
        "decision": "hold" if holds else "reviewed_comparison_complete", "hold_reasons": sorted(set(holds)),
        "families_registered": total, "families_fully_reviewed": len(judged), "metrics": metrics,
        "paired_correctness_delta": statistics.mean(deltas) if all_judged else None,
        "family_bootstrap_95_interval": interval, "paired_regression_rate": statistics.mean(row["regression"] for row in judged) if all_judged else None,
        "latency_ms": {"p50": _percentile(latencies, .5), "p95": _percentile(latencies, .95)},
        "usage": {"expected_calls": expected_calls, "admitted_calls": len(trials), "calls_with_reported_usage": reported,
                  "complete": reported == expected_calls, "reported_input_tokens": sum(row.get("input_tokens") or 0 for row in trials),
                  "reported_output_tokens": sum(row.get("output_tokens") or 0 for row in trials), "billed_cost_usd": None},
        "interpretation": "Descriptive paired family results, not production certification. Accounts do not establish reviewer independence or truthful labels.",
        "serving_changed": False}
