"""Unified read-only AgentOps reliability, evaluation, and trace command center."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Callable

import streamlit as st

RESULTS_ROOT = Path(os.getenv("AGENTOPS_RESULTS_DIR", "data/evaluations"))
DATASETS_ROOT = Path(os.getenv("AGENTOPS_DATASETS_DIR", "evals/datasets"))


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def latest_report(
    root: Path, predicate: Callable[[dict[str, Any]], bool]
) -> tuple[Path, dict[str, Any]] | None:
    matches: list[tuple[Path, dict[str, Any]]] = []
    if not root.exists():
        return None
    for path in root.rglob("*.json"):
        payload = _read_json(path)
        if payload is not None and predicate(payload):
            matches.append((path, payload))
    return max(matches, key=lambda item: item[0].stat().st_mtime) if matches else None


def load_trace_records(roots: list[Path], limit: int = 200) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    paths: list[Path] = []
    for root in roots:
        if root.is_file():
            paths.append(root)
        elif root.exists():
            paths.extend(root.rglob("*.jsonl"))
    for path in sorted(set(paths), key=lambda item: item.stat().st_mtime, reverse=True):
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except OSError:
            continue
        for line in reversed(lines):
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(item, dict) and isinstance(item.get("spans"), list):
                item["_source"] = path.as_posix()
                records.append(item)
                if len(records) >= limit:
                    return records
    return records


def trace_rows(trace: dict[str, Any]) -> list[dict[str, Any]]:
    spans = trace.get("spans") or []
    starts = [int(item.get("start_time_unix_nano", 0)) for item in spans]
    origin = min(starts) if starts else 0
    return [
        {
            "span": item.get("name", "unknown"),
            "kind": item.get("kind", "unknown"),
            "status": item.get("status", "unknown"),
            "offset_ms": round(
                (int(item.get("start_time_unix_nano", origin)) - origin) / 1_000_000,
                3,
            ),
            "duration_ms": round(float(item.get("duration_ms", 0)), 3),
            "parent": item.get("parent_span_id") or "root",
        }
        for item in spans
    ]


def trace_dot(trace: dict[str, Any]) -> str:
    spans = trace.get("spans") or []
    node_by_span = {
        str(span.get("span_id")): f"n{index}" for index, span in enumerate(spans)
    }
    lines = ["digraph trace {", "rankdir=LR;", "node [shape=box, style=rounded];"]
    for index, span in enumerate(spans):
        label = json.dumps(
            f"{span.get('name', 'unknown')}\n{float(span.get('duration_ms', 0)):.2f} ms"
        )
        color = "#ff6b6b" if span.get("status") == "error" else "#4dabf7"
        lines.append(f'n{index} [label={label}, color="{color}"];')
    for index, span in enumerate(spans):
        parent = node_by_span.get(str(span.get("parent_span_id") or ""))
        if parent:
            lines.append(f"{parent} -> n{index};")
    lines.append("}")
    return "\n".join(lines)


def incident_summary(outcome: dict[str, Any]) -> str:
    run = outcome.get("resilient") or {}
    events = run.get("events") or []
    controls = [item.get("action") for item in events if item.get("phase") == "control"]
    control = ", ".join(str(item) for item in controls) or "no recovery action"
    return (
        f"Injected {outcome.get('fault', 'unknown')} into {outcome.get('component', 'unknown')}; "
        f"the runtime applied {control} and finished {run.get('status', 'unknown')} in "
        f"{float(run.get('duration_ms', 0)):.1f} ms across {int(run.get('attempts', 0))} attempt(s)."
    )


def _metric(container, label: str, value: object, delta: object | None = None) -> None:
    container.metric(label, value, delta=delta)


def _render_overview(artifacts: dict[str, tuple[Path, dict[str, Any]] | None]) -> None:
    st.header("Release readiness")
    reliability = artifacts["reliability"][1] if artifacts["reliability"] else {}
    security = artifacts["security"][1] if artifacts["security"] else {}
    retrieval = artifacts["retrieval"][1] if artifacts["retrieval"] else {}
    telemetry = artifacts["telemetry"][1] if artifacts["telemetry"] else {}
    arena = artifacts["arena"][1] if artifacts["arena"] else {}
    gateway = artifacts["model_gateway"][1] if artifacts["model_gateway"] else {}
    online = artifacts["online_monitor"][1] if artifacts["online_monitor"] else {}
    memory = artifacts["memory"][1] if artifacts["memory"] else {}
    grounding = artifacts["grounding"][1] if artifacts["grounding"] else {}
    defended = security.get("defended") or {}
    arena_variants = {
        (item.get("variant") or {}).get("name"): item for item in arena.get("variants") or []
    }
    arena_candidate = arena_variants.get((arena.get("promotion") or {}).get("candidate")) or {}
    columns = st.columns(11)
    _metric(
        columns[0],
        "Reliability gate",
        f"{float((reliability.get('resilient') or {}).get('scenario_pass_rate', 0)):.1%}",
    )
    _metric(
        columns[1],
        "Recovery success",
        f"{float(reliability.get('recovery_success_rate', 0)):.1%}",
    )
    _metric(columns[2], "Security ASR", f"{float(defended.get('attack_success_rate', 0)):.1%}")
    _metric(columns[3], "Retrieval Recall@K", f"{float(retrieval.get('recall_at_k', 0)):.1%}")
    _metric(
        columns[4],
        "Valid traces",
        f"{float(telemetry.get('valid_trace_rate', 0)):.1%}",
    )
    _metric(
        columns[5],
        "Telemetry leakage",
        int(telemetry.get("content_attribute_violations", 0))
        + int(telemetry.get("secret_value_violations", 0)),
    )
    _metric(
        columns[6],
        "Arena promotion",
        "PASS" if (arena.get("promotion") or {}).get("approved") else "NOT READY",
        f"{float(arena_candidate.get('pass_rate', 0)):.1%} scenarios",
    )
    _metric(
        columns[7],
        "Gateway controls",
        f"{float(gateway.get('pass_rate', 0)):.1%}",
        f"{int(gateway.get('privacy_violations', 0))} privacy violations",
    )
    _metric(
        columns[8],
        "Online canary",
        str((online.get("canary_decision") or {}).get("action", "NO DATA")).upper(),
        f"{len(online.get('alerts') or [])} SLO alerts",
    )
    _metric(
        columns[9],
        "Memory controls",
        f"{float(memory.get('pass_rate', 0)):.1%}",
        f"{float(memory.get('cross_tenant_leakage_rate', 0)):.1%} tenant leakage",
    )
    _metric(
        columns[10],
        "Grounding gate",
        f"{float(grounding.get('pass_rate', 0)):.1%}",
        f"{float(grounding.get('unsafe_release_rate', 0)):.1%} unsafe release",
    )
    readiness_rows = []
    for name, artifact in artifacts.items():
        readiness_rows.append(
            {
                "evidence": name,
                "available": artifact is not None,
                "artifact": artifact[0].as_posix() if artifact else "not generated",
                "fingerprint": (
                    str(
                        artifact[1].get("report_fingerprint")
                        or artifact[1].get("dataset_fingerprint")
                        or artifact[1].get("dataset_sha256")
                        or artifact[1].get("matrix_fingerprint")
                        or ""
                    )[:16]
                    if artifact
                    else ""
                ),
            }
        )
    st.dataframe(readiness_rows, width="stretch", hide_index=True)
    st.caption(
        "Green local gates are regression evidence, not a claim of external production scale. "
        "Run live traffic and official benchmark evaluators before publishing those claims."
    )


def arena_outcome_rows(report: dict[str, Any]) -> list[dict[str, Any]]:
    """Flatten arena scenario evidence for tables and lightweight tests."""
    rows: list[dict[str, Any]] = []
    for variant in report.get("variants") or []:
        name = (variant.get("variant") or {}).get("name", "unknown")
        for outcome in variant.get("outcomes") or []:
            rows.append(
                {
                    "variant": name,
                    "scenario": outcome.get("scenario_id"),
                    "domain": outcome.get("domain"),
                    "quality": round(float(outcome.get("quality_score", 0)), 3),
                    "passed": bool(outcome.get("passed")),
                    "safety failed": bool(outcome.get("safety_failed")),
                    "turns": outcome.get("turns_executed", 0),
                    "tokens": outcome.get("tokens", 0),
                    "latency ms": round(float(outcome.get("latency_ms", 0)), 2),
                    "trajectory": str(outcome.get("trajectory_fingerprint", ""))[:16],
                }
            )
    return rows


def _render_arena(artifact: tuple[Path, dict[str, Any]] | None) -> None:
    st.header("Stateful agent behavioral arena")
    if artifact is None:
        st.info(
            "Run `python -m evals.arena evals/experiments/agent_arena.json "
            "--output data/evaluations/arena/latest.json --require-promotion`."
        )
        return
    path, report = artifact
    variants = report.get("variants") or []
    by_name = {(item.get("variant") or {}).get("name"): item for item in variants}
    promotion = report.get("promotion") or {}
    candidate = by_name.get(promotion.get("candidate")) or {}
    baseline = by_name.get(promotion.get("baseline")) or {}
    standings = report.get("standings") or []
    candidate_elo = next(
        (item.get("elo", 0) for item in standings if item.get("variant") == promotion.get("candidate")),
        0,
    )
    st.caption(
        f"Artifact `{path}` | source `{report.get('score_source', 'unknown')}` | dataset "
        f"`{str(report.get('dataset_fingerprint', ''))[:16]}` | report "
        f"`{str(report.get('report_fingerprint', ''))[:16]}`"
    )
    if report.get("score_source") == "deterministic_simulation":
        st.warning(
            "These are deterministic policy simulations over synthetic seed cases—not live-model or "
            "production-quality claims. Use the live adapter and human-calibrated judge before publishing."
        )
    columns = st.columns(6)
    _metric(
        columns[0],
        "Promotion gate",
        "APPROVED" if promotion.get("approved") else "BLOCKED",
    )
    _metric(columns[1], "Candidate pass", f"{float(candidate.get('pass_rate', 0)):.1%}")
    _metric(
        columns[2],
        "Quality delta",
        f"{float(candidate.get('quality_score', 0)) - float(baseline.get('quality_score', 0)):+.3f}",
    )
    _metric(columns[3], "Safety failures", f"{float(candidate.get('safety_failure_rate', 0)):.1%}")
    _metric(columns[4], "Tool F1", f"{float(candidate.get('tool_f1', 0)):.1%}")
    _metric(columns[5], "Candidate Elo", f"{float(candidate_elo):.1f}")

    st.subheader("Variant comparison and pairwise tournament")
    st.dataframe(
        [
            {
                "variant": (item.get("variant") or {}).get("name"),
                "adapter": (item.get("variant") or {}).get("adapter"),
                "quality": round(float(item.get("quality_score", 0)), 3),
                "pass rate": item.get("pass_rate"),
                "safety failure": item.get("safety_failure_rate"),
                "tool F1": item.get("tool_f1"),
                "judge disagreement": item.get("mean_judge_disagreement"),
                "P95 ms": item.get("p95_latency_ms"),
                "tokens": item.get("total_tokens"),
            }
            for item in variants
        ],
        width="stretch",
        hide_index=True,
    )
    left, right = st.columns(2)
    left.write("Elo standings")
    left.dataframe(standings, width="stretch", hide_index=True)
    right.write("Promotion checks")
    right.dataframe(promotion.get("checks") or [], width="stretch", hide_index=True)

    rows = arena_outcome_rows(report)
    st.subheader("Trajectory replay and counterfactual comparison")
    st.dataframe(rows, width="stretch", hide_index=True)
    scenario_ids = sorted({str(item.get("scenario")) for item in rows})
    if not scenario_ids:
        return
    selected_id = st.selectbox("Scenario", scenario_ids)
    comparison = {
        (item.get("variant") or {}).get("name"): next(
            (
                outcome
                for outcome in item.get("outcomes") or []
                if outcome.get("scenario_id") == selected_id
            ),
            None,
        )
        for item in variants
    }
    panels = st.columns(max(1, len(comparison)))
    for panel, (name, outcome) in zip(panels, comparison.items(), strict=False):
        panel.markdown(f"#### {name}")
        if not outcome:
            panel.info("No trajectory recorded.")
            continue
        for turn in outcome.get("transcript") or []:
            panel.markdown(f"**User {turn.get('turn')}**: {turn.get('prompt', '')}")
            panel.markdown(f"**Agent**: {turn.get('response', '')}")
            panel.caption(
                f"{turn.get('action')} · {turn.get('route')} · tools "
                f"{', '.join(turn.get('tool_calls') or []) or 'none'}"
            )
        panel.write("Trajectory graders")
        panel.dataframe(outcome.get("judges") or [], width="stretch", hide_index=True)
        panel.json(
            {
                "metrics": outcome.get("metrics"),
                "trajectory_fingerprint": outcome.get("trajectory_fingerprint"),
                "report_fingerprint": report.get("report_fingerprint"),
            }
        )


def _render_model_gateway(artifact: tuple[Path, dict[str, Any]] | None) -> None:
    st.header("Production LLM inference control plane")
    if artifact is None:
        st.info(
            "Run `python -m evals.model_gateway_evaluation --output "
            "data/evaluations/model-gateway/latest.json`."
        )
        return
    path, report = artifact
    st.caption(
        f"Artifact `{path}` | dataset `{str(report.get('dataset_fingerprint', ''))[:16]}` | "
        f"report `{str(report.get('report_fingerprint', ''))[:16]}`"
    )
    st.warning(
        "This artifact uses deterministic provider doubles to validate gateway control flow. "
        "Replace it with measured provider traffic before claiming production latency or savings."
    )
    columns = st.columns(7)
    _metric(columns[0], "Control pass", f"{float(report.get('pass_rate', 0)):.1%}")
    _metric(columns[1], "Receipt integrity", f"{float(report.get('receipt_integrity_rate', 0)):.1%}")
    _metric(columns[2], "Fallback recovery", f"{float(report.get('fallback_recovery_rate', 0)):.1%}")
    _metric(columns[3], "Budget controls", f"{float(report.get('budget_control_pass_rate', 0)):.1%}")
    _metric(columns[4], "High-risk bypass", f"{float(report.get('high_risk_bypass_rate', 0)):.1%}")
    _metric(columns[5], "Privacy violations", int(report.get("privacy_violations", 0)))
    _metric(
        columns[6],
        "Canary observed",
        f"{float(report.get('canary_observed_percentage', 0)):.1f}%",
        f"target {float(report.get('canary_target_percentage', 0)):.1f}%",
    )
    st.subheader("Control-plane scenarios")
    outcomes = report.get("outcomes") or []
    st.dataframe(
        [
            {
                "scenario": item.get("scenario_id"),
                "control": item.get("expected_control"),
                "observed": ", ".join(item.get("observed_controls") or []),
                "status": item.get("actual_status"),
                "provider": item.get("selected_provider") or "not invoked",
                "cache hit": item.get("cache_hit"),
                "receipt valid": item.get("receipt_verified"),
                "privacy violations": item.get("content_leakage_violations"),
                "passed": item.get("passed"),
                "evidence": str(item.get("evidence_fingerprint", ""))[:16],
            }
            for item in outcomes
        ],
        width="stretch",
        hide_index=True,
    )
    if outcomes:
        selected = st.selectbox(
            "Gateway evidence receipt",
            outcomes,
            format_func=lambda item: f"{item.get('scenario_id')} · {item.get('expected_control')}",
        )
        st.json(selected)


def _render_reliability(artifact: tuple[Path, dict[str, Any]] | None) -> None:
    st.header("Fault injection and recovery SLOs")
    if artifact is None:
        st.info(
            "Run `python -m evals.reliability --output "
            "data/evaluations/reliability/latest.json`."
        )
        return
    path, report = artifact
    baseline = report.get("baseline") or {}
    resilient = report.get("resilient") or {}
    st.caption(
        f"Artifact `{path}` | dataset `{str(report.get('dataset_fingerprint', ''))[:16]}` | "
        f"policy `{str(report.get('policy_fingerprint', ''))[:16]}`"
    )
    columns = st.columns(6)
    _metric(
        columns[0],
        "Scenario pass",
        f"{float(resilient.get('scenario_pass_rate', 0)):.1%}",
        f"{float(report.get('improvement_percentage_points', 0)):+.1f} pp",
    )
    _metric(columns[1], "Recovery", f"{float(report.get('recovery_success_rate', 0)):.1%}")
    _metric(columns[2], "Availability", f"{float(resilient.get('availability_rate', 0)):.1%}")
    _metric(
        columns[3], "Safe containment", f"{float(resilient.get('safe_containment_rate', 0)):.1%}"
    )
    _metric(columns[4], "Recovery P95", f"{float(resilient.get('p95_duration_ms', 0)):.1f} ms")
    _metric(columns[5], "Recovery cost", f"${float(resilient.get('total_cost_usd', 0)):.4f}")
    st.subheader("Baseline versus resilient policy")
    st.bar_chart(
        {
            "baseline": {
                "scenario pass": baseline.get("scenario_pass_rate", 0),
                "availability": baseline.get("availability_rate", 0),
                "safe containment": baseline.get("safe_containment_rate", 0),
            },
            "resilient": {
                "scenario pass": resilient.get("scenario_pass_rate", 0),
                "availability": resilient.get("availability_rate", 0),
                "safe containment": resilient.get("safe_containment_rate", 0),
            },
        }
    )
    outcomes = report.get("outcomes") or []
    components = sorted({str(item.get("component")) for item in outcomes})
    selected_components = st.multiselect("Components", components, default=components)
    filtered = [item for item in outcomes if item.get("component") in selected_components]
    st.dataframe(
        [
            {
                "scenario": item.get("scenario_id"),
                "component": item.get("component"),
                "fault": item.get("fault"),
                "strategy": item.get("strategy"),
                "baseline": (item.get("baseline") or {}).get("status"),
                "resilient": (item.get("resilient") or {}).get("status"),
                "SLO": item.get("slo_compliant"),
                "latency ms": round(float((item.get("resilient") or {}).get("duration_ms", 0)), 2),
                "extra tokens": item.get("additional_tokens"),
                "extra cost": round(float(item.get("additional_cost_usd", 0)), 6),
            }
            for item in filtered
        ],
        width="stretch",
        hide_index=True,
    )
    if filtered:
        selected = st.selectbox(
            "Replay scenario",
            filtered,
            format_func=lambda item: f"{item.get('scenario_id')} · {item.get('fault')}",
        )
        st.info(incident_summary(selected))
        left, right = st.columns(2)
        left.write("Baseline trace")
        left.dataframe((selected.get("baseline") or {}).get("events") or [], hide_index=True)
        right.write("Resilient trace")
        right.dataframe((selected.get("resilient") or {}).get("events") or [], hide_index=True)
    st.json(
        {
                "scenario_id": selected.get("scenario_id"),
                "expected_status": selected.get("expected_status"),
                "baseline_trace": (selected.get("baseline") or {}).get("trace_fingerprint"),
                "resilient_trace": (selected.get("resilient") or {}).get("trace_fingerprint"),
                "report_fingerprint": report.get("report_fingerprint"),
        }
    )


def _render_online_monitor(artifact: tuple[Path, dict[str, Any]] | None) -> None:
    st.header("Online AI quality and canary controller")
    if artifact is None:
        st.info(
            "Run `python -m evals.online_monitor_evaluation --output "
            "data/evaluations/online-monitor/latest.json`."
        )
        return
    path, report = artifact
    decision = report.get("canary_decision") or {}
    st.caption(
        f"Artifact `{path}` | dataset `{str(report.get('dataset_fingerprint', ''))[:16]}` | "
        f"report `{str(report.get('report_fingerprint', ''))[:16]}`"
    )
    st.warning(
        "The checked-in stream is a deterministic incident drill. Connect the content-free runtime "
        "event feed and delayed production labels before treating this as an online SLO."
    )
    columns = st.columns(7)
    _metric(columns[0], "Release action", str(decision.get("action", "unknown")).upper())
    _metric(columns[1], "Accepted events", int(report.get("accepted_events", 0)))
    _metric(columns[2], "Exactly-once duplicates", int(report.get("duplicate_events", 0)))
    _metric(columns[3], "Feedback joined", int(report.get("joined_feedback", 0)))
    _metric(columns[4], "Event integrity", f"{float(report.get('event_integrity_rate', 0)):.1%}")
    _metric(columns[5], "SLO alerts", len(report.get("alerts") or []))
    _metric(columns[6], "Privacy violations", int(report.get("privacy_violations", 0)))

    st.subheader("Rolling quality, safety, latency, cost, and reliability windows")
    st.dataframe(report.get("windows") or [], width="stretch", hide_index=True)
    left, right = st.columns(2)
    left.write("Multi-window error-budget alerts")
    left.dataframe(report.get("alerts") or [], width="stretch", hide_index=True)
    right.write("Statistical canary decision")
    right.json(decision)
    st.json(
        {
            "provider_mix_js_divergence": report.get("provider_mix_js_divergence"),
            "drift_detected": report.get("drift_detected"),
            "policy_version": report.get("policy_version"),
            "report_fingerprint": report.get("report_fingerprint"),
        }
    )


def _render_memory(artifact: tuple[Path, dict[str, Any]] | None) -> None:
    st.header("Trustworthy long-term agent memory")
    if artifact is None:
        st.info(
            "Run `python -m evals.memory_evaluation --output "
            "data/evaluations/memory/latest.json`."
        )
        return
    path, report = artifact
    st.caption(
        f"Artifact `{path}` | dataset `{str(report.get('dataset_fingerprint', ''))[:16]}` | "
        f"report `{str(report.get('report_fingerprint', ''))[:16]}`"
    )
    st.warning(
        "These are deterministic control tests over a synthetic memory corpus. Calibrate retrieval "
        "and extraction on consented production-like conversations before making quality claims."
    )
    columns = st.columns(7)
    _metric(columns[0], "Control pass", f"{float(report.get('pass_rate', 0)):.1%}")
    _metric(columns[1], "Relevant recall", f"{float(report.get('relevant_recall', 0)):.1%}")
    _metric(
        columns[2],
        "Tenant leakage",
        f"{float(report.get('cross_tenant_leakage_rate', 0)):.1%}",
    )
    _metric(
        columns[3],
        "Poisoning ASR",
        f"{float(report.get('poisoning_attack_success_rate', 0)):.1%}",
    )
    _metric(columns[4], "Stale recall", f"{float(report.get('stale_memory_rate', 0)):.1%}")
    _metric(
        columns[5],
        "Deletion violations",
        f"{float(report.get('deletion_violation_rate', 0)):.1%}",
    )
    _metric(columns[6], "Artifact integrity", f"{float(report.get('integrity_rate', 0)):.1%}")
    st.subheader("Memory governance scenarios")
    st.dataframe(
        [
            {
                "scenario": item.get("scenario_id"),
                "control": item.get("kind"),
                "passed": item.get("passed"),
                "observation": item.get("observed"),
                "selected": len(item.get("selected_memory_ids") or []),
                "receipt valid": item.get("receipt_verified"),
                "evidence": str(item.get("evidence_fingerprint", ""))[:16],
            }
            for item in report.get("outcomes") or []
        ],
        width="stretch",
        hide_index=True,
    )


def _render_grounding(artifact: tuple[Path, dict[str, Any]] | None) -> None:
    st.header("Claim-level grounding and hallucination gate")
    if artifact is None:
        st.info(
            "Run `python -m evals.grounding_evaluation --output "
            "data/evaluations/grounding/latest.json`."
        )
        return
    path, report = artifact
    st.caption(
        f"Artifact `{path}` | dataset `{str(report.get('dataset_fingerprint', ''))[:16]}` | "
        f"report `{str(report.get('report_fingerprint', ''))[:16]}`"
    )
    st.warning(
        "The checked-in gate uses deterministic lexical/hashed-semantic alignment over synthetic "
        "cases. Validate thresholds with human-labelled claims before publishing quality results."
    )
    columns = st.columns(7)
    _metric(columns[0], "Control pass", f"{float(report.get('pass_rate', 0)):.1%}")
    _metric(
        columns[1], "Unsafe release", f"{float(report.get('unsafe_release_rate', 0)):.1%}"
    )
    _metric(
        columns[2],
        "Fake citation escape",
        f"{float(report.get('fabricated_citation_escape_rate', 0)):.1%}",
    )
    _metric(
        columns[3],
        "High-risk escape",
        f"{float(report.get('high_risk_claim_escape_rate', 0)):.1%}",
    )
    _metric(columns[4], "Repair success", f"{float(report.get('repair_success_rate', 0)):.1%}")
    _metric(
        columns[5], "Mean coverage", f"{float(report.get('mean_claim_coverage', 0)):.1%}"
    )
    _metric(
        columns[6],
        "Receipt integrity",
        f"{float(report.get('receipt_integrity_rate', 0)):.1%}",
    )
    st.subheader("Claim-control scenarios")
    st.dataframe(
        [
            {
                "scenario": item.get("scenario_id"),
                "kind": item.get("kind"),
                "expected": item.get("expected_action"),
                "actual": item.get("actual_action"),
                "coverage": item.get("claim_coverage"),
                "citation precision": item.get("citation_precision"),
                "unsafe release": item.get("unsafe_answer_released"),
                "passed": item.get("passed"),
                "evidence": str(item.get("evidence_fingerprint", ""))[:16],
            }
            for item in report.get("outcomes") or []
        ],
        width="stretch",
        hide_index=True,
    )


def _render_uncertainty(artifact: tuple[Path, dict[str, Any]] | None) -> None:
    st.header("Conformal uncertainty and selective generation")
    if artifact is None:
        st.info(
            "Run `python -m evals.uncertainty_evaluation --output "
            "data/evaluations/uncertainty/latest.json --artifact "
            "data/evaluations/uncertainty/calibrator.json`."
        )
        return
    path, report = artifact
    st.caption(
        f"Artifact `{path}` | dataset `{str(report.get('dataset_fingerprint', ''))[:16]}` | "
        f"calibrator `{str(report.get('calibrator_fingerprint', ''))[:16]}`"
    )
    st.warning(
        "Coverage and risk figures come from a synthetic calibration/test split. Conformal validity "
        "depends on exchangeability and must be re-evaluated on representative labelled traffic."
    )
    columns = st.columns(7)
    _metric(columns[0], "Answer coverage", f"{float(report.get('coverage_rate', 0)):.1%}")
    _metric(
        columns[1],
        "Selective accuracy",
        f"{float(report.get('selective_accuracy', 0)):.1%}",
    )
    _metric(
        columns[2], "Released error", f"{float(report.get('empirical_error_rate', 0)):.1%}"
    )
    _metric(
        columns[3],
        "Incorrect abstention",
        f"{float(report.get('incorrect_abstention_rate', 0)):.1%}",
    )
    _metric(
        columns[4],
        "Receipt integrity",
        f"{float(report.get('receipt_integrity_rate', 0)):.1%}",
    )
    _metric(columns[5], "Target error", f"{float(report.get('target_error_rate', 0)):.1%}")
    _metric(
        columns[6],
        "Shift drill",
        "DETECTED" if report.get("shift_detected") else "MISSED",
        f"JS {float(report.get('shifted_js_divergence', 0)):.3f}",
    )
    st.subheader("Route-aware conformal thresholds")
    st.json(report.get("route_quantiles") or {})
    st.subheader("Held-out selective decisions")
    st.dataframe(report.get("outcomes") or [], width="stretch", hide_index=True)


def _render_adaptive_compute(artifact: tuple[Path, dict[str, Any]] | None) -> None:
    st.header("Adaptive test-time compute")
    if artifact is None:
        st.info(
            "Run `python -m evals.adaptive_compute_evaluation --output "
            "data/evaluations/adaptive-compute/latest.json`."
        )
        return
    path, report = artifact
    st.caption(
        f"Artifact `{path}` | dataset `{str(report.get('dataset_fingerprint', ''))[:16]}` | "
        f"scenarios {int(report.get('scenario_count', 0))}"
    )
    st.warning(
        "This is a deterministic control-plane ablation. Replace simulated candidate outcomes "
        "with repeated, human-labelled live-model runs before making quality or savings claims."
    )
    columns = st.columns(8)
    _metric(columns[0], "Gate pass", f"{float(report.get('pass_rate', 0)):.1%}")
    _metric(
        columns[1],
        "Baseline accuracy",
        f"{float(report.get('baseline_selective_accuracy', 0)):.1%}",
    )
    _metric(
        columns[2],
        "Adaptive accuracy",
        f"{float(report.get('adaptive_selective_accuracy', 0)):.1%}",
    )
    _metric(columns[3], "Recovery", f"{float(report.get('recovery_rate', 0)):.1%}")
    _metric(
        columns[4],
        "Unsafe release",
        f"{float(report.get('unsafe_release_rate', 0)):.1%}",
    )
    _metric(
        columns[5],
        "Call reduction",
        f"{float(report.get('candidate_call_reduction', 0)):.1%}",
    )
    _metric(columns[6], "Mean extra tokens", f"{float(report.get('mean_extra_tokens', 0)):.0f}")
    _metric(
        columns[7],
        "Receipt integrity",
        f"{float(report.get('receipt_integrity_rate', 0)):.1%}",
    )
    st.subheader("Per-scenario compute allocation")
    st.dataframe(report.get("outcomes") or [], width="stretch", hide_index=True)


def _render_evidence_quality(artifact: tuple[Path, dict[str, Any]] | None) -> None:
    st.header("Evidence intelligence")
    if artifact is None:
        st.info(
            "Run `python -m evals.evidence_quality_evaluation --output "
            "data/evaluations/evidence-quality/latest.json`."
        )
        return
    path, report = artifact
    st.caption(
        f"Artifact `{path}` | dataset `{str(report.get('dataset_fingerprint', ''))[:16]}` | "
        f"scenarios {int(report.get('scenario_count', 0))}"
    )
    st.warning(
        "This corpus validates deterministic evidence controls. Authority signals and heuristic "
        "contradiction detection are not substitutes for domain review or a calibrated NLI model."
    )
    columns = st.columns(8)
    _metric(columns[0], "Gate pass", f"{float(report.get('pass_rate', 0)):.1%}")
    _metric(
        columns[1], "Benign utility", f"{float(report.get('benign_utility_rate', 0)):.1%}"
    )
    _metric(
        columns[2],
        "Attack containment",
        f"{float(report.get('attack_containment_rate', 0)):.1%}",
    )
    _metric(
        columns[3],
        "Injection quarantine",
        f"{float(report.get('prompt_injection_quarantine_rate', 0)):.1%}",
    )
    _metric(
        columns[4],
        "Duplicate blocking",
        f"{float(report.get('duplicate_laundering_block_rate', 0)):.1%}",
    )
    _metric(
        columns[5],
        "Conflict detection",
        f"{float(report.get('contradiction_detection_rate', 0)):.1%}",
    )
    _metric(
        columns[6],
        "Stale blocking",
        f"{float(report.get('stale_evidence_block_rate', 0)):.1%}",
    )
    _metric(
        columns[7],
        "Unsafe release",
        f"{float(report.get('unsafe_evidence_release_rate', 0)):.1%}",
    )
    st.subheader("Evidence adjudication outcomes")
    st.dataframe(report.get("outcomes") or [], width="stretch", hide_index=True)


def _render_traces(records: list[dict[str, Any]], telemetry_artifact) -> None:
    st.header("GenAI trace debugger")
    if telemetry_artifact:
        _, report = telemetry_artifact
        columns = st.columns(5)
        _metric(columns[0], "Valid", f"{float(report.get('valid_trace_rate', 0)):.1%}")
        _metric(columns[1], "Semantic coverage", f"{float(report.get('semantic_attribute_coverage', 0)):.1%}")
        _metric(columns[2], "Fingerprint errors", report.get("fingerprint_violations", 0))
        _metric(columns[3], "Hierarchy errors", report.get("hierarchy_violations", 0))
        _metric(
            columns[4],
            "Leakage violations",
            int(report.get("content_attribute_violations", 0))
            + int(report.get("secret_value_violations", 0)),
        )
    if not records:
        st.info("Enable `GENAI_TRACE_JSONL_PATH` and run an agent request to inspect a trace.")
        return
    selected = st.selectbox(
        "Trace",
        records,
        format_func=lambda item: (
            f"{str(item.get('trace_id', 'unknown'))[:16]} · {item.get('service_name', 'unknown')}"
        ),
    )
    st.caption(
        f"Source `{selected.get('_source')}` | fingerprint "
        f"`{str(selected.get('fingerprint', ''))[:20]}` | content capture "
        f"`{selected.get('content_capture_enabled', False)}`"
    )
    st.graphviz_chart(trace_dot(selected), width="stretch")
    rows = trace_rows(selected)
    st.write("Span waterfall")
    st.bar_chart(rows, x="span", y="duration_ms", color="kind")
    st.dataframe(rows, width="stretch", hide_index=True)
    span_names = list(range(len(selected.get("spans") or [])))
    if span_names:
        span_index = st.selectbox(
            "Inspect safe span attributes",
            span_names,
            format_func=lambda index: (selected.get("spans") or [])[index].get("name", "unknown"),
        )
        st.json((selected.get("spans") or [])[span_index].get("attributes") or {})


def _render_specialized(artifacts: dict[str, tuple[Path, dict[str, Any]] | None]) -> None:
    st.header("Security, retrieval, and benchmark evidence")
    security = artifacts["security"]
    if security:
        _, report = security
        defended = report.get("defended") or {}
        st.subheader("Agent security")
        st.bar_chart(
            {
                "baseline": {
                    "attack success": (report.get("baseline") or {}).get("attack_success_rate", 0),
                    "secret leakage": (report.get("baseline") or {}).get("secret_leakage_rate", 0),
                },
                "defended": {
                    "attack success": defended.get("attack_success_rate", 0),
                    "secret leakage": defended.get("secret_leakage_rate", 0),
                },
            }
        )
    retrieval = artifacts["retrieval"]
    if retrieval:
        _, report = retrieval
        st.subheader("Code retrieval")
        st.json(
            {
                "strategy": report.get("strategy"),
                "recall_at_k": report.get("recall_at_k"),
                "mrr": report.get("mrr"),
                "ndcg_at_k": report.get("ndcg_at_k"),
                "mean_context_tokens": report.get("mean_context_tokens"),
                "dataset_fingerprint": report.get("dataset_fingerprint"),
            }
        )
    benchmark = artifacts["benchmark"]
    if benchmark:
        _, report = benchmark
        st.subheader("Benchmark release")
        st.write(
            {
                "winner": report.get("winner"),
                "score_source": report.get("score_source"),
                "externally_verified": report.get("externally_verified"),
                "cases": report.get("total_cases"),
            }
        )
        st.dataframe(report.get("variants") or [], width="stretch", hide_index=True)


def main() -> None:
    st.set_page_config(page_title="AgentForge Reliability Command Center", page_icon="⚡", layout="wide")
    st.title("AgentForge Reliability Command Center")
    st.caption(
        "Unified behavioral evaluation, inference controls, online canary SLOs, fault injection, GenAI traces, "
        "security, retrieval, and benchmark evidence."
    )
    artifacts = {
        "reliability": latest_report(
            RESULTS_ROOT / "reliability",
            lambda item: item.get("generated_by") == "agentforge-reliability-lab",
        ),
        "telemetry": latest_report(
            RESULTS_ROOT / "telemetry", lambda item: "valid_trace_rate" in item
        ),
        "security": latest_report(
            RESULTS_ROOT / "security",
            lambda item: item.get("generated_by") == "agentforge-security-eval",
        ),
        "retrieval": latest_report(
            RESULTS_ROOT / "code-context",
            lambda item: "recall_at_k" in item and "index_fingerprint" in item,
        ),
        "benchmark": latest_report(
            RESULTS_ROOT / "benchmark-release",
            lambda item: "matrix_fingerprint" in item and "variants" in item,
        ),
        "arena": latest_report(
            RESULTS_ROOT / "arena",
            lambda item: "arena_id" in item and "standings" in item and "promotion" in item,
        ),
        "model_gateway": latest_report(
            RESULTS_ROOT / "model-gateway",
            lambda item: item.get("generated_by") == "agentforge-inference-gateway-eval",
        ),
        "online_monitor": latest_report(
            RESULTS_ROOT / "online-monitor",
            lambda item: item.get("generated_by") == "agentforge-online-eval-controller",
        ),
        "memory": latest_report(
            RESULTS_ROOT / "memory",
            lambda item: item.get("generated_by") == "agentforge-memory-eval",
        ),
        "grounding": latest_report(
            RESULTS_ROOT / "grounding",
            lambda item: item.get("generated_by") == "agentforge-grounding-eval",
        ),
        "uncertainty": latest_report(
            RESULTS_ROOT / "uncertainty",
            lambda item: item.get("generated_by") == "agentforge-uncertainty-eval",
        ),
        "adaptive_compute": latest_report(
            RESULTS_ROOT / "adaptive-compute",
            lambda item: item.get("generated_by") == "agentforge-adaptive-compute-eval",
        ),
        "evidence_quality": latest_report(
            RESULTS_ROOT / "evidence-quality",
            lambda item: item.get("generated_by") == "agentforge-evidence-quality-eval",
        ),
    }
    traces = load_trace_records(
        [RESULTS_ROOT / "telemetry", DATASETS_ROOT / "genai_telemetry_smoke.jsonl"]
    )
    overview, arena, gateway, online, memory, grounding, uncertainty, evidence, adaptive, reliability, traces_tab, specialized = st.tabs(
        [
            "Overview",
            "Behavior arena",
            "Inference gateway",
            "Online AI monitor",
            "Memory governance",
            "Grounding gate",
            "Uncertainty control",
            "Evidence intelligence",
            "Adaptive compute",
            "Reliability lab",
            "Trace debugger",
            "Evaluation evidence",
        ]
    )
    with overview:
        _render_overview(artifacts)
    with arena:
        _render_arena(artifacts["arena"])
    with gateway:
        _render_model_gateway(artifacts["model_gateway"])
    with online:
        _render_online_monitor(artifacts["online_monitor"])
    with memory:
        _render_memory(artifacts["memory"])
    with grounding:
        _render_grounding(artifacts["grounding"])
    with uncertainty:
        _render_uncertainty(artifacts["uncertainty"])
    with evidence:
        _render_evidence_quality(artifacts["evidence_quality"])
    with adaptive:
        _render_adaptive_compute(artifacts["adaptive_compute"])
    with reliability:
        _render_reliability(artifacts["reliability"])
    with traces_tab:
        _render_traces(traces, artifacts["telemetry"])
    with specialized:
        _render_specialized(artifacts)


if __name__ == "__main__":
    main()
