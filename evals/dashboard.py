"""Read-only Streamlit dashboard for experiment reports and trace inspection."""

from __future__ import annotations

import json
import os
from pathlib import Path

import streamlit as st

from evals.platform import ExperimentReport, ExperimentStore

RESULTS_DIR = Path(os.getenv("EVAL_RESULTS_DIR", "data/evaluations"))
FLYWHEEL_DIR = RESULTS_DIR / "flywheel"


def _format_variant(report) -> dict:
    return {
        "variant": report.variant.name,
        "quality": round(report.quality_score, 4),
        "quality_ci": "–".join(f"{value:.3f}" for value in report.quality_confidence_interval),
        "quality_delta": round(report.quality_delta_vs_baseline, 4),
        "pass_rate": round(report.pass_rate, 4),
        "p50_ms": round(report.p50_latency_ms, 2),
        "p95_ms": round(report.p95_latency_ms, 2),
        "cost_usd": round(report.total_cost_usd, 6),
        "tokens": report.total_tokens,
        "pareto": report.pareto_optimal,
    }


def _load_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _render_flywheel() -> None:
    st.divider()
    st.header("Failure-to-improvement flywheel")
    st.caption(
        "Redacted failure clusters, reviewed regression candidates, offline gates, and canary state."
    )
    traces_path = FLYWHEEL_DIR / "traces.jsonl"
    clusters = _load_json(FLYWHEEL_DIR / "clusters.json") or []
    decision = _load_json(FLYWHEEL_DIR / "promotion.json")
    deployment = _load_json(FLYWHEEL_DIR / "deployment.json")
    trace_rows = []
    if traces_path.exists():
        for line in traces_path.read_text(encoding="utf-8").splitlines():
            try:
                trace_rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    if not any((trace_rows, clusters, decision, deployment)):
        st.info("Run `python -m evals.flywheel ingest ...` to create the first redacted trace set.")
        return

    left, middle, right = st.columns(3)
    left.metric("Redacted traces", len(trace_rows))
    middle.metric("Failure clusters", len(clusters))
    right.metric(
        "Canary gate",
        "approved" if decision and decision.get("approved_for_canary") else "not approved",
    )
    if trace_rows:
        category_counts = {}
        for row in trace_rows:
            for category in row.get("failure_categories", []):
                category_counts[category] = category_counts.get(category, 0) + 1
        st.subheader("Observed failure taxonomy")
        st.bar_chart(category_counts)
    if clusters:
        st.subheader("Recurring failure clusters")
        st.dataframe(
            [
                {
                    "cluster": item.get("cluster_id"),
                    "category": item.get("primary_category"),
                    "label": item.get("label"),
                    "size": item.get("size"),
                    "average_quality": item.get("average_quality"),
                    "intervention": item.get("recommended_intervention"),
                }
                for item in clusters
            ],
            use_container_width=True,
        )
    if decision:
        st.subheader("Offline promotion evidence")
        st.write(
            {
                "baseline": decision.get("baseline"),
                "candidate": decision.get("candidate"),
                "approved_for_canary": decision.get("approved_for_canary"),
                "decision_fingerprint": decision.get("decision_fingerprint"),
                "regressions": decision.get("regressions", []),
            }
        )
        st.dataframe(
            [
                {
                    **check,
                    "actual": str(check.get("actual", "")),
                    "threshold": str(check.get("threshold", "")),
                }
                for check in decision.get("checks", [])
            ],
            use_container_width=True,
        )
    if deployment:
        st.subheader("Canary deployment state")
        st.write(
            {
                "status": deployment.get("status"),
                "active_version": deployment.get("active_version"),
                "canary_version": deployment.get("canary_version"),
                "previous_version": deployment.get("previous_version"),
            }
        )
        st.dataframe(deployment.get("audit_log", []), use_container_width=True)


def main() -> None:
    st.set_page_config(page_title="AgentForge EvalOps", page_icon="🧪", layout="wide")
    st.title("🧪 AgentForge Evaluation & Reliability Lab")
    st.caption("Reproducible model, prompt, retrieval and tool-policy experiments.")
    store = ExperimentStore(RESULTS_DIR)
    experiments = store.list()
    if not experiments:
        st.info("Run `python -m evals.run_experiments` to generate the first report.")
        _render_flywheel()
        return

    selected = st.selectbox(
        "Experiment",
        experiments,
        format_func=lambda row: f"{row['created_at'][:19]} · {row['experiment_name']} · winner: {row['winner']}",
    )
    report = ExperimentReport.model_validate_json(Path(selected["report_path"]).read_text(encoding="utf-8"))
    st.subheader(f"Winner: {report.winner}")
    st.caption(f"Dataset SHA-256: `{report.dataset_fingerprint}`")
    st.write(
        {
            "evaluated_splits": report.evaluated_splits,
            "review_status": report.review_status_counts,
            "threshold": report.pass_threshold,
        }
    )
    st.dataframe([_format_variant(item) for item in report.reports], use_container_width=True)

    names = [item.variant.name for item in report.reports]
    st.subheader("Quality comparison")
    st.bar_chart({item.variant.name: item.quality_score for item in report.reports})
    selected_variant = st.selectbox("Inspect variant", names)
    variant = next(item for item in report.reports if item.variant.name == selected_variant)
    left, right = st.columns(2)
    left.write("Metric scores")
    left.json(variant.metric_scores)
    right.write("Failure taxonomy")
    right.json(variant.failure_categories or {"none": 0})
    st.write("Slice metrics")
    st.dataframe(
        [{"slice": name, **values} for name, values in variant.slice_scores.items()],
        use_container_width=True,
    )

    failed_only = st.toggle("Failures only", value=True)
    cases = [case for case in variant.cases if not failed_only or not case.passed]
    for case in cases:
        with st.expander(f"{'✅' if case.passed else '❌'} {case.case_id}: {case.input[:90]}"):
            st.write("Expected", case.expected.model_dump(exclude_none=True))
            st.write("Metrics", {name: metric.model_dump() for name, metric in case.metrics.items()})
            st.write("Trace")
            st.code(json.dumps(case.actual.trace, indent=2), language="json")

    auxiliary = sorted(RESULTS_DIR.glob("*calibration*.json")) + sorted(
        RESULTS_DIR.glob("*router*.report.json")
    )
    if auxiliary:
        st.subheader("Calibration and adaptive-routing artifacts")
        artifact = st.selectbox("Artifact", auxiliary, format_func=lambda path: path.name)
        st.json(json.loads(artifact.read_text(encoding="utf-8")))
    _render_flywheel()


if __name__ == "__main__":
    main()
