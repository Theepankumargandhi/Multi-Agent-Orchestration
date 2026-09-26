"""Streamlit comparison and trajectory-replay UI for coding-agent evaluations."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import streamlit as st

from code_agent.evaluation_models import CodeBenchmarkReport

RESULTS_DIR = Path(os.getenv("CODE_AGENT_EVAL_DIR", "data/evaluations/code-agent"))


def load_reports(root: Path) -> list[tuple[Path, CodeBenchmarkReport]]:
    reports = []
    if not root.exists():
        return reports
    for path in root.glob("*/report.json"):
        try:
            reports.append((path, CodeBenchmarkReport.model_validate_json(path.read_text(encoding="utf-8"))))
        except (OSError, ValueError):
            continue
    return sorted(reports, key=lambda item: item[1].generated_at, reverse=True)


def _safe_artifact(report_path: Path, relative: str) -> Path | None:
    if not relative:
        return None
    root = report_path.parent.resolve()
    candidate = (root / relative).resolve()
    try:
        candidate.relative_to(root)
    except ValueError:
        return None
    return candidate if candidate.is_file() else None


def _comparison_row(report: CodeBenchmarkReport) -> dict:
    low, high = report.pass_at_1_confidence_interval
    return {
        "run": report.run_id,
        "model": report.model,
        "workflow": report.workflow,
        "context": report.context_strategy,
        "pass@1": round(report.pass_at_1, 4),
        "95% CI": f"{low:.3f}-{high:.3f}",
        "tests passed": round(report.test_pass_rate, 4),
        "p95 ms": round(report.p95_duration_ms, 1),
        "tokens": report.total_prompt_tokens + report.total_completion_tokens,
        "cost USD": round(report.total_estimated_cost_usd, 6),
        "cost/resolved": round(report.cost_per_resolved_usd or 0.0, 6),
        "telemetry": round(report.telemetry_coverage_rate, 4),
    }


def main() -> None:
    st.set_page_config(page_title="AgentForge Coding Lab", page_icon="🧪", layout="wide")
    st.title("AgentForge Coding-Agent Evaluation Lab")
    st.caption("Reproducible pass@1, cost, safety, and step-by-step trajectory evidence.")
    reports = load_reports(RESULTS_DIR)
    if not reports:
        st.info(
            "No reports found. Run `python -m code_agent.evaluation <dataset.jsonl>` first."
        )
        return

    selected_path, report = st.selectbox(
        "Evaluation run",
        reports,
        format_func=lambda item: (
            f"{item[1].generated_at[:19]} | {item[1].model} | {item[1].workflow} | "
            f"pass@1 {item[1].pass_at_1:.1%} | {item[1].run_id}"
        ),
    )
    st.caption(
        f"Dataset `{report.dataset_fingerprint}` | Config `{report.config_fingerprint}`"
    )
    columns = st.columns(8)
    columns[0].metric("Pass@1", f"{report.pass_at_1:.1%}")
    columns[1].metric("Resolved", f"{report.resolved}/{report.total}")
    columns[2].metric("Tests passed", f"{report.test_pass_rate:.1%}")
    columns[3].metric("p95 latency", f"{report.p95_duration_ms / 1000:.1f}s")
    columns[4].metric("Tokens", f"{report.total_prompt_tokens + report.total_completion_tokens:,}")
    columns[5].metric("Estimated cost", f"${report.total_estimated_cost_usd:.4f}")
    columns[6].metric("Verified", f"{report.verification_pass_rate:.1%}")
    columns[7].metric("Telemetry", f"{report.telemetry_coverage_rate:.1%}")
    st.caption(f"Workflow: {report.workflow} | Context strategy: {report.context_strategy}")

    comparable = [item[1] for item in reports if item[1].dataset_fingerprint == report.dataset_fingerprint]
    st.subheader("Model and configuration comparison")
    st.dataframe([_comparison_row(item) for item in comparable], use_container_width=True)
    if len(comparable) > 1:
        st.bar_chart({item.model + ":" + item.run_id[-8:]: item.pass_at_1 for item in comparable})

    left, right = st.columns(2)
    left.write("Failure taxonomy")
    left.bar_chart(report.failure_categories)
    right.write("Sandbox policy")
    right.json(report.sandbox_policy)

    st.subheader("Repository and tag slices")
    st.dataframe(
        [{"slice": name, **metrics} for name, metrics in report.slice_metrics.items()],
        use_container_width=True,
    )

    st.subheader("Case explorer and trajectory replay")
    failure_only = st.toggle("Failures only", value=False)
    outcomes = [
        outcome
        for outcome in report.outcomes
        if not failure_only or not (outcome.score and outcome.score.resolved)
    ]
    if not outcomes:
        st.success("No matching cases.")
        return
    outcome = st.selectbox(
        "Case",
        outcomes,
        format_func=lambda item: (
            f"{'PASS' if item.score and item.score.resolved else 'FAIL'} | "
            f"{item.case_id} | {item.failure_category}"
        ),
    )
    if outcome.error:
        st.error(outcome.error)
        return
    st.json(outcome.score.model_dump() if outcome.score else {})
    trajectory_path = _safe_artifact(selected_path, outcome.trajectory_path)
    if trajectory_path is None:
        st.warning("Trajectory artifact is missing or outside the run directory.")
        return
    trajectory = json.loads(trajectory_path.read_text(encoding="utf-8"))
    case = trajectory.get("case", {})
    st.write("Issue")
    st.code(case.get("issue", ""), language="text")
    observations = trajectory.get("result", {}).get("observations", [])
    st.dataframe(
        [
            {
                "step": item.get("iteration"),
                "action": item.get("action"),
                "ok": item.get("ok"),
                "path": item.get("path"),
                "duration_ms": round(float(item.get("duration_ms") or 0), 2),
                "summary": item.get("summary"),
            }
            for item in observations
        ],
        use_container_width=True,
    )
    for item in observations:
        label = f"Step {item.get('iteration')} | {item.get('action')} | {'ok' if item.get('ok') else 'failed'}"
        with st.expander(label):
            st.write(item.get("summary", ""))
            if item.get("path"):
                st.code(item["path"], language="text")
            if item.get("output"):
                st.code(item["output"], language="text")

    patch_path = _safe_artifact(selected_path, outcome.patch_path)
    if patch_path and st.toggle("Show generated patch", value=False):
        patch_bytes = patch_path.read_bytes()
        patch = patch_bytes.decode("utf-8")
        if outcome.patch_sha256 and outcome.patch_sha256 != hashlib.sha256(patch_bytes).hexdigest():
            st.error("Patch integrity check failed.")
        else:
            st.code(patch, language="diff")


if __name__ == "__main__":
    main()
