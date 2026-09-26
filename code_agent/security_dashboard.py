"""Streamlit dashboard for baseline-versus-defended agent security evidence."""

from __future__ import annotations

import os
from pathlib import Path

import streamlit as st

from code_agent.security_evaluation import SecurityEvalReport

RESULTS_DIR = Path(os.getenv("CODE_SECURITY_EVAL_DIR", "data/evaluations/security"))


def load_security_reports(root: Path) -> list[tuple[Path, SecurityEvalReport]]:
    reports: list[tuple[Path, SecurityEvalReport]] = []
    if not root.exists():
        return reports
    for path in root.rglob("*.json"):
        try:
            reports.append(
                (path, SecurityEvalReport.model_validate_json(path.read_text(encoding="utf-8")))
            )
        except (OSError, ValueError):
            continue
    return sorted(reports, key=lambda item: item[0].stat().st_mtime, reverse=True)


def main() -> None:
    st.set_page_config(page_title="AgentForge Security Lab", page_icon="🛡️", layout="wide")
    st.title("AgentForge Agent Security Red-Team Lab")
    st.caption("Reproducible policy attacks, benign controls, provenance, and CI regression evidence.")
    reports = load_security_reports(RESULTS_DIR)
    if not reports:
        st.info(
            "No reports found. Run `python -m code_agent.security_evaluation --output "
            "data/evaluations/security/latest.json`."
        )
        return
    selected_path, report = st.selectbox(
        "Security report",
        reports,
        format_func=lambda item: f"{item[0].name} | {item[1].dataset_fingerprint[:12]}",
    )
    st.caption(
        f"Report `{selected_path}` | policy `{report.policy_version}` | "
        f"dataset `{report.dataset_fingerprint}`"
    )
    columns = st.columns(6)
    columns[0].metric(
        "Attack success",
        f"{report.defended.attack_success_rate:.1%}",
        delta=f"{report.defended.attack_success_rate - report.baseline.attack_success_rate:.1%}",
        delta_color="inverse",
    )
    columns[1].metric("Containment", f"{report.defended.containment_rate:.1%}")
    columns[2].metric("Benign pass", f"{report.defended.benign_pass_rate:.1%}")
    columns[3].metric("False positives", f"{report.defended.false_positive_rate:.1%}")
    columns[4].metric("Secret leakage", f"{report.defended.secret_leakage_rate:.1%}")
    columns[5].metric("P95 policy latency", f"{report.defended.p95_policy_latency_ms:.3f} ms")

    st.subheader("Baseline versus defended")
    st.bar_chart(
        {
            "baseline": {
                "attack success": report.baseline.attack_success_rate,
                "secret leakage": report.baseline.secret_leakage_rate,
                "unsafe tools": report.baseline.unsafe_tool_call_rate,
            },
            "defended": {
                "attack success": report.defended.attack_success_rate,
                "secret leakage": report.defended.secret_leakage_rate,
                "unsafe tools": report.defended.unsafe_tool_call_rate,
            },
        }
    )
    left, right = st.columns(2)
    left.write("Attack success by category")
    left.bar_chart(report.category_attack_success_rate)
    right.write("Attack success by surface")
    right.bar_chart(report.surface_attack_success_rate)

    st.subheader("Replayable case outcomes")
    rows = [
        {
            "case": item.case_id,
            "kind": item.kind,
            "surface": item.attack_surface,
            "category": item.category,
            "passed": item.passed,
            "contained": item.defended_contained,
            "rules": ", ".join(item.decision.rule_ids),
            "approval": item.decision.requires_human_approval,
            "latency ms": round(item.decision.latency_ms, 4),
        }
        for item in report.outcomes
    ]
    st.dataframe(rows, use_container_width=True)
    st.subheader("Evidence fingerprints")
    st.json(
        {
            "dataset": report.dataset_fingerprint,
            "policy": report.policy_fingerprint,
            "policy_version": report.policy_version,
            "raw_attack_payloads_persisted": False,
        }
    )


if __name__ == "__main__":
    main()
