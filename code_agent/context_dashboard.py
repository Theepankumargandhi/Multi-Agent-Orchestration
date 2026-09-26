"""Streamlit dashboard for retrieval ablations and recall/token curves."""

from __future__ import annotations

import os
from pathlib import Path

import streamlit as st

from code_agent.context_evaluation import ContextAblationReport

RESULTS_DIR = Path(os.getenv("CODE_CONTEXT_EVAL_DIR", "data/evaluations/code-context"))


def load_ablation_reports(root: Path) -> list[tuple[Path, ContextAblationReport]]:
    reports = []
    if not root.exists():
        return reports
    for path in root.rglob("*.json"):
        try:
            report = ContextAblationReport.model_validate_json(path.read_text(encoding="utf-8"))
            reports.append((path, report))
        except (OSError, ValueError):
            continue
    return sorted(reports, key=lambda item: item[0].stat().st_mtime, reverse=True)


def curve_rows(report: ContextAblationReport) -> list[dict[str, object]]:
    return [
        {
            "strategy": point.strategy,
            "token budget": point.max_tokens,
            "mean tokens used": round(point.mean_context_tokens, 1),
            "Recall@K": round(point.recall_at_k, 4),
            "MRR": round(point.mrr, 4),
            "NDCG@K": round(point.ndcg_at_k, 4),
            "selected files": round(point.mean_selected_files, 2),
            "compression": round(point.mean_compression_ratio, 4),
            "deduplicated lines": round(point.mean_duplicate_lines_removed, 2),
        }
        for point in report.points
    ]


def main() -> None:
    st.set_page_config(page_title="AgentForge Retrieval Lab", page_icon="🔬", layout="wide")
    st.title("AgentForge Code Retrieval Ablation Lab")
    st.caption(
        "Compare lexical, graph, semantic, and reranked retrieval under fixed context budgets."
    )
    reports = load_ablation_reports(RESULTS_DIR)
    if not reports:
        st.info(
            "No ablation reports found. Run `python -m code_agent.context_evaluation "
            "--repository-root . --ablation --output "
            "data/evaluations/code-context/latest.json`."
        )
        return
    selected_path, report = st.selectbox(
        "Ablation report",
        reports,
        format_func=lambda item: f"{item[0].name} | {item[1].dataset_fingerprint[:12]}",
    )
    st.caption(
        f"Report `{selected_path}` | dataset `{report.dataset_fingerprint}` | "
        f"index `{report.index_fingerprint}`"
    )
    rows = curve_rows(report)
    best = max(report.points, key=lambda item: (item.recall_at_k, item.mrr, -item.mean_context_tokens))
    columns = st.columns(6)
    columns[0].metric("Best Recall@K", f"{best.recall_at_k:.1%}")
    columns[1].metric("Best MRR", f"{best.mrr:.3f}")
    columns[2].metric("Best NDCG", f"{best.ndcg_at_k:.3f}")
    columns[3].metric("Strategy", best.strategy)
    columns[4].metric("Budget", f"{best.max_tokens:,} tokens")
    columns[5].metric("Mean used", f"{best.mean_context_tokens:,.0f}")

    st.subheader("Recall versus token budget")
    st.line_chart(rows, x="token budget", y="Recall@K", color="strategy")
    left, right = st.columns(2)
    left.write("Mean reciprocal rank")
    left.line_chart(rows, x="token budget", y="MRR", color="strategy")
    right.write("Context efficiency")
    right.line_chart(rows, x="token budget", y="mean tokens used", color="strategy")
    st.write("Normalized discounted cumulative gain")
    st.line_chart(rows, x="token budget", y="NDCG@K", color="strategy")

    st.subheader("All ablation points")
    st.dataframe(rows, use_container_width=True)
    st.subheader("Runtime provenance")
    st.json(
        {
            "embedding_backend": report.embedding_backend,
            "reranker_backend": report.reranker_backend,
            "parser_backends": report.parser_backends,
            "strategies": report.strategies,
            "token_budgets": report.token_budgets,
        }
    )


if __name__ == "__main__":
    main()
