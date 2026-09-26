from __future__ import annotations

from code_agent.context_dashboard import curve_rows, load_ablation_reports
from code_agent.context_evaluation import AblationPoint, ContextAblationReport


def test_context_dashboard_loads_valid_reports_and_builds_curve_rows(tmp_path):
    report = ContextAblationReport(
        dataset_fingerprint="dataset",
        index_fingerprint="index",
        top_k=8,
        strategies=["hybrid_rerank"],
        token_budgets=[512],
        embedding_backend="hashing-semantic-384",
        reranker_backend="query-facet-reranker-v1",
        parser_backends={"tree-sitter": 6},
        points=[
            AblationPoint(
                strategy="hybrid_rerank",
                max_tokens=512,
                recall_at_k=1.0,
                mrr=0.9,
                ndcg_at_k=0.87,
                mean_context_tokens=480,
                mean_selected_files=4,
                mean_compression_ratio=0.12,
                mean_duplicate_lines_removed=3,
            )
        ],
    )
    nested = tmp_path / "run"
    nested.mkdir()
    (nested / "report.json").write_text(report.model_dump_json(), encoding="utf-8")
    (nested / "invalid.json").write_text("{}", encoding="utf-8")

    loaded = load_ablation_reports(tmp_path)
    assert len(loaded) == 1
    rows = curve_rows(loaded[0][1])
    assert rows == [
        {
            "strategy": "hybrid_rerank",
            "token budget": 512,
            "mean tokens used": 480.0,
            "Recall@K": 1.0,
            "MRR": 0.9,
            "NDCG@K": 0.87,
            "selected files": 4.0,
            "compression": 0.12,
            "deduplicated lines": 3.0,
        }
    ]
