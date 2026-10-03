"""Credential-free evaluation and ablation for code-context retrieval."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
from pathlib import Path

from pydantic import BaseModel, Field

from code_agent.code_parsing import LANGUAGE_BY_SUFFIX
from code_agent.intelligence import STRATEGIES, CodeIntelligenceIndex
from code_agent.retrieval_backends import RetrievalConfig


class ContextEvalCase(BaseModel):
    id: str = Field(min_length=1, max_length=200)
    query: str = Field(min_length=5, max_length=4000)
    relevant_paths: list[str] = Field(min_length=1, max_length=30)
    tags: list[str] = Field(default_factory=list, max_length=20)


class ContextEvalOutcome(BaseModel):
    case_id: str
    selected_paths: list[str]
    relevant_paths: list[str]
    recall_at_k: float
    reciprocal_rank: float
    ndcg_at_k: float
    context_tokens: int = 0
    selected_files: int = 0
    compression_ratio: float = 0.0
    duplicate_lines_removed: int = 0


class ContextEvalReport(BaseModel):
    schema_version: str = "2.0"
    dataset_fingerprint: str
    index_fingerprint: str
    strategy: str = "hybrid_rerank"
    embedding_backend: str = "none"
    reranker_backend: str = "none"
    fusion_backend: str = "fixed-weight-v1"
    parser_backends: dict[str, int] = Field(default_factory=dict)
    top_k: int
    max_tokens: int | None = None
    total: int
    recall_at_k: float
    mrr: float
    ndcg_at_k: float
    mean_context_tokens: float = 0.0
    mean_selected_files: float = 0.0
    mean_compression_ratio: float = 0.0
    mean_duplicate_lines_removed: float = 0.0
    outcomes: list[ContextEvalOutcome]


class AblationPoint(BaseModel):
    strategy: str
    max_tokens: int
    recall_at_k: float
    mrr: float
    ndcg_at_k: float
    mean_context_tokens: float
    mean_selected_files: float
    mean_compression_ratio: float
    mean_duplicate_lines_removed: float


class ContextAblationReport(BaseModel):
    schema_version: str = "2.0"
    dataset_fingerprint: str
    index_fingerprint: str
    generated_by: str = "agentforge-code-context-eval"
    top_k: int
    strategies: list[str]
    token_budgets: list[int]
    embedding_backend: str
    reranker_backend: str
    fusion_backend: str = "fixed-weight-v1"
    parser_backends: dict[str, int]
    points: list[AblationPoint]


class DirectoryWorkspace:
    """Read-only evaluation corpus; Git roots use tracked, LF-normalized source."""

    EXCLUDED = {
        ".git", ".venv", "venv", "env", "node_modules", "data", "__pycache__",
        ".pytest_cache", ".ruff_cache", "conda", "rag_docs", "graph_rag_docs",
        "chroma_db", "graph_chroma_db",
    }
    EVALUATION_ONLY_ROOTS = {
        "tests", "evals", "docs", "media", ".github", "k8s", "docker", "repositories",
    }
    SOURCE_SUFFIXES = frozenset(LANGUAGE_BY_SUFFIX) | {
        ".sh", ".ps1", ".html", ".css", ".json", ".toml", ".yaml", ".yml", ".ini", ".cfg",
    }

    def __init__(self, root: Path, *, source_only: bool = True):
        self.root = root.resolve()
        self.source_only = source_only

    def list_files(self, limit: int = 500) -> list[str]:
        files = []
        if self.source_only and (self.root / ".git").exists():
            try:
                tracked = subprocess.run(
                    ["git", "-C", str(self.root), "ls-files", "-z", "--cached"],
                    check=True, capture_output=True, timeout=10,
                )
            except (OSError, subprocess.SubprocessError) as exc:
                raise ValueError("cannot enumerate tracked evaluation source files") from exc
            candidates = (
                self.root / name
                for name in tracked.stdout.decode("utf-8").split("\0") if name
            )
        else:
            candidates = self.root.rglob("*")
        for path in candidates:
            if path.is_symlink() or not path.is_file():
                continue
            relative = path.relative_to(self.root)
            # A tracked file can still be beneath a replaced/symlinked directory.
            if not path.resolve().is_relative_to(self.root):
                continue
            if any(part in self.EXCLUDED or part.endswith(".egg-info") for part in relative.parts):
                continue
            if (
                self.source_only
                and relative.parts
                and relative.parts[0] in self.EVALUATION_ONLY_ROOTS
            ):
                continue
            if self.source_only and relative.parts[:2] == ("scripts", "ci"):
                continue
            if self.source_only and len(relative.parts) == 1:
                continue
            is_dependency_list = relative.name.lower().startswith("requirements") and relative.suffix.lower() == ".txt"
            if self.source_only and relative.suffix.lower() not in self.SOURCE_SUFFIXES and not is_dependency_list:
                continue
            if self.source_only and (
                relative.name.startswith("test_") or relative.name.endswith("_test.py")
            ):
                continue
            if path.stat().st_size > 250_000:
                continue
            files.append(relative.as_posix())
        return sorted(files)[: max(1, min(limit, 5000))]

    def read_file(self, relative_path: str) -> str:
        path = (self.root / relative_path).resolve()
        path.relative_to(self.root)
        data = path.read_bytes()
        if b"\0" in data[:4096]:
            raise ValueError("binary file")
        text = data.decode("utf-8", errors="replace")
        return text.replace("\r\n", "\n").replace("\r", "\n") if self.source_only else text


def load_cases(path: Path) -> list[ContextEvalCase]:
    cases = [
        ContextEvalCase.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not cases:
        raise ValueError("context evaluation dataset is empty")
    ids = [case.id for case in cases]
    if len(ids) != len(set(ids)):
        raise ValueError("context evaluation case IDs must be unique")
    return cases


def dataset_fingerprint(cases: list[ContextEvalCase]) -> str:
    payload = json.dumps(
        [case.model_dump(mode="json") for case in cases],
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def index_fingerprint(index: CodeIntelligenceIndex) -> str:
    payload = [(path, item.sha256) for path, item in sorted(index.files.items())]
    return hashlib.sha256(json.dumps(payload, separators=(",", ":")).encode("utf-8")).hexdigest()


def evaluate_context(
    index: CodeIntelligenceIndex,
    cases: list[ContextEvalCase],
    *,
    top_k: int = 8,
    max_tokens: int | None = None,
    strategy: str = "hybrid_rerank",
) -> ContextEvalReport:
    outcomes = []
    for case in cases:
        pack = index.select(
            case.query,
            top_k=top_k,
            max_tokens=max_tokens,
            strategy=strategy,
        )
        selected = [item.path for item in pack.receipt.selected_files]
        relevant = set(case.relevant_paths)
        hits = [path for path in selected if path in relevant]
        recall = len(set(hits)) / len(relevant)
        first_rank = next(
            (rank for rank, path in enumerate(selected, start=1) if path in relevant),
            0,
        )
        reciprocal_rank = 1 / first_rank if first_rank else 0.0
        dcg = sum(
            1 / math.log2(rank + 1)
            for rank, path in enumerate(selected, start=1)
            if path in relevant
        )
        ideal_count = min(len(relevant), top_k)
        ideal = sum(1 / math.log2(rank + 1) for rank in range(1, ideal_count + 1))
        outcomes.append(
            ContextEvalOutcome(
                case_id=case.id,
                selected_paths=selected,
                relevant_paths=case.relevant_paths,
                recall_at_k=recall,
                reciprocal_rank=reciprocal_rank,
                ndcg_at_k=dcg / ideal if ideal else 0.0,
                context_tokens=pack.receipt.estimated_tokens,
                selected_files=len(selected),
                compression_ratio=(
                    1 - pack.receipt.context_chars / pack.receipt.original_context_chars
                    if pack.receipt.original_context_chars
                    else 0.0
                ),
                duplicate_lines_removed=pack.receipt.duplicate_lines_removed,
            )
        )
    total = len(outcomes)
    return ContextEvalReport(
        dataset_fingerprint=dataset_fingerprint(cases),
        index_fingerprint=index_fingerprint(index),
        strategy=strategy,
        embedding_backend=index.embedder.name if index.embedder else "none",
        reranker_backend=index.reranker.name if index.reranker else "none",
        fusion_backend=(index.fusion_scorer.name if index.fusion_scorer else "fixed-weight-v1"),
        parser_backends=dict(index.stats.parser_backends),
        top_k=top_k,
        max_tokens=max_tokens,
        total=total,
        recall_at_k=sum(item.recall_at_k for item in outcomes) / total,
        mrr=sum(item.reciprocal_rank for item in outcomes) / total,
        ndcg_at_k=sum(item.ndcg_at_k for item in outcomes) / total,
        mean_context_tokens=sum(item.context_tokens for item in outcomes) / total,
        mean_selected_files=sum(item.selected_files for item in outcomes) / total,
        mean_compression_ratio=sum(item.compression_ratio for item in outcomes) / total,
        mean_duplicate_lines_removed=(
            sum(item.duplicate_lines_removed for item in outcomes) / total
        ),
        outcomes=outcomes,
    )


def run_ablation(
    index: CodeIntelligenceIndex,
    cases: list[ContextEvalCase],
    *,
    top_k: int = 8,
    strategies: list[str] | None = None,
    token_budgets: list[int] | None = None,
) -> ContextAblationReport:
    strategies = strategies or list(STRATEGIES)
    token_budgets = token_budgets or [512, 1024, 2048, 4096]
    unknown = sorted(set(strategies) - set(STRATEGIES))
    if unknown:
        raise ValueError("unknown strategies: " + ", ".join(unknown))
    points = []
    for strategy in strategies:
        for budget in token_budgets:
            report = evaluate_context(
                index,
                cases,
                top_k=top_k,
                max_tokens=max(64, budget),
                strategy=strategy,
            )
            points.append(
                AblationPoint(
                    strategy=strategy,
                    max_tokens=max(64, budget),
                    recall_at_k=report.recall_at_k,
                    mrr=report.mrr,
                    ndcg_at_k=report.ndcg_at_k,
                    mean_context_tokens=report.mean_context_tokens,
                    mean_selected_files=report.mean_selected_files,
                    mean_compression_ratio=report.mean_compression_ratio,
                    mean_duplicate_lines_removed=report.mean_duplicate_lines_removed,
                )
            )
    return ContextAblationReport(
        dataset_fingerprint=dataset_fingerprint(cases),
        index_fingerprint=index_fingerprint(index),
        top_k=top_k,
        strategies=strategies,
        token_budgets=token_budgets,
        embedding_backend=index.embedder.name if index.embedder else "none",
        reranker_backend=index.reranker.name if index.reranker else "none",
        fusion_backend=(index.fusion_scorer.name if index.fusion_scorer else "fixed-weight-v1"),
        parser_backends=dict(index.stats.parser_backends),
        points=points,
    )


def _csv_strings(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _csv_ints(value: str) -> list[int]:
    return [max(64, int(item)) for item in _csv_strings(value)]


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate hybrid code-context retrieval")
    parser.add_argument(
        "dataset",
        type=Path,
        nargs="?",
        default=Path("evals/datasets/code_context_smoke.jsonl"),
    )
    parser.add_argument("--repository-root", type=Path, default=Path("."))
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--max-files", type=int, default=1000)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--strategy", choices=STRATEGIES, default="hybrid_rerank")
    parser.add_argument("--embedding-backend", default="hashing")
    parser.add_argument("--reranker-backend", default="feature")
    parser.add_argument("--fusion-artifact", default="")
    parser.add_argument("--no-tree-sitter", action="store_true")
    parser.add_argument("--ablation", action="store_true")
    parser.add_argument("--strategies", default=",".join(STRATEGIES))
    parser.add_argument("--token-budgets", default="512,1024,2048,4096")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--min-recall", type=float, default=0.0)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    cases = load_cases(args.dataset)
    if args.validate_only:
        print(
            json.dumps(
                {"cases": len(cases), "dataset_fingerprint": dataset_fingerprint(cases)},
                sort_keys=True,
            )
        )
        return
    config = RetrievalConfig(
        embedding_backend=args.embedding_backend,
        reranker_backend=args.reranker_backend,
        fusion_artifact=args.fusion_artifact,
        prefer_tree_sitter=not args.no_tree_sitter,
    )
    workspace = DirectoryWorkspace(args.repository_root, source_only=True)
    index = CodeIntelligenceIndex.build(
        workspace,
        max_files=max(1, min(args.max_files, 5000)),
        config=config,
    )
    if args.ablation:
        report = run_ablation(
            index,
            cases,
            top_k=max(1, min(args.top_k, 30)),
            strategies=_csv_strings(args.strategies),
            token_budgets=_csv_ints(args.token_budgets),
        )
        recall = max((point.recall_at_k for point in report.points), default=0.0)
    else:
        report = evaluate_context(
            index,
            cases,
            top_k=max(1, min(args.top_k, 30)),
            max_tokens=args.max_tokens,
            strategy=args.strategy,
        )
        recall = report.recall_at_k
    payload = report.model_dump_json(indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")
    if recall < max(0.0, min(args.min_recall, 1.0)):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
