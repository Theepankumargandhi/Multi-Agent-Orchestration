"""Hard-negative mining and pairwise learning-to-rank for code retrieval."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from code_agent.context_evaluation import (
    ContextEvalCase,
    ContextEvalReport,
    DirectoryWorkspace,
    dataset_fingerprint,
    evaluate_context,
    index_fingerprint,
    load_cases,
)
from code_agent.intelligence import CodeIntelligenceIndex
from code_agent.retrieval_backends import (
    FUSION_FEATURES,
    LearnedFusionScorer,
    RetrievalConfig,
)


class HardNegativePair(BaseModel):
    case_id: str
    query: str
    positive_path: str
    negative_path: str
    positive_sha256: str
    negative_sha256: str
    positive_rank: int = Field(ge=1)
    negative_rank: int = Field(ge=1)
    positive_features: dict[str, float]
    negative_features: dict[str, float]
    baseline_margin: float


class HardNegativeReport(BaseModel):
    schema_version: str = "1.0"
    generated_by: str = "agentforge-retrieval-learning"
    dataset_fingerprint: str
    index_fingerprint: str
    embedding_backend: str
    reranker_backend: str
    cases: int
    pairs: int
    positives_not_retrieved: list[str] = Field(default_factory=list)
    source_content_included: bool = False
    hard_negatives: list[HardNegativePair]


class FusionArtifact(BaseModel):
    schema_version: str = "1.0"
    model_type: str = "pairwise-logistic-fusion"
    feature_names: list[str] = Field(default_factory=lambda: list(FUSION_FEATURES))
    weights: dict[str, float]
    dataset_fingerprint: str
    index_fingerprint: str
    training_pairs: int = Field(ge=1)
    hard_negatives_per_positive: int = Field(ge=1)
    epochs: int = Field(ge=1)
    learning_rate: float = Field(gt=0)
    l2: float = Field(ge=0)
    pairwise_accuracy: float = Field(ge=0, le=1)
    mean_pairwise_margin: float
    artifact_fingerprint: str = ""

    def seal(self) -> "FusionArtifact":
        unsigned = self.model_dump(mode="json", exclude={"artifact_fingerprint"})
        self.artifact_fingerprint = hashlib.sha256(
            json.dumps(unsigned, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        return self


class FusionValidationReport(BaseModel):
    schema_version: str = "1.0"
    artifact_fingerprint: str
    dataset_fingerprint: str
    index_fingerprint: str
    base: ContextEvalReport
    learned: ContextEvalReport
    recall_delta: float
    mrr_delta: float
    ndcg_delta: float
    promotion_approved: bool
    promotion_reasons: list[str]


def _feature_rows(
    index: CodeIntelligenceIndex,
    case: ContextEvalCase,
    *,
    candidate_limit: int,
) -> list[dict[str, object]]:
    pack = index.select(
        case.query,
        top_k=max(2, min(candidate_limit, 30)),
        max_chars=2_000_000,
        strategy="hybrid_rerank",
    )
    lexical_max = max(
        (item.lexical_score for item in pack.receipt.selected_files),
        default=1.0,
    ) or 1.0
    return [
        {
            "path": item.path,
            "rank": item.rank,
            "sha256": item.sha256,
            "score": item.score,
            "features": {
                "lexical": item.lexical_score / lexical_max,
                "semantic": item.semantic_score,
                "graph": item.graph_score,
                "rerank": item.rerank_score,
            },
        }
        for item in pack.receipt.selected_files
    ]


def mine_hard_negatives(
    index: CodeIntelligenceIndex,
    cases: list[ContextEvalCase],
    *,
    negatives_per_positive: int = 3,
    candidate_limit: int = 30,
) -> HardNegativeReport:
    pairs: list[HardNegativePair] = []
    missing: list[str] = []
    per_positive = max(1, min(negatives_per_positive, 20))
    for case in cases:
        rows = _feature_rows(index, case, candidate_limit=candidate_limit)
        by_path = {str(row["path"]): row for row in rows}
        relevant = set(case.relevant_paths)
        negatives = [row for row in rows if row["path"] not in relevant]
        for positive_path in case.relevant_paths:
            positive = by_path.get(positive_path)
            if positive is None:
                missing.append(f"{case.id}:{positive_path}")
                continue
            for negative in negatives[:per_positive]:
                positive_features = dict(positive["features"])
                negative_features = dict(negative["features"])
                baseline_margin = float(positive["score"]) - float(negative["score"])
                pairs.append(
                    HardNegativePair(
                        case_id=case.id,
                        query=case.query,
                        positive_path=positive_path,
                        negative_path=str(negative["path"]),
                        positive_sha256=str(positive["sha256"]),
                        negative_sha256=str(negative["sha256"]),
                        positive_rank=int(positive["rank"]),
                        negative_rank=int(negative["rank"]),
                        positive_features=positive_features,
                        negative_features=negative_features,
                        baseline_margin=baseline_margin,
                    )
                )
    return HardNegativeReport(
        dataset_fingerprint=dataset_fingerprint(cases),
        index_fingerprint=index_fingerprint(index),
        embedding_backend=index.embedder.name if index.embedder else "none",
        reranker_backend=index.reranker.name if index.reranker else "none",
        cases=len(cases),
        pairs=len(pairs),
        positives_not_retrieved=sorted(missing),
        hard_negatives=pairs,
    )


def _difference(pair: HardNegativePair) -> tuple[float, ...]:
    return tuple(
        pair.positive_features.get(name, 0.0) - pair.negative_features.get(name, 0.0)
        for name in FUSION_FEATURES
    )


def train_pairwise_fusion(
    report: HardNegativeReport,
    *,
    negatives_per_positive: int,
    epochs: int = 300,
    learning_rate: float = 0.15,
    l2: float = 0.01,
) -> FusionArtifact:
    if not report.hard_negatives:
        raise ValueError("hard-negative report contains no trainable pairs")
    epochs = max(1, min(epochs, 10_000))
    learning_rate = max(1e-5, min(learning_rate, 2.0))
    l2 = max(0.0, min(l2, 1.0))
    differences = [_difference(pair) for pair in report.hard_negatives]
    weights = [0.5, 0.3, 0.2, 0.25]
    for epoch in range(epochs):
        gradient = [0.0] * len(FUSION_FEATURES)
        for difference in differences:
            margin = sum(weight * value for weight, value in zip(weights, difference, strict=True))
            factor = -1.0 / (1.0 + math.exp(max(-60.0, min(margin, 60.0))))
            for index, value in enumerate(difference):
                gradient[index] += factor * value
        step = learning_rate / math.sqrt(1.0 + epoch / 25.0)
        for index in range(len(weights)):
            gradient[index] = gradient[index] / len(differences) + l2 * weights[index]
            weights[index] -= step * gradient[index]

    margins = [
        sum(weight * value for weight, value in zip(weights, difference, strict=True))
        for difference in differences
    ]
    return FusionArtifact(
        weights={name: weights[index] for index, name in enumerate(FUSION_FEATURES)},
        dataset_fingerprint=report.dataset_fingerprint,
        index_fingerprint=report.index_fingerprint,
        training_pairs=len(differences),
        hard_negatives_per_positive=max(1, negatives_per_positive),
        epochs=epochs,
        learning_rate=learning_rate,
        l2=l2,
        pairwise_accuracy=sum(margin > 0 for margin in margins) / len(margins),
        mean_pairwise_margin=sum(margins) / len(margins),
    ).seal()


def check_artifact_reproducibility(generated: FusionArtifact, expected_path: Path) -> str:
    expected = FusionArtifact.model_validate_json(expected_path.read_text(encoding="utf-8"))
    LearnedFusionScorer.load(expected_path)
    if generated != expected:
        raise ValueError("fusion artifact is stale; retrain and review the changed weights")
    return expected.artifact_fingerprint


def validate_fusion(
    index: CodeIntelligenceIndex,
    cases: list[ContextEvalCase],
    artifact: FusionArtifact,
    *,
    top_k: int = 8,
    max_tokens: int = 4096,
) -> FusionValidationReport:
    scorer = LearnedFusionScorer(
        weights=tuple(float(artifact.weights[name]) for name in FUSION_FEATURES),
        artifact_fingerprint=artifact.artifact_fingerprint,
        source="in-memory-validation",
    )
    learned_index = CodeIntelligenceIndex(
        index.files,
        index.edges,
        index.stats,
        config=index.config,
        embedder=index.embedder,
        reranker=index.reranker,
        fusion_scorer=scorer,
        parser=index.parser,
    )
    base = evaluate_context(index, cases, top_k=top_k, max_tokens=max_tokens)
    learned = evaluate_context(learned_index, cases, top_k=top_k, max_tokens=max_tokens)
    reasons = []
    if learned.recall_at_k < base.recall_at_k:
        reasons.append("recall regressed")
    if learned.ndcg_at_k + 1e-12 < base.ndcg_at_k:
        reasons.append("NDCG regressed")
    if artifact.pairwise_accuracy < 0.5:
        reasons.append("pairwise training accuracy is below 0.5")
    return FusionValidationReport(
        artifact_fingerprint=artifact.artifact_fingerprint,
        dataset_fingerprint=dataset_fingerprint(cases),
        index_fingerprint=index_fingerprint(index),
        base=base,
        learned=learned,
        recall_delta=learned.recall_at_k - base.recall_at_k,
        mrr_delta=learned.mrr - base.mrr,
        ndcg_delta=learned.ndcg_at_k - base.ndcg_at_k,
        promotion_approved=not reasons,
        promotion_reasons=reasons,
    )


def fine_tune_neural_ranker(
    index: CodeIntelligenceIndex,
    report: HardNegativeReport,
    *,
    kind: Literal["bi-encoder", "cross-encoder"],
    model_id: str,
    output_dir: Path,
    epochs: int = 1,
    batch_size: int = 8,
) -> dict[str, object]:
    try:
        from sentence_transformers import CrossEncoder, InputExample, SentenceTransformer, losses
        from torch.utils.data import DataLoader
    except ImportError as exc:
        raise RuntimeError(
            "Install requirements-code-intelligence-ml.txt to fine-tune neural retrieval"
        ) from exc
    examples = []
    for pair in report.hard_negatives:
        positive = index.files[pair.positive_path].retrieval_text
        negative = index.files[pair.negative_path].retrieval_text
        if kind == "bi-encoder":
            examples.append(InputExample(texts=[pair.query, positive, negative]))
        else:
            examples.append(InputExample(texts=[pair.query, positive], label=1.0))
            examples.append(InputExample(texts=[pair.query, negative], label=0.0))
    if not examples:
        raise ValueError("no neural training examples were generated")
    output_dir.mkdir(parents=True, exist_ok=True)
    loader = DataLoader(examples, shuffle=True, batch_size=max(1, min(batch_size, 128)))
    if kind == "bi-encoder":
        model = SentenceTransformer(model_id)
        model.fit(
            train_objectives=[(loader, losses.TripletLoss(model=model))],
            epochs=max(1, min(epochs, 100)),
            output_path=str(output_dir),
            show_progress_bar=False,
        )
    else:
        model = CrossEncoder(model_id, num_labels=1)
        model.fit(
            train_dataloader=loader,
            epochs=max(1, min(epochs, 100)),
            output_path=str(output_dir),
            show_progress_bar=False,
        )
    manifest = {
        "schema_version": "1.0",
        "kind": kind,
        "base_model": model_id,
        "dataset_fingerprint": report.dataset_fingerprint,
        "index_fingerprint": report.index_fingerprint,
        "training_pairs": report.pairs,
        "epochs": max(1, min(epochs, 100)),
        "batch_size": max(1, min(batch_size, 128)),
    }
    manifest["fingerprint"] = hashlib.sha256(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    (output_dir / "agentforge_training_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def _write_json(path: Path | None, model: BaseModel) -> None:
    payload = model.model_dump_json(indent=2) + "\n"
    if path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(payload, encoding="utf-8")
    print(payload, end="")


def _add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "dataset",
        type=Path,
        nargs="?",
        default=Path("evals/datasets/code_context_learning.jsonl"),
    )
    parser.add_argument("--repository-root", type=Path, default=Path("."))
    parser.add_argument("--max-files", type=int, default=1000)
    parser.add_argument("--candidate-limit", type=int, default=30)
    parser.add_argument("--negatives-per-positive", type=int, default=3)
    parser.add_argument("--embedding-backend", default="hashing")
    parser.add_argument("--reranker-backend", default="feature")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and validate AgentForge code retrieval")
    commands = parser.add_subparsers(dest="command", required=True)
    mine = commands.add_parser("mine", help="Mine safe hard-negative path pairs.")
    _add_common_arguments(mine)
    mine.add_argument("--output", type=Path)
    train = commands.add_parser("train-fusion", help="Train a pairwise fusion ranker.")
    _add_common_arguments(train)
    train.add_argument("--epochs", type=int, default=300)
    train.add_argument("--learning-rate", type=float, default=0.15)
    train.add_argument("--l2", type=float, default=0.01)
    train_destination = train.add_mutually_exclusive_group(required=True)
    train_destination.add_argument("--output", type=Path)
    train_destination.add_argument("--check", type=Path)
    validate = commands.add_parser("validate-fusion", help="Compare a learned artifact with baseline.")
    _add_common_arguments(validate)
    validate.add_argument("--artifact", type=Path, required=True)
    validate.add_argument("--top-k", type=int, default=8)
    validate.add_argument("--max-tokens", type=int, default=4096)
    validate.add_argument("--require-promotion", action="store_true")
    validate.add_argument("--output", type=Path)
    neural = commands.add_parser("fine-tune", help="Fine-tune a bi-encoder or cross-encoder.")
    _add_common_arguments(neural)
    neural.add_argument("--kind", choices=["bi-encoder", "cross-encoder"], required=True)
    neural.add_argument("--model", required=True)
    neural.add_argument("--epochs", type=int, default=1)
    neural.add_argument("--batch-size", type=int, default=8)
    neural.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    cases = load_cases(args.dataset)
    config = RetrievalConfig(
        embedding_backend=args.embedding_backend,
        reranker_backend=args.reranker_backend,
    )
    index = CodeIntelligenceIndex.build(
        DirectoryWorkspace(args.repository_root, source_only=True),
        max_files=max(1, min(args.max_files, 5000)),
        config=config,
    )
    mined = mine_hard_negatives(
        index,
        cases,
        negatives_per_positive=args.negatives_per_positive,
        candidate_limit=args.candidate_limit,
    )
    if args.command == "mine":
        _write_json(args.output, mined)
        return
    if args.command == "train-fusion":
        artifact = train_pairwise_fusion(
            mined,
            negatives_per_positive=args.negatives_per_positive,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            l2=args.l2,
        )
        if args.check:
            fingerprint = check_artifact_reproducibility(artifact, args.check)
            print(json.dumps({"artifact_fingerprint": fingerprint, "reproducible": True}))
            return
        _write_json(args.output, artifact)
        return
    if args.command == "validate-fusion":
        artifact = FusionArtifact.model_validate_json(args.artifact.read_text(encoding="utf-8"))
        LearnedFusionScorer.load(args.artifact)
        report = validate_fusion(
            index,
            cases,
            artifact,
            top_k=max(1, min(args.top_k, 30)),
            max_tokens=max(64, args.max_tokens),
        )
        _write_json(args.output, report)
        if args.require_promotion and not report.promotion_approved:
            raise SystemExit(1)
        return
    manifest = fine_tune_neural_ranker(
        index,
        mined,
        kind=args.kind,
        model_id=args.model,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
