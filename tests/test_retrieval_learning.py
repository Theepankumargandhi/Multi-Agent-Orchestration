from __future__ import annotations

import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from code_agent.context_evaluation import ContextEvalCase, DirectoryWorkspace
from code_agent.intelligence import CodeIntelligenceIndex
from code_agent.retrieval_backends import LearnedFusionScorer, RetrievalConfig
from code_agent.retrieval_learning import (
    FusionArtifact,
    check_artifact_reproducibility,
    fine_tune_neural_ranker,
    mine_hard_negatives,
    train_pairwise_fusion,
    validate_fusion,
)


class MemoryWorkspace:
    def __init__(self):
        self.files = {
            "code/worker.py": (
                "class LeaseWorker:\n"
                "    def claim_lease(self): return True\n"
                "    def heartbeat(self): return True\n"
            ),
            "code/queue.py": "def enqueue_job(job): return job\n",
            "code/security.py": "def reject_secret(value): return 'token' not in value\n",
            "code/math.py": "def add(left, right): return left + right\n",
            "code/formatting.py": "def format_name(value): return value.title()\n",
        }

    def list_files(self, limit: int = 500) -> list[str]:
        return sorted(self.files)[:limit]

    def read_file(self, relative_path: str) -> str:
        return self.files[relative_path]


def _cases() -> list[ContextEvalCase]:
    return [
        ContextEvalCase(
            id="lease",
            query="claim worker lease and send heartbeat",
            relevant_paths=["code/worker.py"],
        ),
        ContextEvalCase(
            id="security",
            query="reject secrets and unsafe tokens",
            relevant_paths=["code/security.py"],
        ),
    ]


def test_hard_negative_mining_never_persists_source_content():
    index = CodeIntelligenceIndex.build(MemoryWorkspace())
    report = mine_hard_negatives(index, _cases(), negatives_per_positive=2)

    assert report.pairs == 4
    assert report.source_content_included is False
    assert not report.positives_not_retrieved
    assert all(pair.positive_path != pair.negative_path for pair in report.hard_negatives)
    assert all(pair.negative_path not in _cases()[0].relevant_paths for pair in report.hard_negatives[:2])
    assert "LeaseWorker" not in report.model_dump_json()


def test_pairwise_fusion_artifact_is_deterministic_and_integrity_checked(tmp_path: Path, monkeypatch):
    index = CodeIntelligenceIndex.build(MemoryWorkspace())
    mined = mine_hard_negatives(index, _cases(), negatives_per_positive=2)
    first = train_pairwise_fusion(mined, negatives_per_positive=2, epochs=80)

    def legacy_sum(items, start=0):
        for item in items:
            start += item
        return start

    # Exercise the older summation behavior without requiring two Python installs.
    monkeypatch.setattr("code_agent.retrieval_learning.sum", legacy_sum, raising=False)
    monkeypatch.setattr("code_agent.retrieval_backends.sum", legacy_sum, raising=False)
    legacy_index = CodeIntelligenceIndex.build(MemoryWorkspace())
    assert mine_hard_negatives(legacy_index, _cases(), negatives_per_positive=2) == mined
    second = train_pairwise_fusion(mined, negatives_per_positive=2, epochs=80)

    assert first == second
    assert first.pairwise_accuracy >= 0.5
    artifact_path = tmp_path / "fusion.json"
    artifact_path.write_text(first.model_dump_json(indent=2) + "\n", encoding="utf-8")
    scorer = LearnedFusionScorer.load(artifact_path)
    assert scorer.name.startswith("pairwise-logistic:")
    assert 0 <= scorer.score({"lexical": 1.0}) <= 1
    assert check_artifact_reproducibility(first, artifact_path) == first.artifact_fingerprint

    tampered = json.loads(artifact_path.read_text(encoding="utf-8"))
    tampered["weights"]["lexical"] += 1
    artifact_path.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(ValueError, match="fingerprint"):
        LearnedFusionScorer.load(artifact_path)
    with pytest.raises(ValueError):
        check_artifact_reproducibility(first, artifact_path)


def test_stale_artifact_diagnostic_identifies_corpus_and_weight_changes(tmp_path: Path):
    index = CodeIntelligenceIndex.build(MemoryWorkspace())
    report = mine_hard_negatives(index, _cases(), negatives_per_positive=2)
    artifact = train_pairwise_fusion(report, negatives_per_positive=2, epochs=10)
    destination = tmp_path / "fusion.json"
    destination.write_text(artifact.model_dump_json(), encoding="utf-8")
    changed = artifact.model_copy(deep=True)
    changed.index_fingerprint = "new-corpus"
    changed.weights["lexical"] += 0.01
    changed.seal()
    with pytest.raises(ValueError, match="changed fields: weights, index_fingerprint"):
        check_artifact_reproducibility(changed, destination)


def test_fusion_training_is_independent_of_checkout_newlines_and_prose(tmp_path: Path):
    fixture = MemoryWorkspace()
    for name, content in fixture.files.items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content.replace("\n", "\r\n").encode("utf-8"))
    config = RetrievalConfig(prefer_tree_sitter=False)
    first_index = CodeIntelligenceIndex.build(DirectoryWorkspace(tmp_path), config=config)
    first = train_pairwise_fusion(
        mine_hard_negatives(first_index, _cases(), negatives_per_positive=2),
        negatives_per_positive=2, epochs=10,
    )
    for name, content in fixture.files.items():
        (tmp_path / name).write_bytes(content.encode("utf-8"))
    (tmp_path / "code" / "README.md").write_text("lease secret heartbeat " * 1000, encoding="utf-8")
    second_index = CodeIntelligenceIndex.build(DirectoryWorkspace(tmp_path), config=config)
    second = train_pairwise_fusion(
        mine_hard_negatives(second_index, _cases(), negatives_per_positive=2),
        negatives_per_positive=2, epochs=10,
    )
    assert first == second


def test_checked_in_fusion_matches_current_tracked_source_and_regression_gate():
    root = Path(__file__).resolve().parents[1]
    from code_agent.context_evaluation import load_cases

    index = CodeIntelligenceIndex.build(
        DirectoryWorkspace(root),
        config=RetrievalConfig(embedding_backend="hashing", reranker_backend="feature"),
    )
    artifact_path = root / "evals/experiments/code_context_fusion.json"
    expected = FusionArtifact.model_validate_json(artifact_path.read_text(encoding="utf-8"))
    generated = train_pairwise_fusion(
        mine_hard_negatives(index, load_cases(root / "evals/datasets/code_context_learning.jsonl"),
                            negatives_per_positive=4, candidate_limit=30),
        negatives_per_positive=4, epochs=400,
    )
    assert check_artifact_reproducibility(generated, artifact_path) == expected.artifact_fingerprint
    regression = validate_fusion(index, load_cases(root / "evals/datasets/code_context_smoke.jsonl"), generated)
    assert regression.base.recall_at_k >= 0.90
    # Reproducibility is not approval: an experimental model can still regress.
    if regression.ndcg_delta < -1e-12:
        assert not regression.promotion_approved
        assert "NDCG regressed" in regression.promotion_reasons


def test_learned_fusion_is_loaded_into_context_receipts(tmp_path: Path):
    baseline = CodeIntelligenceIndex.build(MemoryWorkspace())
    mined = mine_hard_negatives(baseline, _cases(), negatives_per_positive=2)
    artifact = train_pairwise_fusion(mined, negatives_per_positive=2, epochs=50)
    artifact_path = tmp_path / "fusion.json"
    artifact_path.write_text(artifact.model_dump_json(indent=2) + "\n", encoding="utf-8")

    learned = CodeIntelligenceIndex.build(
        MemoryWorkspace(),
        config=RetrievalConfig(fusion_artifact=str(artifact_path)),
    )
    pack = learned.select("claim lease worker heartbeat", top_k=3)
    assert pack.receipt.fusion_backend.startswith("pairwise-logistic:")
    assert pack.receipt.selected_files[0].score <= 1.0
    assert not pack.receipt.fallbacks


def test_fusion_validation_compares_baseline_and_learned_rankers():
    index = CodeIntelligenceIndex.build(MemoryWorkspace())
    cases = _cases()
    mined = mine_hard_negatives(index, cases, negatives_per_positive=2)
    artifact = train_pairwise_fusion(mined, negatives_per_positive=2, epochs=50)
    report = validate_fusion(index, cases, artifact, top_k=3, max_tokens=512)

    assert report.artifact_fingerprint == artifact.artifact_fingerprint
    assert report.base.total == report.learned.total == 2
    assert report.learned.fusion_backend.startswith("pairwise-logistic:")


def test_optional_bi_encoder_fine_tuning_writes_lineage_manifest(
    tmp_path: Path,
    monkeypatch,
):
    trained: dict[str, object] = {}

    class FakeInputExample:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class FakeSentenceTransformer:
        def __init__(self, model_id):
            trained["model"] = model_id

        def fit(self, **kwargs):
            trained["fit"] = kwargs

    class FakeCrossEncoder(FakeSentenceTransformer):
        def __init__(self, model_id, num_labels=1):
            super().__init__(model_id)
            trained["num_labels"] = num_labels

    class FakeDataLoader:
        def __init__(self, examples, **kwargs):
            self.examples = examples
            self.kwargs = kwargs

    sentence_module = ModuleType("sentence_transformers")
    sentence_module.CrossEncoder = FakeCrossEncoder
    sentence_module.InputExample = FakeInputExample
    sentence_module.SentenceTransformer = FakeSentenceTransformer
    sentence_module.losses = SimpleNamespace(TripletLoss=lambda model: ("triplet", model))
    torch_module = ModuleType("torch")
    torch_utils = ModuleType("torch.utils")
    torch_data = ModuleType("torch.utils.data")
    torch_data.DataLoader = FakeDataLoader
    monkeypatch.setitem(sys.modules, "sentence_transformers", sentence_module)
    monkeypatch.setitem(sys.modules, "torch", torch_module)
    monkeypatch.setitem(sys.modules, "torch.utils", torch_utils)
    monkeypatch.setitem(sys.modules, "torch.utils.data", torch_data)

    index = CodeIntelligenceIndex.build(MemoryWorkspace())
    mined = mine_hard_negatives(index, _cases(), negatives_per_positive=1)
    manifest = fine_tune_neural_ranker(
        index,
        mined,
        kind="bi-encoder",
        model_id="local-code-model",
        output_dir=tmp_path / "adapter",
    )
    assert trained["model"] == "local-code-model"
    assert manifest["training_pairs"] == mined.pairs
    assert (tmp_path / "adapter" / "agentforge_training_manifest.json").is_file()
