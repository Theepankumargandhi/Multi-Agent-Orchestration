import json
from pathlib import Path

import pytest

from evals.flywheel import DeploymentState, evaluate_promotion
from evals.platform import (
    AgentRun,
    CaseResult,
    ExpectedBehavior,
    ExperimentReport,
    MetricResult,
    VariantConfig,
    VariantReport,
)
from post_training.dataset import build_training_datasets, load_and_verify_manifest, load_preferences
from post_training.inference import InferenceResult, validate_inference_url
from post_training.models import TrainingConfig
from post_training.registry import ModelRegistry
from post_training.smoke import run_lora_smoke
from post_training.trainer import load_training_config, require_training_dependencies, verify_run_manifest


def _write_preferences(path: Path, count: int = 6) -> None:
    rows = [
        {
            "id": f"preference-{index}",
            "prompt": f"Explain reviewed agent behavior number {index}",
            "chosen": f"Grounded reviewed answer {index}",
            "rejected": f"Unsupported answer {index}",
            "source_trace_fingerprints": [f"{'a' * 63}{index}"],
            "provenance": "human_correction",
            "reviewer": "reviewer",
        }
        for index in range(count)
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def test_dataset_builder_creates_reproducible_splits_and_integrity_manifest(tmp_path: Path):
    source = tmp_path / "preferences.jsonl"
    _write_preferences(source)
    protected = tmp_path / "protected.jsonl"
    protected.write_text('{"id":"hidden","input":"A completely separate hidden task"}\n', encoding="utf-8")
    output = tmp_path / "dataset"
    manifest = build_training_datasets(source, output, protected_paths=[protected])
    assert manifest.split_counts == {"train": 4, "validation": 1, "test": 1}
    assert manifest.all_records_human_reviewed
    assert len(manifest.manifest_fingerprint) == 64
    assert load_and_verify_manifest(output / "manifest.json") == manifest
    rows = [json.loads(line) for line in (output / "sft.jsonl").read_text(encoding="utf-8").splitlines()]
    assert all(row["messages"][-1]["role"] == "assistant" for row in rows)


def test_dataset_builder_blocks_exact_protected_set_leakage(tmp_path: Path):
    source = tmp_path / "preferences.jsonl"
    _write_preferences(source, count=3)
    protected = tmp_path / "protected.jsonl"
    protected.write_text(
        '{"id":"hidden","input":"Explain reviewed agent behavior number 1"}\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="contamination"):
        build_training_datasets(source, tmp_path / "dataset", protected_paths=[protected])


def test_preference_loader_rejects_conflicting_duplicate_labels(tmp_path: Path):
    source = tmp_path / "preferences.jsonl"
    _write_preferences(source, count=1)
    row = json.loads(source.read_text(encoding="utf-8"))
    row["id"] = "conflict"
    row["chosen"] = "A contradictory choice"
    with source.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row) + "\n")
    with pytest.raises(ValueError, match="conflicting preference"):
        load_preferences(source)


def test_low_rank_smoke_training_reduces_loss_and_hashes_artifacts(tmp_path: Path):
    manifest = run_lora_smoke(tmp_path / "smoke", seed=7)
    assert float(manifest.metrics["loss_reduction"]) > 0.65
    assert manifest.trainable_parameters < manifest.total_parameters
    path = tmp_path / "smoke" / "run_manifest.json"
    assert verify_run_manifest(path).run_fingerprint == manifest.run_fingerprint
    adapter = json.loads((tmp_path / "smoke" / "adapter.json").read_text(encoding="utf-8"))
    assert adapter["base_frozen"] is True


def _case(case_id: str, passed: bool) -> CaseResult:
    return CaseResult(
        case_id=case_id,
        input="reviewed prompt",
        expected=ExpectedBehavior(answer_contains=["grounded"]),
        actual=AgentRun(answer="grounded" if passed else "wrong", latency_ms=10),
        metrics={
            "required_term_recall": MetricResult(
                score=float(passed), passed=passed, detail="fixture"
            )
        },
        quality_score=float(passed),
        passed=passed,
        split="validation",
        review_status="human_reviewed",
    )


def _variant(name: str, passed: bool) -> VariantReport:
    case = _case("one", passed)
    return VariantReport(
        variant=VariantConfig(name=name, adapter="openai-compatible", model=name),
        quality_score=case.quality_score,
        pass_rate=float(case.passed),
        metric_scores={"required_term_recall": case.quality_score},
        p50_latency_ms=10,
        p95_latency_ms=10,
        total_cost_usd=0,
        total_tokens=10,
        failure_categories={} if passed else {"answer_grounding_failure": 1},
        cases=[case],
    )


def test_registry_links_training_lineage_offline_gate_and_deployment(tmp_path: Path):
    manifest = run_lora_smoke(tmp_path / "smoke")
    registry = ModelRegistry(tmp_path / "registry.json")
    record = registry.register("smoke-v1", tmp_path / "smoke" / "run_manifest.json")
    assert record.run_fingerprint == manifest.run_fingerprint
    report = ExperimentReport(
        experiment_id="22222222-2222-2222-2222-222222222222",
        experiment_name="adapter-ablation",
        created_at="2026-09-10T00:00:00Z",
        dataset_path="reviewed.jsonl",
        dataset_fingerprint="b" * 64,
        winner="smoke-v1",
        pass_threshold=0.8,
        review_status_counts={"human_reviewed": 1},
        reports=[_variant("base", False), _variant("smoke-v1", True)],
    )
    decision = evaluate_promotion(report, "base", "smoke-v1")
    canary = registry.promote_to_canary("smoke-v1", decision)
    assert canary.canary_version == "smoke-v1"
    deployment = DeploymentState(
        active_version="smoke-v1",
        previous_version="base",
        status="stable",
        decision_fingerprint=decision.decision_fingerprint,
    )
    production = registry.sync_deployment(deployment)
    assert production.production_version == "smoke-v1"
    assert production.models[0].status == "production"


def test_registry_and_manifest_detect_duplicates_and_tampering(tmp_path: Path):
    run_lora_smoke(tmp_path / "smoke")
    manifest_path = tmp_path / "smoke" / "run_manifest.json"
    registry = ModelRegistry(tmp_path / "registry.json")
    registry.register("v1", manifest_path)
    with pytest.raises(ValueError, match="already exists"):
        registry.register("v1", manifest_path)
    (tmp_path / "smoke" / "adapter.json").write_text("tampered", encoding="utf-8")
    with pytest.raises(ValueError, match="integrity"):
        verify_run_manifest(manifest_path)


def test_inference_url_is_local_by_default():
    assert validate_inference_url("http://127.0.0.1:8001/v1").endswith("/v1")
    with pytest.raises(ValueError, match="allow_remote"):
        validate_inference_url("https://models.example.com/v1")


def test_training_config_paths_resolve_relative_to_config(tmp_path: Path):
    config_path = tmp_path / "configs" / "sft.json"
    config_path.parent.mkdir()
    payload = TrainingConfig(
        run_name="test",
        stage="sft",
        base_model="fixture",
        dataset_manifest=Path("../dataset/manifest.json"),
        output_dir=Path("../runs/test"),
    )
    config_path.write_text(payload.model_dump_json(), encoding="utf-8")
    loaded = load_training_config(config_path)
    assert loaded.dataset_manifest == (tmp_path / "dataset" / "manifest.json").resolve()
    assert loaded.output_dir == (tmp_path / "runs" / "test").resolve()


def test_missing_heavy_training_dependencies_have_actionable_error(monkeypatch):
    def missing(_package):
        raise ImportError

    monkeypatch.setattr("post_training.trainer.importlib.import_module", missing)
    with pytest.raises(RuntimeError, match="requirements-post-training.txt"):
        require_training_dependencies()


@pytest.mark.asyncio
async def test_openai_compatible_eval_adapter_records_usage(monkeypatch):
    async def fake_generate(**kwargs):
        assert kwargs["base_url"] == "http://127.0.0.1:8001/v1"
        return InferenceResult(
            answer="grounded response",
            prompt_tokens=11,
            completion_tokens=7,
            model="adapter-v1",
            tool_calls=("search_local",),
        )

    monkeypatch.setattr("post_training.inference.generate", fake_generate)
    from evals.platform import ADAPTERS, EvalCase

    run = await ADAPTERS["openai-compatible"].run(
        EvalCase(id="local", input="question", expected=ExpectedBehavior(answer_contains=["grounded"])),
        VariantConfig(
            name="adapter-v1",
            adapter="openai-compatible",
            model="adapter-v1",
            parameters={"base_url": "http://127.0.0.1:8001/v1"},
        ),
    )
    assert run.answer == "grounded response"
    assert run.prompt_tokens == 11
    assert run.completion_tokens == 7
    assert run.tool_calls == ["search_local"]
