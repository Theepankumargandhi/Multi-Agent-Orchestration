"""Dependency-free low-rank adapter optimization used as the CI training smoke test."""

from __future__ import annotations

import json
import math
import random
from pathlib import Path
from uuid import uuid4

from post_training.dataset import canonical_json_sha256, sha256_file
from post_training.models import ArtifactDigest, TrainingRunManifest, utc_now


def _matmul(left: list[list[float]], right: list[list[float]]) -> list[list[float]]:
    return [
        [sum(left[row][k] * right[k][column] for k in range(len(right))) for column in range(len(right[0]))]
        for row in range(len(left))
    ]


def _loss(
    examples: list[tuple[list[float], int]],
    base: list[list[float]],
    adapter_b: list[list[float]],
    adapter_a: list[list[float]],
) -> float:
    adapter = _matmul(adapter_b, adapter_a)
    weight = [
        [base[row][column] + adapter[row][column] for column in range(len(base[0]))]
        for row in range(len(base))
    ]
    losses = []
    for features, label in examples:
        logits = [sum(value * feature for value, feature in zip(row, features, strict=True)) for row in weight]
        maximum = max(logits)
        denominator = sum(math.exp(value - maximum) for value in logits)
        probability = math.exp(logits[label] - maximum) / denominator
        losses.append(-math.log(max(probability, 1e-12)))
    return sum(losses) / len(losses)


def run_lora_smoke(output_dir: Path, *, seed: int = 42, epochs: int = 180) -> TrainingRunManifest:
    """Optimize only rank-two matrices while a tiny base matrix remains frozen."""
    if epochs < 1:
        raise ValueError("epochs must be positive")
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    input_size, output_size, rank = 4, 3, 2
    base = [[0.0 for _ in range(input_size)] for _ in range(output_size)]
    adapter_a = [[rng.uniform(-0.2, 0.2) for _ in range(input_size)] for _ in range(rank)]
    adapter_b = [[0.0 for _ in range(rank)] for _ in range(output_size)]
    examples = [
        ([1.0, 0.0, 0.0, 0.2], 0),
        ([0.8, 0.1, 0.0, 0.1], 0),
        ([0.0, 1.0, 0.0, 0.2], 1),
        ([0.1, 0.8, 0.0, 0.1], 1),
        ([0.0, 0.0, 1.0, 0.2], 2),
        ([0.0, 0.1, 0.8, 0.1], 2),
    ]
    initial_loss = _loss(examples, base, adapter_b, adapter_a)
    learning_rate = 0.35
    for _ in range(epochs):
        grad_a = [[0.0 for _ in range(input_size)] for _ in range(rank)]
        grad_b = [[0.0 for _ in range(rank)] for _ in range(output_size)]
        adapter = _matmul(adapter_b, adapter_a)
        weight = [
            [base[row][column] + adapter[row][column] for column in range(input_size)]
            for row in range(output_size)
        ]
        for features, label in examples:
            logits = [sum(value * feature for value, feature in zip(row, features, strict=True)) for row in weight]
            maximum = max(logits)
            exponentials = [math.exp(value - maximum) for value in logits]
            denominator = sum(exponentials)
            errors = [value / denominator - float(index == label) for index, value in enumerate(exponentials)]
            grad_weight = [[error * feature for feature in features] for error in errors]
            for output in range(output_size):
                for low_rank in range(rank):
                    grad_b[output][low_rank] += sum(
                        grad_weight[output][column] * adapter_a[low_rank][column]
                        for column in range(input_size)
                    )
            for low_rank in range(rank):
                for column in range(input_size):
                    grad_a[low_rank][column] += sum(
                        adapter_b[output][low_rank] * grad_weight[output][column]
                        for output in range(output_size)
                    )
        scale = learning_rate / len(examples)
        for output in range(output_size):
            for low_rank in range(rank):
                adapter_b[output][low_rank] -= scale * grad_b[output][low_rank]
        for low_rank in range(rank):
            for column in range(input_size):
                adapter_a[low_rank][column] -= scale * grad_a[low_rank][column]
    final_loss = _loss(examples, base, adapter_b, adapter_a)
    if not final_loss < initial_loss * 0.35:
        raise RuntimeError(f"low-rank smoke optimization did not converge: {initial_loss} -> {final_loss}")

    adapter_path = output_dir / "adapter.json"
    metrics_path = output_dir / "metrics.json"
    card_path = output_dir / "MODEL_CARD.md"
    adapter_payload = {
        "format": "agentforge-low-rank-smoke-v1",
        "rank": rank,
        "base_frozen": True,
        "adapter_a": adapter_a,
        "adapter_b": adapter_b,
    }
    adapter_path.write_text(json.dumps(adapter_payload, indent=2) + "\n", encoding="utf-8")
    metrics = {
        "initial_loss": initial_loss,
        "final_loss": final_loss,
        "loss_reduction": 1 - final_loss / initial_loss,
        "epochs": epochs,
    }
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    card_path.write_text(
        "# AgentForge low-rank training smoke artifact\n\n"
        "This is a dependency-free mathematical CI fixture. It verifies frozen-base, "
        "low-rank optimization, artifact hashing, and registry plumbing. It is not a "
        "language model and must not be reported as an LLM fine-tuning result.\n",
        encoding="utf-8",
    )
    artifacts = [
        ArtifactDigest(path=path.name, sha256=sha256_file(path), size_bytes=path.stat().st_size)
        for path in (adapter_path, metrics_path, card_path)
    ]
    created = utc_now()
    unsigned = {
        "base_model": "frozen-linear-fixture",
        "dataset": canonical_json_sha256({"examples": examples}),
        "config": canonical_json_sha256({"seed": seed, "epochs": epochs, "rank": rank}),
        "artifacts": [artifact.model_dump() for artifact in artifacts],
        "metrics": metrics,
    }
    manifest = TrainingRunManifest(
        run_id=f"smoke-{uuid4()}",
        run_name="low-rank-ci-smoke",
        stage="lora_smoke",
        status="completed",
        created_at=created,
        completed_at=utc_now(),
        base_model="frozen-linear-fixture",
        dataset_id="three-class-ci-fixture",
        dataset_manifest_fingerprint=unsigned["dataset"],
        config_fingerprint=unsigned["config"],
        seed=seed,
        framework_versions={"python": "standard-library"},
        metrics=metrics,
        trainable_parameters=rank * input_size + output_size * rank,
        total_parameters=output_size * input_size + rank * input_size + output_size * rank,
        trainable_percentage=(rank * input_size + output_size * rank)
        / (output_size * input_size + rank * input_size + output_size * rank)
        * 100,
        artifacts=artifacts,
        output_dir=str(output_dir.resolve()),
        model_card_path=card_path.name,
        run_fingerprint=canonical_json_sha256(unsigned),
    )
    (output_dir / "run_manifest.json").write_text(
        manifest.model_dump_json(indent=2) + "\n", encoding="utf-8"
    )
    return manifest
