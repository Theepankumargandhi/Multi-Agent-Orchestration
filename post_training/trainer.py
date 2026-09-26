"""Optional TRL/PEFT training runner with reproducible lineage manifests."""

from __future__ import annotations

import importlib
import importlib.metadata
import inspect
import json
import os
import random
from pathlib import Path
from typing import Any
from uuid import uuid4

from post_training.dataset import canonical_json_sha256, load_and_verify_manifest, sha256_file
from post_training.models import ArtifactDigest, TrainingConfig, TrainingRunManifest, utc_now

OPTIONAL_PACKAGES = ("torch", "transformers", "datasets", "accelerate", "peft", "trl")


def require_training_dependencies() -> dict[str, str]:
    missing = []
    versions = {}
    for package in OPTIONAL_PACKAGES:
        try:
            importlib.import_module(package)
            versions[package] = importlib.metadata.version(package)
        except (ImportError, importlib.metadata.PackageNotFoundError):
            missing.append(package)
    if missing:
        raise RuntimeError(
            "post-training dependencies are missing: "
            + ", ".join(missing)
            + ". Install requirements-post-training.txt in a dedicated GPU environment."
        )
    return versions


def load_training_config(path: Path) -> TrainingConfig:
    config = TrainingConfig.model_validate_json(path.read_text(encoding="utf-8"))
    root = path.parent.resolve()
    if not config.dataset_manifest.is_absolute():
        config.dataset_manifest = (root / config.dataset_manifest).resolve()
    if not config.output_dir.is_absolute():
        config.output_dir = (root / config.output_dir).resolve()
    return config


def _records(path: Path, split: str, stage: str) -> list[dict[str, Any]]:
    output = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            if row.get("split") == split:
                row.pop("split", None)
                row.pop("source_trace_fingerprints", None)
                row.pop("reviewer", None)
                row.pop("id", None)
                if stage == "sft":
                    messages = row.pop("messages")
                    row = {"prompt": messages[:-1], "completion": [messages[-1]]}
                output.append(row)
    return output


def _filter_kwargs(callable_object, values: dict[str, Any]) -> dict[str, Any]:
    parameters = inspect.signature(callable_object).parameters
    if any(item.kind == inspect.Parameter.VAR_KEYWORD for item in parameters.values()):
        return values
    return {key: value for key, value in values.items() if key in parameters}


def _artifact_digests(root: Path) -> list[ArtifactDigest]:
    artifacts = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        if path.name == "run_manifest.json":
            continue
        artifacts.append(
            ArtifactDigest(
                path=path.relative_to(root).as_posix(),
                sha256=sha256_file(path),
                size_bytes=path.stat().st_size,
            )
        )
    return artifacts


def _write_model_card(
    path: Path,
    config: TrainingConfig,
    dataset_id: str,
    metrics: dict[str, Any],
    trainable_parameters: int,
    total_parameters: int,
) -> None:
    path.write_text(
        f"""# {config.run_name}

## Lineage

- Base model: `{config.base_model}`
- Stage: `{config.stage}`
- Dataset: `{dataset_id}`
- Seed: `{config.seed}`
- Quantized LoRA: `{config.qlora_4bit}`
- Trainable parameters: `{trainable_parameters:,}` / `{total_parameters:,}`

## Training metrics

```json
{json.dumps(metrics, indent=2, sort_keys=True)}
```

## Intended use

This adapter is an AgentForge experiment candidate. It must pass held-out quality,
safety, regression, latency, and cost gates before canary use.

## Limitations

Training success does not establish production quality. Review the source dataset,
protected-set leakage checks, evaluation report, and promotion decision. Do not use
the adapter for consequential decisions without domain-specific validation.
""",
        encoding="utf-8",
    )


def train(config: TrainingConfig) -> TrainingRunManifest:
    """Run SFT or DPO through TRL, saving only local artifacts and lineage evidence."""
    versions = require_training_dependencies()
    import torch
    from datasets import Dataset
    from peft import AutoPeftModelForCausalLM, LoraConfig, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, set_seed
    from trl import DPOConfig, DPOTrainer, SFTConfig, SFTTrainer

    manifest_path = config.dataset_manifest.resolve()
    dataset_manifest = load_and_verify_manifest(manifest_path)
    if dataset_manifest.split_counts.get("train", 0) < 1:
        raise ValueError("training dataset has no train split")
    output_dir = config.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"training output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    random.seed(config.seed)
    set_seed(config.seed)
    created_at = utc_now()
    config_payload = config.model_dump(mode="json")
    config_fingerprint = canonical_json_sha256(config_payload)

    tokenizer = AutoTokenizer.from_pretrained(
        config.base_model,
        trust_remote_code=config.trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model_kwargs: dict[str, Any] = {"trust_remote_code": config.trust_remote_code}
    if config.qlora_4bit:
        if not torch.cuda.is_available():
            raise RuntimeError("QLoRA training requires an available supported accelerator")
        compute_dtype = torch.bfloat16 if config.bf16 and torch.cuda.is_bf16_supported() else torch.float16
        model_kwargs.update(
            {
                "device_map": "auto",
                "quantization_config": BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_compute_dtype=compute_dtype,
                ),
            }
        )
    elif config.bf16:
        dtype_key = "dtype" if int(versions["transformers"].split(".", 1)[0]) >= 5 else "torch_dtype"
        model_kwargs[dtype_key] = torch.bfloat16
    is_existing_adapter = (Path(config.base_model) / "adapter_config.json").is_file()
    if is_existing_adapter:
        model = AutoPeftModelForCausalLM.from_pretrained(
            config.base_model, is_trainable=True, **model_kwargs
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(config.base_model, **model_kwargs)
    model.config.use_cache = False
    if config.qlora_4bit:
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=config.gradient_checkpointing
        )
    peft_config = LoraConfig(
        r=config.lora.rank,
        lora_alpha=config.lora.alpha,
        lora_dropout=config.lora.dropout,
        target_modules=config.lora.target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )
    source_relative = dataset_manifest.sft_path if config.stage == "sft" else dataset_manifest.dpo_path
    source_path = (manifest_path.parent / source_relative).resolve()
    train_rows = _records(source_path, "train", config.stage)
    validation_rows = _records(source_path, "validation", config.stage)
    train_dataset = Dataset.from_list(train_rows)
    eval_dataset = Dataset.from_list(validation_rows) if validation_rows else None
    common_args = {
        "output_dir": str(output_dir / "checkpoints"),
        "num_train_epochs": config.epochs,
        "learning_rate": config.learning_rate,
        "per_device_train_batch_size": config.per_device_batch_size,
        "per_device_eval_batch_size": config.per_device_batch_size,
        "gradient_accumulation_steps": config.gradient_accumulation_steps,
        "warmup_ratio": config.warmup_ratio,
        "logging_steps": config.logging_steps,
        "save_steps": config.save_steps,
        "gradient_checkpointing": config.gradient_checkpointing,
        "bf16": bool(config.bf16 and torch.cuda.is_available() and torch.cuda.is_bf16_supported()),
        "seed": config.seed,
        "report_to": "none",
        "eval_strategy": "steps" if eval_dataset is not None else "no",
        "save_total_limit": 2,
        "remove_unused_columns": False,
    }
    if config.stage == "sft":
        args_values = {
            **common_args,
            "max_length": config.max_length,
            "completion_only_loss": config.completion_only_loss,
        }
        training_args = SFTConfig(**_filter_kwargs(SFTConfig, args_values))
        trainer_values = {
            "model": model,
            "args": training_args,
            "train_dataset": train_dataset,
            "eval_dataset": eval_dataset,
            "peft_config": None if is_existing_adapter else peft_config,
            "processing_class": tokenizer,
            "tokenizer": tokenizer,
        }
        trainer = SFTTrainer(**_filter_kwargs(SFTTrainer, trainer_values))
    else:
        args_values = {**common_args, "max_length": config.max_length, "beta": config.dpo_beta}
        training_args = DPOConfig(**_filter_kwargs(DPOConfig, args_values))
        trainer_values = {
            "model": model,
            "args": training_args,
            "train_dataset": train_dataset,
            "eval_dataset": eval_dataset,
            "peft_config": None if is_existing_adapter else peft_config,
            "processing_class": tokenizer,
            "tokenizer": tokenizer,
        }
        trainer = DPOTrainer(**_filter_kwargs(DPOTrainer, trainer_values))
    train_output = trainer.train()
    adapter_dir = output_dir / "adapter"
    trainer.save_model(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))
    metrics = {
        key: value
        for key, value in dict(train_output.metrics).items()
        if isinstance(value, (int, float, str))
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    trainable_parameters = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    card_path = output_dir / "MODEL_CARD.md"
    _write_model_card(
        card_path,
        config,
        dataset_manifest.dataset_id,
        metrics,
        trainable_parameters,
        total_parameters,
    )
    artifacts = _artifact_digests(output_dir)
    unsigned = {
        "base_model": config.base_model,
        "dataset": dataset_manifest.manifest_fingerprint,
        "config": config_fingerprint,
        "artifacts": [artifact.model_dump() for artifact in artifacts],
        "metrics": metrics,
    }
    run_manifest = TrainingRunManifest(
        run_id=f"train-{uuid4()}",
        run_name=config.run_name,
        stage=config.stage,
        status="completed",
        created_at=created_at,
        completed_at=utc_now(),
        base_model=config.base_model,
        dataset_id=dataset_manifest.dataset_id,
        dataset_manifest_fingerprint=dataset_manifest.manifest_fingerprint,
        config_fingerprint=config_fingerprint,
        seed=config.seed,
        framework_versions=versions,
        metrics=metrics,
        trainable_parameters=trainable_parameters,
        total_parameters=total_parameters,
        trainable_percentage=(trainable_parameters / total_parameters * 100) if total_parameters else 0,
        artifacts=artifacts,
        output_dir=str(output_dir),
        model_card_path=card_path.name,
        run_fingerprint=canonical_json_sha256(unsigned),
    )
    (output_dir / "run_manifest.json").write_text(
        run_manifest.model_dump_json(indent=2) + "\n", encoding="utf-8"
    )
    return run_manifest


def verify_run_manifest(path: Path) -> TrainingRunManifest:
    manifest = TrainingRunManifest.model_validate_json(path.read_text(encoding="utf-8"))
    root = path.parent.resolve()
    for artifact in manifest.artifacts:
        candidate = (root / artifact.path).resolve()
        try:
            candidate.relative_to(root)
        except ValueError as exc:
            raise ValueError(f"training artifact escapes run directory: {artifact.path}") from exc
        if (
            not candidate.is_file()
            or candidate.stat().st_size != artifact.size_bytes
            or sha256_file(candidate) != artifact.sha256
        ):
            raise ValueError(f"training artifact integrity check failed: {artifact.path}")
    unsigned = {
        "base_model": manifest.base_model,
        "dataset": manifest.dataset_manifest_fingerprint,
        "config": manifest.config_fingerprint,
        "artifacts": [artifact.model_dump() for artifact in manifest.artifacts],
        "metrics": manifest.metrics,
    }
    if canonical_json_sha256(unsigned) != manifest.run_fingerprint:
        raise ValueError("training run fingerprint verification failed")
    return manifest
