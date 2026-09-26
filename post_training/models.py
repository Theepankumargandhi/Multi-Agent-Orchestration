"""Portable contracts for post-training datasets, runs, and model lineage."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


class ChatTurn(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: str = Field(min_length=1, max_length=100_000)


class SFTRecord(BaseModel):
    id: str
    messages: list[ChatTurn] = Field(min_length=2)
    split: Literal["train", "validation", "test"]
    source_trace_fingerprints: list[str] = Field(min_length=1)
    reviewer: str = Field(min_length=1)

    @model_validator(mode="after")
    def valid_conversation(self) -> "SFTRecord":
        if self.messages[-1].role != "assistant":
            raise ValueError("SFT conversation must end with an assistant response")
        if not any(turn.role == "user" for turn in self.messages):
            raise ValueError("SFT conversation requires a user turn")
        return self


class DPORecord(BaseModel):
    id: str
    prompt: str = Field(min_length=1, max_length=100_000)
    chosen: str = Field(min_length=1, max_length=100_000)
    rejected: str = Field(min_length=1, max_length=100_000)
    split: Literal["train", "validation", "test"]
    source_trace_fingerprints: list[str] = Field(min_length=1)
    reviewer: str = Field(min_length=1)

    @model_validator(mode="after")
    def distinct_responses(self) -> "DPORecord":
        if self.chosen.strip() == self.rejected.strip():
            raise ValueError("chosen and rejected responses must differ")
        return self


class LeakageFinding(BaseModel):
    training_id: str
    protected_id: str
    similarity: float = Field(ge=0, le=1)
    kind: Literal["exact", "near_duplicate"]


class DatasetManifest(BaseModel):
    schema_version: str = "1.0"
    dataset_id: str
    created_at: str = Field(default_factory=utc_now)
    source_path: str
    source_sha256: str
    sft_path: str
    sft_sha256: str
    dpo_path: str
    dpo_sha256: str
    split_counts: dict[str, int]
    unique_prompts: int = Field(ge=0)
    duplicate_records_removed: int = Field(ge=0)
    protected_dataset_sha256: dict[str, str] = Field(default_factory=dict)
    leakage_findings: list[LeakageFinding] = Field(default_factory=list)
    leakage_threshold: float = Field(default=0.85, ge=0, le=1)
    human_review_required: bool = True
    all_records_human_reviewed: bool
    manifest_fingerprint: str


class LoRAConfig(BaseModel):
    rank: int = Field(default=16, ge=1, le=256)
    alpha: int = Field(default=32, ge=1, le=1024)
    dropout: float = Field(default=0.05, ge=0, lt=1)
    target_modules: list[str] = Field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"]
    )


class TrainingConfig(BaseModel):
    schema_version: str = "1.0"
    run_name: str = Field(min_length=1, max_length=100)
    stage: Literal["sft", "dpo"]
    base_model: str = Field(min_length=1, max_length=500)
    dataset_manifest: Path
    output_dir: Path
    seed: int = 42
    epochs: float = Field(default=1.0, gt=0, le=100)
    learning_rate: float = Field(default=0.0001, gt=0, le=1)
    per_device_batch_size: int = Field(default=1, ge=1, le=128)
    gradient_accumulation_steps: int = Field(default=8, ge=1, le=1024)
    max_length: int = Field(default=2048, ge=64, le=131_072)
    warmup_ratio: float = Field(default=0.03, ge=0, le=1)
    logging_steps: int = Field(default=5, ge=1)
    save_steps: int = Field(default=100, ge=1)
    gradient_checkpointing: bool = True
    bf16: bool = True
    qlora_4bit: bool = True
    completion_only_loss: bool = True
    dpo_beta: float = Field(default=0.1, gt=0, le=10)
    lora: LoRAConfig = Field(default_factory=LoRAConfig)
    trust_remote_code: bool = False

    @field_validator("output_dir")
    @classmethod
    def output_must_not_be_workspace_root(cls, value: Path) -> Path:
        if str(value).strip() in {"", ".", "/", "\\"}:
            raise ValueError("output_dir must be a dedicated run directory")
        return value


class ArtifactDigest(BaseModel):
    path: str
    sha256: str
    size_bytes: int = Field(ge=0)


class TrainingRunManifest(BaseModel):
    schema_version: str = "1.0"
    run_id: str
    run_name: str
    stage: Literal["sft", "dpo", "lora_smoke"]
    status: Literal["completed", "failed"]
    created_at: str
    completed_at: str
    base_model: str
    dataset_id: str
    dataset_manifest_fingerprint: str
    config_fingerprint: str
    seed: int
    framework_versions: dict[str, str]
    metrics: dict[str, float | int | str]
    trainable_parameters: int = Field(ge=0)
    total_parameters: int = Field(ge=0)
    trainable_percentage: float = Field(ge=0, le=100)
    artifacts: list[ArtifactDigest]
    output_dir: str
    model_card_path: str
    run_fingerprint: str


class ModelRecord(BaseModel):
    version: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,99}$")
    registered_at: str = Field(default_factory=utc_now)
    run_id: str
    stage: str
    base_model: str
    dataset_id: str
    run_fingerprint: str
    manifest_path: str
    artifact_sha256: str
    status: Literal["candidate", "canary", "production", "retired"] = "candidate"
    promotion_evidence_fingerprint: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ModelRegistryState(BaseModel):
    schema_version: str = "1.0"
    models: list[ModelRecord] = Field(default_factory=list)
    production_version: str | None = None
    canary_version: str | None = None
    audit_log: list[dict[str, Any]] = Field(default_factory=list)
