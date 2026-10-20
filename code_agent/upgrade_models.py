"""Strict owner contracts for execution-verified dependency migration."""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import PurePosixPath
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

PROTOCOL = "upgradepilot-pydantic-v1"
SOURCE_VERSION = "1.10.24"
TARGET_VERSION = "2.12.4"
BASELINE_VERSION = "0.8.0"
V1_PYTHON = "/opt/v1/bin/python"
V2_PYTHON = "/opt/v2/bin/python"
PROBE_PATH = "/opt/upgradepilot/probe.py"
CONTRACT_PATH = "upgradepilot-contracts.json"
DEFAULT_IMAGE = "agentforge-upgradepilot:local"


def fingerprint(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def safe_relative(value: str) -> str:
    if "\\" in value or not re.fullmatch(r"[A-Za-z0-9_.\-/]+", value) or value.startswith(("/", "-")):
        raise ValueError("a portable repository-relative path is required")
    if any(part in {"", ".", ".."} for part in value.split("/")) or PurePosixPath(value).is_absolute():
        raise ValueError("path traversal is not allowed")
    from code_agent.workspace import _is_secret_or_excluded
    if _is_secret_or_excluded(PurePosixPath(value)):
        raise ValueError("secret and runtime-data paths are not migration inputs")
    return value


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)


class ContractCase(StrictModel):
    case_id: str = Field(pattern=r"^[A-Za-z0-9_.-]{1,80}$")
    entrypoint: str = Field(pattern=r"^[A-Za-z][A-Za-z0-9_]*(?:\.[A-Za-z][A-Za-z0-9_]*)*:[A-Za-z][A-Za-z0-9_]*$", max_length=200)
    args: list[Any] = Field(default_factory=list, max_length=20)
    kwargs: dict[str, Any] = Field(default_factory=dict, max_length=20)
    expected_kind: Literal["value", "error"] = "value"
    expected_error: str | None = Field(default=None, pattern=r"^[A-Za-z][A-Za-z0-9_]{0,100}$")

    @model_validator(mode="after")
    def bounded_json(self):
        if (self.expected_kind == "error") != bool(self.expected_error):
            raise ValueError("error contracts require the exact expected exception class")
        if len(json.dumps([self.args, self.kwargs], allow_nan=False).encode()) > 8000:
            raise ValueError("contract input exceeds the byte budget")
        return self


class AcceptedDifference(StrictModel):
    case_id: str = Field(pattern=r"^[A-Za-z0-9_.-]{1,80}$")
    source_result_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    target_result_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    rationale: str = Field(min_length=10, max_length=1000)
    rule_id: Literal["validators", "field-options", "optional-default", "union-coercion", "config", "model-methods", "v1-namespace"]


class MigrationSpec(StrictModel):
    protocol: Literal["upgradepilot-pydantic-v1"] = PROTOCOL
    task_id: str = Field(pattern=r"^[A-Za-z0-9_.-]{1,80}$")
    family: str = Field(min_length=1, max_length=100)
    source_paths: list[str] = Field(min_length=1, max_length=20)
    dependency_file: str = "requirements.txt"
    test_command: list[str] = Field(default_factory=lambda: ["-m", "pytest", "-q"], min_length=1, max_length=25)
    contracts: list[ContractCase] = Field(min_length=1, max_length=50)
    accepted_differences: list[AcceptedDifference] = Field(default_factory=list, max_length=50)
    # Model/provider requests are never enabled by loading a spec.
    image: str = DEFAULT_IMAGE
    command_timeout_seconds: int = Field(default=60, ge=5, le=180)
    task_timeout_seconds: int = Field(default=600, ge=60, le=1800)
    max_model_calls: int = Field(default=16, ge=4, le=30)
    max_writes: int = Field(default=8, ge=1, le=20)
    max_reserved_tokens: int = Field(default=200000, ge=10000, le=500000)

    @field_validator("dependency_file")
    @classmethod
    def dependency_path(cls, value):
        return safe_relative(value)

    @field_validator("image")
    @classmethod
    def image_reference(cls, value):
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._/:@-]{0,200}", value):
            raise ValueError("invalid local image reference")
        return value

    @field_validator("source_paths")
    @classmethod
    def confined_sources(cls, values):
        clean = [safe_relative(value) for value in values]
        if len(set(clean)) != len(clean):
            raise ValueError("source paths must be unique")
        return clean

    @field_validator("test_command")
    @classmethod
    def owner_test_command(cls, values):
        # Arguments only: fixed interpreter, no pip/install, host execution or model-supplied commands.
        if values[:2] != ["-m", "pytest"] or any(not item or len(item) > 200 or "\x00" in item for item in values):
            raise ValueError("only fixed-interpreter pytest arguments are supported")
        if set(values) & {"--collect-only", "--co", "--help", "-h", "--version", "--setup-only"}:
            raise ValueError("test discovery/help cannot count as test execution")
        return values

    @model_validator(mode="after")
    def unique_cases(self):
        ids = [row.case_id for row in self.contracts]
        approved = [row.case_id for row in self.accepted_differences]
        if len(set(ids)) != len(ids) or len(set(approved)) != len(approved) or not set(approved) <= set(ids):
            raise ValueError("contract and approved-difference IDs must be unique and bound")
        if len(self.model_dump_json().encode()) > 40000:
            raise ValueError("migration specification exceeds its byte budget")
        return self


class ImpactFinding(StrictModel):
    path: str
    line: int = Field(ge=1)
    symbol: str
    rule_id: str
    severity: Literal["mechanical", "semantic", "unsupported"]
    source_url: str


class MigrationPlan(StrictModel):
    protocol: str = PROTOCOL
    spec_fingerprint: str
    source_fingerprint: str
    catalog_fingerprint: str
    findings: list[ImpactFinding]
    affected_files: list[str]
    status: Literal["planned", "held"]
    hold_reasons: list[str] = Field(default_factory=list)
