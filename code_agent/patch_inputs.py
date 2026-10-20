"""Data-only admission for external patches and owner-scoped Python behavior checks."""
from __future__ import annotations

import hashlib
import json
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator, model_validator

from code_agent.patch_json import ContractApprovalBinding, JsonDomain, ReductionPolicy, generate_json_cases
from code_agent.upgrade_models import ContractCase, StrictModel, fingerprint, safe_relative
from code_agent.verification import is_test_path

PROTOCOL = "agentforge-patch-verify-v1"
GENERATOR = "owner-integer-domain-v1"
MAX_PATCH_BYTES = 200_000


def test_source_path(selector: str) -> str:
    """A fixed pytest file/directory, optionally narrowed to named classes/tests."""
    parts = selector.split("::")
    path = safe_relative(parts[0])
    if (len(selector) > 400 or len(parts) > 4
            or (len(parts) > 1 and not path.endswith(".py"))
            or any(not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]{0,99}", item) for item in parts[1:])):
        raise ValueError("only bounded named pytest selectors are supported")
    return path


def strict_json(raw: str | bytes):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("duplicate JSON key")
            result[key] = value
        return result
    return json.loads(raw, object_pairs_hook=pairs,
                      parse_constant=lambda _: (_ for _ in ()).throw(ValueError("nonfinite JSON")))


def read_bounded(path: Path, limit: int) -> bytes:
    if any(item.is_symlink() or (item.exists() and getattr(item.lstat(), "st_file_attributes", 0) & 0x400)
           for item in (path, *path.parents)):
        raise ValueError("input links and reparse points are not supported")
    if not path.is_file() or not 0 < path.stat().st_size <= limit:
        raise ValueError("input is absent or exceeds its byte limit")
    with path.open("rb") as stream:
        raw = stream.read(limit + 1)
    if len(raw) > limit:
        raise ValueError("input changed beyond its byte limit")
    return raw


class IntegerDomain(StrictModel):
    """The owner explicitly declares preservation for this integer input domain."""
    domain_id: str = Field(pattern=r"^[A-Za-z0-9_.-]{1,60}$")
    entrypoint: str = Field(pattern=r"^[A-Za-z][A-Za-z0-9_]*(?:\.[A-Za-z][A-Za-z0-9_]*)*:[A-Za-z][A-Za-z0-9_]*$", max_length=200)
    minimum: int = Field(ge=-1_000_000, le=1_000_000)
    maximum: int = Field(ge=-1_000_000, le=1_000_000)
    samples: int = Field(default=8, ge=2, le=12)

    @model_validator(mode="after")
    def ordered(self):
        if self.minimum >= self.maximum:
            raise ValueError("integer preservation domain must have nonzero ordered width")
        return self


class PatchVerifySpec(StrictModel):
    protocol: Literal["agentforge-patch-verify-v1"] = PROTOCOL
    task_id: str = Field(pattern=r"^[A-Za-z0-9_.-]{1,80}$")
    source_paths: list[str] = Field(min_length=1, max_length=20)
    test_paths: list[str] = Field(min_length=1, max_length=20)
    contracts: list[ContractCase] = Field(default_factory=list, max_length=40)
    integer_domains: list[IntegerDomain] = Field(default_factory=list, max_length=4)
    json_domains: list[JsonDomain] = Field(default_factory=list, max_length=4)
    reduction: ReductionPolicy = Field(default_factory=ReductionPolicy)
    contract_review: ContractApprovalBinding | None = None
    reference_source_sha256: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")
    reference_patch_sha256: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")
    # This is an owner statement of intended preservation, not inferred semantics.
    preservation_authorized: bool = False
    image: str = Field(default="agentforge-code-sandbox:local", pattern=r"^[A-Za-z0-9][A-Za-z0-9._/:@-]{0,200}$")
    seed: int = Field(default=1729, ge=0, le=2**31 - 1)
    command_timeout_seconds: int = Field(default=45, ge=5, le=180)

    @field_validator("source_paths")
    @classmethod
    def paths(cls, values):
        clean = [safe_relative(value) for value in values]
        if len(set(clean)) != len(clean):
            raise ValueError("paths must be unique")
        return clean

    @field_validator("test_paths")
    @classmethod
    def test_selectors(cls, values):
        for value in values:
            test_source_path(value)
        if len(set(values)) != len(values):
            raise ValueError("test selectors must be unique")
        return values

    @model_validator(mode="after")
    def scope(self):
        ids = [row.case_id for row in self.contracts]
        domains = [row.domain_id for row in [*self.integer_domains, *self.json_domains]]
        if len(set(ids)) != len(ids) or len(set(domains)) != len(domains) or any(name.startswith("generated.") for name in ids):
            raise ValueError("duplicate or reserved check identities")
        test_sources = {test_source_path(name) for name in self.test_paths}
        if any(is_test_path(name) or name.endswith("conftest.py") or name in test_sources for name in self.source_paths):
            raise ValueError("test sources cannot be declared as editable production scope")
        if sum(row.samples for row in [*self.integer_domains, *self.json_domains]) > 64:
            raise ValueError("combined generated-case budget exceeded")
        if len(self.model_dump_json().encode()) > 40_000:
            raise ValueError("verification specification exceeds its byte limit")
        return self


def generated_cases(spec: PatchVerifySpec) -> list[ContractCase]:
    cases = []
    for domain in spec.integer_domains:
        rng = random.Random(f"{GENERATOR}:{spec.seed}:{domain.domain_id}")
        # Boundary-adjacent and sign-transition values precede seeded interior samples.
        values = []
        for value in (domain.minimum, domain.maximum, domain.minimum + 1, domain.maximum - 1, 0, -1, 1):
            if domain.minimum <= value <= domain.maximum and value not in values:
                values.append(value)
        remaining = domain.samples - len(values)
        if remaining > 0:
            population = range(domain.minimum, domain.maximum + 1)
            for value in rng.sample(population, min(len(population), domain.samples * 2)):
                if value not in values:
                    values.append(value)
                if len(values) >= domain.samples:
                    break
        for index, value in enumerate(values[:domain.samples]):
            cases.append(ContractCase(case_id=f"generated.{domain.domain_id}.{index}", entrypoint=domain.entrypoint, args=[value]))
    for domain in spec.json_domains:
        cases.extend(generate_json_cases(domain, spec.seed))
    return cases


@dataclass(frozen=True)
class FilePatch:
    path: str
    hunks: tuple[tuple[int, int, int, int, tuple[str, ...]], ...]


def parse_patch(patch: str, spec: PatchVerifySpec) -> list[FilePatch]:
    """Strict unified hunks: existing Python files only, no fuzzy application or Git execution."""
    if not patch or len(patch.encode("utf-8")) > MAX_PATCH_BYTES or "\x00" in patch or "\r" in patch:
        raise ValueError("patch must be bounded LF UTF-8 text")
    lines = [line + "\n" for line in patch.split("\n")[:-1]] + ([patch.rsplit("\n", 1)[-1]] if not patch.endswith("\n") else [])
    result, index = [], 0
    while index < len(lines):
        if lines[index].startswith("diff --git "):
            match = re.fullmatch(r"diff --git a/([A-Za-z0-9_./-]+) b/\1\n", lines[index])
            if not match:
                raise ValueError("unsupported Git diff metadata")
            git_name = match[1]
            index += 1
            if index < len(lines) and lines[index].startswith("index "):
                if not re.fullmatch(r"index [a-f0-9]{7,64}\.\.[a-f0-9]{7,64}(?: 100644)?\n", lines[index]):
                    raise ValueError("unsupported Git index metadata")
                index += 1
        else:
            git_name = None
        if index + 1 >= len(lines) or not lines[index].startswith("--- a/") or not lines[index + 1].startswith("+++ b/"):
            raise ValueError("only edits to existing Python source files are supported")
        old, new = lines[index][6:].rstrip("\n"), lines[index + 1][6:].rstrip("\n")
        name = safe_relative(old)
        if old != new or (git_name is not None and git_name != name) or name in [row.path for row in result]:
            raise ValueError("ambiguous or repeated file identity")
        if (not name.endswith(".py") or is_test_path(name) or Path(name).name in {"conftest.py", "setup.py"}
                or any(name == scope or name.startswith(scope.rstrip("/") + "/")
                       for scope in map(test_source_path, spec.test_paths))
                or not any(name == scope or name.startswith(scope.rstrip("/") + "/") for scope in spec.source_paths)):
            raise ValueError("patch changes protected files or leaves owner production scope")
        index += 2
        hunks = []
        while index < len(lines) and lines[index].startswith("@@ "):
            match = re.fullmatch(r"@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@[^\n]*\n", lines[index])
            if not match:
                raise ValueError("invalid hunk header")
            start, count, target, new_count = int(match[1]), int(match[2] or 1), int(match[3]), int(match[4] or 1)
            index += 1
            body, removed, added = [], 0, 0
            while removed < count or added < new_count:
                if index >= len(lines) or lines[index][:1] not in {" ", "+", "-"}:
                    raise ValueError("hunk counts do not match its body")
                line = lines[index]
                removed += line[0] in {" ", "-"}
                added += line[0] in {" ", "+"}
                body.append(line)
                index += 1
                if index < len(lines) and lines[index].rstrip("\n") == "\\ No newline at end of file":
                    body[-1] = body[-1].removesuffix("\n")
                    index += 1
            if removed != count or added != new_count or not any(line[0] in {"+", "-"} for line in body):
                raise ValueError("invalid or empty hunk")
            hunks.append((start, count, target, new_count, tuple(body)))
            if len(hunks) > 100:
                raise ValueError("hunk budget exceeded")
        if not hunks:
            raise ValueError("file patch has no hunks")
        result.append(FilePatch(name, tuple(hunks)))
        if len(result) > 20:
            raise ValueError("changed-file budget exceeded")
    return result


def apply_file_patch(source: str, edit: FilePatch) -> str:
    # Git hunks split only on LF; Unicode separators inside source strings are data.
    lines = [line + "\n" for line in source.split("\n")[:-1]] + ([source.rsplit("\n", 1)[-1]] if not source.endswith("\n") else [])
    output, cursor = [], 0
    for start, count, target, new_count, body in edit.hunks:
        offset = start - 1 if count else start
        if not cursor <= offset <= len(lines):
            raise ValueError("hunks overlap or leave source bounds")
        output.extend(lines[cursor:offset])
        target_offset = target - 1 if new_count else target
        if len(output) != target_offset:
            raise ValueError("target hunk position differs from exact application")
        cursor = offset
        for line in body:
            if line[0] in {" ", "-"}:
                if cursor >= len(lines) or lines[cursor] != line[1:]:
                    raise ValueError("patch context does not match the original source")
                cursor += 1
            if line[0] in {" ", "+"}:
                output.append(line[1:])
    output.extend(lines[cursor:])
    candidate = "".join(output)
    if candidate == source or len(candidate.encode()) > 1_000_000:
        raise ValueError("no-op or oversized candidate source")
    return candidate


def import_prediction(raw: bytes, instance_id: str) -> tuple[str, dict]:
    """mini-SWE-agent batch preds.json / SWE-bench records, never a raw trajectory."""
    if len(raw) > 2_000_000 or not re.fullmatch(r"[A-Za-z0-9_.-]{1,160}", instance_id):
        raise ValueError("prediction input exceeds admission bounds")
    payload = strict_json(raw)
    if isinstance(payload, dict) and "model_patch" not in payload:
        if instance_id not in payload:
            raise ValueError("prediction instance is absent")
        rows = [payload[instance_id]]
    else:
        rows = payload if isinstance(payload, list) else [payload]
    matches = [row for row in rows if isinstance(row, dict) and row.get("instance_id") == instance_id]
    if len(matches) != 1 or not isinstance(matches[0].get("model_patch"), str):
        raise ValueError("prediction identity is ambiguous or patch is absent")
    patch = matches[0]["model_patch"]
    if not 0 < len(patch.encode()) <= MAX_PATCH_BYTES:
        raise ValueError("prediction patch is empty or oversized")
    return patch, {"adapter": "mini-swe-agent-swebench-preds-v1", "instance_id": instance_id,
                   "input_sha256": hashlib.sha256(raw).hexdigest(), "producer_authenticated": False,
                   "producer_label_sha256": fingerprint(matches[0].get("model_name_or_path"))}
