"""Fresh Docker patch replay. Git applies bytes; repository code never runs on the host."""
from __future__ import annotations

import asyncio
import fnmatch
import hashlib
import json
import os
import re
import subprocess
import time
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from code_agent.models import CodeTask, QualityGateResult
from code_agent.sandbox import DockerSandbox
from code_agent.workspace import ALLOWED_FILENAMES, ALLOWED_SUFFIXES, _is_secret_or_excluded


def digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


class DeliveryReceipt(BaseModel):
    model_config = ConfigDict(extra="forbid")
    protocol: str = "fresh-docker-patch-v1"
    status: str
    reason: str
    patch_sha256: str
    source_sha256: str
    candidate_sha256: str
    command_sha256: str
    image: str
    image_id: str = ""
    application_exit_code: int | None = None
    test_exit_code: int | None = None
    tests_timed_out: bool = False
    candidate_matches: bool = False
    source_unchanged: bool = False
    protected_files_unchanged: bool = False
    execution_mode: str = "docker"
    duration_ms: float = Field(default=0, ge=0)
    fingerprint: str = ""


def safe_patch_paths(patch: str, limit: int) -> list[str]:
    """Accept conventional unified diffs only; reject metadata, links and unsafe paths."""
    paths = []
    before = None
    for line in patch.split("\n"):
        if line.startswith(("diff --git", "GIT binary patch", "rename ", "copy ", "new file mode", "deleted file mode")):
            raise ValueError("unsupported patch metadata")
        if line.startswith("--- "):
            before = line[4:].rstrip("\r")
        elif line.startswith("+++ "):
            after = line[4:].rstrip("\r")
            if before is None:
                raise ValueError("missing old-file header")
            old = before[2:] if before.startswith("a/") else None
            new = after[2:] if after.startswith("b/") else None
            if (old is None and before != "/dev/null") or (new is None and after != "/dev/null"):
                raise ValueError("invalid patch header")
            name = new or old
            if not name or (old and new and old != new) or name in paths:
                raise ValueError("ambiguous or duplicate patch path")
            path = Path(name)
            if (not re.fullmatch(r"[A-Za-z0-9_./-]+", name) or name.startswith("/") or ".." in path.parts
                    or _is_secret_or_excluded(path)
                    or (path.suffix.lower() not in ALLOWED_SUFFIXES and path.name not in ALLOWED_FILENAMES)):
                raise ValueError("unsafe patch path")
            paths.append(name)
            before = None
    if not paths or len(paths) > limit:
        raise ValueError("empty or oversized patch scope")
    return paths


def manifest(workspace) -> dict[str, str]:
    return {path: hashlib.sha256(workspace._read_bytes(path)).hexdigest()
            for path in workspace.list_files(limit=workspace.policy.max_files)
            if workspace.resolve(path).stat().st_size <= workspace.policy.max_file_bytes
            and (Path(path).suffix.lower() in ALLOWED_SUFFIXES or Path(path).name in ALLOWED_FILENAMES)}


def delivery_gate(receipt: DeliveryReceipt) -> QualityGateResult:
    return QualityGateResult(name="fresh_patch_delivery", status="passed" if valid_delivery(receipt) else "failed",
        summary=f"Fresh Docker patch replay: {receipt.status}/{receipt.reason}.",
        duration_ms=receipt.duration_ms, evidence=receipt.model_dump(mode="json"))


def valid_delivery(receipt: DeliveryReceipt, *, patch: str | None = None, command: list[str] | None = None) -> bool:
    return bool(receipt.protocol == "fresh-docker-patch-v1" and receipt.status == "passed"
        and receipt.reason == "fresh_candidate_tests_passed" and receipt.execution_mode == "docker"
        and receipt.application_exit_code == receipt.test_exit_code == 0 and not receipt.tests_timed_out
        and receipt.source_unchanged and receipt.candidate_matches and receipt.protected_files_unchanged
        and receipt.image_id.startswith("sha256:")
        and all(re.fullmatch(r"[a-f0-9]{64}", item) for item in (
            receipt.patch_sha256, receipt.source_sha256, receipt.candidate_sha256, receipt.command_sha256))
        and receipt.fingerprint == digest(receipt.model_dump(mode="json", exclude={"fingerprint"}))
        and (patch is None or receipt.patch_sha256 == hashlib.sha256(patch.encode("utf-8")).hexdigest())
        and (command is None or receipt.command_sha256 == digest(command)))


def result_delivery_valid(result, command: list[str] | None = None) -> bool:
    evidence = result.sandbox.get("delivery_receipt")
    if result.verification and result.verification.rounds:
        gate = next((row for row in result.verification.rounds[-1].quality_gates if row.name == "fresh_patch_delivery"), None)
        evidence = gate.evidence if gate and gate.status == "passed" else None
    try:
        return bool(evidence and valid_delivery(DeliveryReceipt.model_validate(evidence), patch=result.patch, command=command))
    except (TypeError, ValueError):
        return False


async def verify_delivery(task: CodeTask, candidate, patch: str, source_before: str) -> DeliveryReceipt:
    from code_agent.verification import repository_fingerprint

    started = time.perf_counter()
    expected = await asyncio.to_thread(manifest, candidate.workspace)
    receipt = DeliveryReceipt(status="failed", reason="not_executed",
        patch_sha256=hashlib.sha256(patch.encode("utf-8")).hexdigest(), source_sha256=source_before,
        candidate_sha256=digest(expected), command_sha256=digest(task.test_command), image=task.policy.image)
    fresh = None
    try:
        if repository_fingerprint(candidate.repository, task.policy.max_files, task.policy.max_repository_bytes) != source_before:
            raise ValueError("source drift")
        paths = safe_patch_paths(patch, task.policy.max_changed_files)
        if len(patch) > task.policy.max_patch_chars:
            raise ValueError("patch limit")
        # The controller alone may apply explicitly managed changes. General protected
        # paths remain protected; the expected manifest still binds every candidate byte.
        replay_policy = task.policy.model_copy(update={"controller_managed_paths": []})
        fresh = DockerSandbox(candidate.repository, replay_policy)
        await asyncio.to_thread(fresh.start)
        image = await asyncio.to_thread(subprocess.run,
            ["docker", "image", "inspect", task.policy.image, "--format", "{{.Id}}"],
            capture_output=True, text=True, timeout=10)
        if image.returncode != 0:
            raise ValueError("image identity unavailable")
        receipt.image_id = image.stdout.strip()
        baseline = manifest(fresh.workspace)
        for name in paths:
            fresh.workspace.check_agent_edit_path(name)
        applied = await asyncio.to_thread(subprocess.run,
            ["git", "-c", "core.autocrlf=false", "apply", "--no-index", "--whitespace=nowarn", "-"],
            cwd=fresh.workspace.root, input=patch.encode("utf-8"), capture_output=True, timeout=10,
            env={**os.environ, "GIT_CONFIG_GLOBAL": os.devnull, "GIT_CONFIG_NOSYSTEM": "1"})
        receipt.application_exit_code = applied.returncode
        receipt.reason = "patch_did_not_apply"
        if applied.returncode == 0:
            actual = manifest(fresh.workspace)
            receipt.candidate_matches = actual == expected
            protected = [name for name in baseline if any(
                fnmatch.fnmatchcase(name, rule) for rule in task.policy.protected_paths)]
            receipt.protected_files_unchanged = all(actual.get(name) == baseline[name] for name in protected)
            receipt.reason = "candidate_manifest_mismatch"
            if receipt.candidate_matches and receipt.protected_files_unchanged:
                before_tests = digest(actual)
                tests = await asyncio.to_thread(fresh.run, task.test_command)
                receipt.test_exit_code, receipt.tests_timed_out = tests.exit_code, tests.timed_out
                receipt.candidate_matches = before_tests == digest(manifest(fresh.workspace)) == digest(manifest(candidate.workspace))
                receipt.reason = "fresh_tests_failed_or_source_changed"
                if tests.exit_code == 0 and not tests.timed_out and receipt.candidate_matches:
                    receipt.status, receipt.reason = "passed", "fresh_candidate_tests_passed"
    except (OSError, RuntimeError, ValueError, subprocess.SubprocessError):
        receipt.status, receipt.reason = "failed", "replay_unavailable_or_policy_rejected"
    finally:
        if fresh is not None:
            try:
                await asyncio.to_thread(fresh.close)
            except (OSError, RuntimeError, subprocess.SubprocessError):
                receipt.status, receipt.reason = "failed", "cleanup_failed"
        try:
            receipt.source_unchanged = repository_fingerprint(candidate.repository, task.policy.max_files, task.policy.max_repository_bytes) == source_before
        except (OSError, ValueError):
            receipt.source_unchanged = False
        if not receipt.source_unchanged:
            receipt.status, receipt.reason = "failed", "source_integrity_error"
        receipt.duration_ms = (time.perf_counter() - started) * 1000
        receipt.fingerprint = digest(receipt.model_dump(mode="json", exclude={"fingerprint"}))
    return receipt
