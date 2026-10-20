"""Pinned public-source intake. Git data is read, never imported or installed."""

from __future__ import annotations

import hashlib
import re
import subprocess
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator

from code_agent.bounded_process import run_bounded
from code_agent.patch_inputs import PatchVerifySpec, test_source_path
from code_agent.regression_benchmark import confined
from code_agent.upgrade_models import StrictModel, safe_relative

DIGEST = r"^[a-f0-9]{64}$"
COMMIT = r"^[a-f0-9]{40}$"


class UpstreamFile(StrictModel):
    upstream_path: str
    local_path: str
    sha256: str = Field(pattern=DIGEST)
    candidate_sha256: str | None = Field(default=None, pattern=DIGEST)

    @model_validator(mode="after")
    def paths(self):
        safe_relative(self.upstream_path)
        safe_relative(self.local_path)
        return self


class UpstreamCase(StrictModel):
    case_id: str = Field(pattern=r"^[A-Za-z0-9_.-]{1,80}$")
    family: str = Field(pattern=r"^[A-Za-z0-9_.-]{1,100}$")
    repository_url: str = Field(pattern=r"^https://github\.com/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")
    base_commit: str = Field(pattern=COMMIT)
    candidate_commit: str = Field(pattern=COMMIT)
    expected: Literal["regression", "correct"]
    evidence_url: str = Field(pattern=r"^https://github\.com/[^\s]{5,500}$")
    files: list[UpstreamFile] = Field(min_length=3, max_length=30)
    support_files: dict[str, str] = Field(default_factory=dict, max_length=2)
    license_file: str
    license_identifier: Literal["MIT", "BSD-3-Clause"]
    spec: PatchVerifySpec
    expectations: dict[Literal["tests_only", "contracts", "generated"], Literal["alarm", "no_alarm"]]

    @model_validator(mode="after")
    def scope(self):
        paths = [row.local_path for row in self.files]
        originals = [row.upstream_path for row in self.files]
        if len(set(paths)) != len(paths) or len(set(originals)) != len(originals):
            raise ValueError("upstream/local file identities must be unique")
        changed = {row.local_path for row in self.files if row.candidate_sha256 is not None}
        if (
            changed != set(self.spec.source_paths)
            or not changed
            or not changed <= set(paths)
            or self.license_file not in paths
            or self.base_commit == self.candidate_commit
            or self.spec.task_id != self.case_id
            or not self.spec.preservation_authorized
            or not self.spec.contracts
            or not self.spec.json_domains
            or not {test_source_path(path) for path in self.spec.test_paths} <= set(paths)
            or set(self.expectations) != {"tests_only", "contracts", "generated"}
        ):
            raise ValueError(
                "pilot needs exact production scope, tests, contracts, JSON domains and expectations"
            )
        for path, text in self.support_files.items():
            safe_relative(path)
            if path in paths or not path.endswith("_version.py") or len(text.encode()) > 1000:
                raise ValueError("only bounded, non-shadowing version build metadata is supported")
        return self


class UpstreamCatalog(StrictModel):
    protocol: Literal["agentforge-upstream-pilot-v1"] = "agentforge-upstream-pilot-v1"
    catalog_id: str = Field(pattern=r"^[A-Za-z0-9_.-]{1,80}$")
    cases: list[UpstreamCase] = Field(min_length=1, max_length=12)

    @model_validator(mode="after")
    def unique_cases(self):
        if len({row.case_id for row in self.cases}) != len(self.cases):
            raise ValueError("upstream case identities must be unique")
        if {row.expected for row in self.cases} != {"regression", "correct"}:
            raise ValueError("pilot requires positive and negative controls")
        return self


class GitSource:
    """Read immutable Git objects with bounded output and Git object-ID checks."""

    def __init__(self, cache: Path, url: str, *, allow_download=False):
        if not re.fullmatch(r"https://github\.com/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", url):
            raise ValueError("only explicit public GitHub HTTPS repositories are supported")
        self.path = confined(cache, hashlib.sha256(url.encode()).hexdigest()[:24] + ".git")
        if not self.path.exists():
            if allow_download is not True:
                raise ValueError("upstream cache absent; downloads require explicit --allow-download")
            self.path.parent.mkdir(parents=True, exist_ok=True)
            # No checkout, submodules, package installation or repository hooks.
            subprocess.run(
                ["git", "-c", "init.templateDir=", "clone", "--bare", url, str(self.path)],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=120,
                check=True,
            )
        if self.read("config", "--get", "remote.origin.url", limit=1000).decode().strip() != url:
            raise ValueError("upstream cache origin does not match the pinned repository")

    def read(self, *args, limit=100_000):
        result = run_bounded(
            ["git", "--no-replace-objects", f"--git-dir={self.path}", *args], timeout=15, max_chars=limit
        )
        raw = result.stdout.encode("utf-8")
        if result.timed_out or result.returncode != 0:
            raise ValueError("pinned Git object is absent or unreadable")
        if result.stdout_bytes > limit or len(raw) != result.stdout_bytes:
            raise ValueError("Git data exceeds its UTF-8 byte profile")
        return raw

    def object(self, identity: str, kind: str, limit: int):
        if not re.fullmatch(COMMIT, identity) or kind not in {"blob", "commit"}:
            raise ValueError("only pinned Git object identities are supported")
        if self.read("cat-file", "-t", identity).decode().strip() != kind:
            raise ValueError("Git object kind differs from the declared input")
        size = int(self.read("cat-file", "-s", identity, limit=30))
        if not 0 <= size <= limit:
            raise ValueError("Git object exceeds its byte budget")
        raw = self.read("cat-file", kind, identity, limit=limit)
        header = f"{kind} {len(raw)}\0".encode()
        if len(raw) != size or hashlib.sha1(header + raw, usedforsecurity=False).hexdigest() != identity:
            raise ValueError("Git object content does not match its identity")
        return raw

    def commit(self, identity: str):
        return self.object(identity, "commit", 100_000)

    def blob(self, commit: str, path: str, expected: str):
        safe_relative(path)
        if not re.fullmatch(COMMIT, commit):
            raise ValueError("a full commit ID is required")
        # ls-tree also checks mode; a symlink's blob must never become executable source.
        entry = self.read("ls-tree", "-z", commit, "--", path, limit=2000)
        records = entry.rstrip(b"\0").split(b"\0")
        if len(records) != 1 or b"\t" not in records[0]:
            raise ValueError("pinned source path is missing or ambiguous")
        metadata, name = records[0].split(b"\t", 1)
        mode, kind, oid = metadata.split()
        if mode not in {b"100644", b"100755"} or kind != b"blob" or name.decode() != path:
            raise ValueError("only exact regular-file Git blobs are supported")
        raw = self.object(oid.decode(), "blob", 1_000_000)
        if hashlib.sha256(raw).hexdigest() != expected or b"\0" in raw:
            raise ValueError("upstream bytes changed or contain binary data")
        raw.decode("utf-8")
        return raw
