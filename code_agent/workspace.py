"""Filtered, temporary workspace with path confinement and diff generation."""

from __future__ import annotations

import difflib
import os
import re
import shutil
import stat
import tempfile
from pathlib import Path

from code_agent.models import SandboxPolicy

EXCLUDED_PARTS = {
    ".git",
    ".venv",
    "venv",
    "env",
    "__pycache__",
    ".pytest_cache",
    ".ruff_cache",
    "node_modules",
    "chroma_db",
    "graph_chroma_db",
    "data",
    ".runtime_logs",
}
SECRET_SUFFIXES = {".pem", ".key", ".p12", ".pfx", ".crt"}
ALLOWED_SUFFIXES = {
    ".py",
    ".pyi",
    ".toml",
    ".yaml",
    ".yml",
    ".json",
    ".jsonl",
    ".md",
    ".txt",
    ".ini",
    ".cfg",
    ".sql",
    ".html",
    ".css",
    ".js",
    ".jsx",
    ".ts",
    ".tsx",
    ".sh",
    ".ps1",
}
ALLOWED_FILENAMES = {"Dockerfile", "Makefile", "LICENSE"}


class WorkspacePolicyError(ValueError):
    pass


def _is_secret_or_excluded(relative: Path) -> bool:
    if any(part in EXCLUDED_PARTS for part in relative.parts):
        return True
    name = relative.name.lower()
    return name == ".env" or name.startswith(".env.") or relative.suffix.lower() in SECRET_SUFFIXES


def _is_binary(data: bytes) -> bool:
    return b"\x00" in data[:4096]


class EphemeralWorkspace:
    def __init__(self, source: Path, policy: SandboxPolicy):
        self.source = source.resolve()
        self.policy = policy
        self.root: Path | None = None
        self._original: dict[str, str] = {}

    def prepare(self) -> Path:
        if not self.source.is_dir():
            raise WorkspacePolicyError(f"repository does not exist: {self.source}")
        root = Path(tempfile.mkdtemp(prefix="agentforge-code-"))
        self.root = root.resolve()
        try:
            self.root.chmod(0o777)
        except OSError:
            pass
        file_count = 0
        total_bytes = 0
        try:
            for source_path in self.source.rglob("*"):
                relative = source_path.relative_to(self.source)
                if _is_secret_or_excluded(relative) or source_path.is_symlink():
                    continue
                destination = self.root / relative
                if source_path.is_dir():
                    destination.mkdir(parents=True, exist_ok=True)
                    try:
                        destination.chmod(0o777)
                    except OSError:
                        pass
                    continue
                if not source_path.is_file():
                    continue
                size = source_path.stat().st_size
                file_count += 1
                total_bytes += size
                if file_count > self.policy.max_files or total_bytes > self.policy.max_repository_bytes:
                    raise WorkspacePolicyError("repository exceeds sandbox copy limits")
                destination.parent.mkdir(parents=True, exist_ok=True)
                try:
                    destination.parent.chmod(0o777)
                except OSError:
                    pass
                shutil.copy2(source_path, destination)
                try:
                    destination.chmod(
                        destination.stat().st_mode | stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH
                    )
                except OSError:
                    pass
            self._original = self._snapshot()
            return self.root
        except Exception:
            self.cleanup()
            raise

    def _require_root(self) -> Path:
        if self.root is None:
            raise RuntimeError("workspace has not been prepared")
        return self.root

    def resolve(self, relative_path: str, *, allow_missing: bool = False) -> Path:
        root = self._require_root().resolve()
        clean = (relative_path or "").strip().replace("\\", "/")
        if not clean or clean.startswith("/") or "\x00" in clean:
            raise WorkspacePolicyError("a safe relative file path is required")
        candidate = (root / clean).resolve(strict=False)
        try:
            candidate.relative_to(root)
        except ValueError as exc:
            raise WorkspacePolicyError("path escapes the sandbox workspace") from exc
        if not allow_missing and not candidate.exists():
            raise WorkspacePolicyError(f"file does not exist: {clean}")
        return candidate

    def list_files(self, limit: int = 500) -> list[str]:
        root = self._require_root()
        files = [
            path.relative_to(root).as_posix()
            for path in root.rglob("*")
            if path.is_file()
            and not path.is_symlink()
            and not _is_secret_or_excluded(path.relative_to(root))
        ]
        return sorted(files)[: max(1, min(limit, 50000))]

    def read_file(self, relative_path: str) -> str:
        path = self.resolve(relative_path)
        if not path.is_file() or path.stat().st_size > self.policy.max_file_bytes:
            raise WorkspacePolicyError("file is not readable within configured limits")
        data = path.read_bytes()
        if _is_binary(data):
            raise WorkspacePolicyError("binary files are not exposed to the agent")
        return data.decode("utf-8", errors="replace")

    def search(self, pattern: str, limit: int = 100) -> list[str]:
        if not pattern or len(pattern) > 300:
            raise WorkspacePolicyError("search pattern must contain 1-300 characters")
        try:
            expression = re.compile(pattern, re.IGNORECASE)
        except re.error as exc:
            raise WorkspacePolicyError(f"invalid search expression: {exc}") from exc
        matches: list[str] = []
        for relative in self.list_files(limit=self.policy.max_files):
            try:
                text = self.read_file(relative)
            except WorkspacePolicyError:
                continue
            for number, line in enumerate(text.splitlines(), start=1):
                if expression.search(line):
                    matches.append(f"{relative}:{number}: {line[:300]}")
                    if len(matches) >= limit:
                        return matches
        return matches

    def write_file(self, relative_path: str, content: str) -> None:
        path = self.resolve(relative_path, allow_missing=True)
        relative = path.relative_to(self._require_root())
        if _is_secret_or_excluded(relative):
            raise WorkspacePolicyError("writing secret or excluded files is forbidden")
        if path.suffix.lower() not in ALLOWED_SUFFIXES and path.name not in ALLOWED_FILENAMES:
            raise WorkspacePolicyError("file type is not allowlisted for agent edits")
        encoded = content.encode("utf-8")
        if len(encoded) > self.policy.max_file_bytes:
            raise WorkspacePolicyError("file content exceeds configured limit")
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.agentforge-tmp")
        temporary.write_bytes(encoded)
        os.replace(temporary, path)

    def delete_file(self, relative_path: str) -> None:
        path = self.resolve(relative_path)
        if not path.is_file() or path.is_symlink():
            raise WorkspacePolicyError("only regular files can be deleted")
        path.unlink()

    def _snapshot(self) -> dict[str, str]:
        snapshot: dict[str, str] = {}
        root = self._require_root()
        for relative in self.list_files(limit=self.policy.max_files):
            path = root / relative
            if path.stat().st_size > self.policy.max_file_bytes:
                continue
            data = path.read_bytes()
            if _is_binary(data):
                continue
            snapshot[relative] = data.decode("utf-8", errors="replace")
        return snapshot

    def changed_files(self) -> list[str]:
        current = self._snapshot()
        names = set(self._original) | set(current)
        return sorted(name for name in names if self._original.get(name) != current.get(name))

    def unified_diff(self) -> str:
        current = self._snapshot()
        chunks: list[str] = []
        for name in sorted(set(self._original) | set(current)):
            before = self._original.get(name)
            after = current.get(name)
            if before == after:
                continue
            chunks.extend(
                difflib.unified_diff(
                    [] if before is None else before.splitlines(keepends=True),
                    [] if after is None else after.splitlines(keepends=True),
                    fromfile="/dev/null" if before is None else f"a/{name}",
                    tofile="/dev/null" if after is None else f"b/{name}",
                )
            )
        patch = "".join(chunks)
        if len(patch) > self.policy.max_patch_chars:
            raise WorkspacePolicyError("generated patch exceeds configured limit")
        return patch

    def cleanup(self) -> None:
        if self.root is None:
            return
        root = self.root.resolve()
        temp_root = Path(tempfile.gettempdir()).resolve()
        if root.parent != temp_root or not root.name.startswith("agentforge-code-"):
            raise WorkspacePolicyError("refusing to remove an unexpected workspace path")
        if root.exists():
            shutil.rmtree(root)
        self.root = None
