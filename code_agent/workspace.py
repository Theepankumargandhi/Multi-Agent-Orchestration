"""Filtered, temporary workspace with path confinement and diff generation."""

from __future__ import annotations

import difflib
import fnmatch
import hashlib
import os
import re
import shutil
import stat
import tempfile
import threading
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
        self._edit_lock = threading.RLock()

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

    def read_lines(self, relative_path: str, start_line: int = 1, end_line: int = 100) -> dict:
        path = self.resolve(relative_path)
        relative = path.relative_to(self._require_root())
        if _is_secret_or_excluded(relative) or not path.is_file():
            raise WorkspacePolicyError("file is not exposed for bounded reads")
        if start_line < 1 or end_line < start_line or end_line - start_line >= 200:
            raise WorkspacePolicyError("read_lines permits an ordered range of at most 200 lines")
        if path.stat().st_size > self.policy.max_file_bytes:
            raise WorkspacePolicyError("file exceeds configured limit")
        data = path.read_bytes()
        if _is_binary(data):
            raise WorkspacePolicyError("binary files are not exposed to the agent")
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise WorkspacePolicyError("bounded edits require valid UTF-8") from exc
        parts = text.split("\n")
        lines = [part + "\n" for part in parts[:-1]] + ([parts[-1]] if parts[-1] else [])
        selected = "".join(lines[start_line - 1:end_line])
        if start_line > len(lines) or len(selected) > 4000:
            raise WorkspacePolicyError("range is absent or too large; request a smaller range")
        return {"path": relative.as_posix(), "file_sha256": hashlib.sha256(data).hexdigest(),
                "start_line": start_line, "end_line": min(end_line, len(lines)), "text": selected}

    def check_agent_edit_path(self, relative_path: str) -> Path:
        path = self.resolve(relative_path, allow_missing=True)
        relative = path.relative_to(self._require_root())
        raw = self._require_root() / relative_path.replace("\\", "/")
        for part in (raw, *raw.parents):
            if part == self._require_root().parent:
                break
            if part.is_symlink() or (part.exists() and getattr(part.lstat(), "st_file_attributes", 0) & 0x400):
                raise WorkspacePolicyError("agent edits may not traverse reparse points")
        if (_is_secret_or_excluded(relative) or relative.as_posix() in self.policy.controller_managed_paths
                or any(fnmatch.fnmatchcase(relative.as_posix(), rule) for rule in self.policy.protected_paths)):
            raise WorkspacePolicyError("agent edits cannot change protected or excluded files")
        if path.suffix.lower() not in ALLOWED_SUFFIXES and path.name not in ALLOWED_FILENAMES:
            raise WorkspacePolicyError("file type is not allowlisted for agent edits")
        return path

    def edit_file(self, relative_path: str, expected_sha256: str, old_text: str, new_text: str) -> str:
        """Compare-and-swap one unique exact block; no normalization or fuzzy matching."""
        with self._edit_lock:
            path = self.check_agent_edit_path(relative_path)
            if not path.is_file() or not old_text or max(len(old_text), len(new_text)) > 16000:
                raise WorkspacePolicyError("edit requires a bounded nonempty block in an existing file")
            data = path.read_bytes()
            if len(data) > self.policy.max_file_bytes or hashlib.sha256(data).hexdigest() != expected_sha256:
                raise WorkspacePolicyError("stale or missing file hash; read current lines again")
            try:
                text = data.decode("utf-8")
            except UnicodeDecodeError as exc:
                raise WorkspacePolicyError("edits require valid UTF-8") from exc
            first = text.find(old_text)
            if first < 0 or first != text.rfind(old_text):
                raise WorkspacePolicyError("old_text must match exactly once, including overlapping matches")
            if old_text == new_text:
                raise WorkspacePolicyError("no-op edit rejected")
            encoded = (text[:first] + new_text + text[first + len(old_text):]).encode("utf-8")
            if len(encoded) > self.policy.max_file_bytes or b"\x00" in encoded:
                raise WorkspacePolicyError("edited file exceeds text limits")
            temporary = None
            try:
                with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".agentforge-edit-", delete=False) as stream:
                    temporary = Path(stream.name)
                    stream.write(encoded)
                    stream.flush()
                    os.fsync(stream.fileno())
                temporary.chmod(path.stat().st_mode & 0o777)
                # Recheck after preparing the replacement, before atomic publication.
                if path.read_bytes() != data:
                    raise WorkspacePolicyError("file changed during edit")
                os.replace(temporary, path)
            finally:
                if temporary is not None and temporary.exists():
                    temporary.unlink()
            return hashlib.sha256(encoded).hexdigest()

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

    def original_file(self, relative_path: str) -> str | None:
        """Read the immutable filtered text snapshot, without opening the source repo."""
        path = self.resolve(relative_path, allow_missing=True)
        return self._original.get(path.relative_to(self._require_root()).as_posix())

    def unified_diff(self) -> str:
        current = self._snapshot()
        chunks: list[str] = []

        def lines(text: str | None) -> list[str]:
            if text is None:
                return []
            # Git's patch format splits on LF, not Unicode line separators.
            parts = text.split("\n")
            return [part + "\n" for part in parts[:-1]] + ([parts[-1]] if parts[-1] else [])

        for name in sorted(set(self._original) | set(current)):
            before = self._original.get(name)
            after = current.get(name)
            if before == after:
                continue
            for line in difflib.unified_diff(
                lines(before),
                lines(after),
                fromfile="/dev/null" if before is None else f"a/{name}",
                tofile="/dev/null" if after is None else f"b/{name}",
            ):
                chunks.append(line)
                if not line.endswith("\n"):
                    # difflib omits this marker, producing corrupt patches when
                    # an agent writes a file without a terminal newline.
                    chunks.append("\n\\ No newline at end of file\n")
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
