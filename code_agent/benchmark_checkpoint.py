"""Single-host, case-boundary benchmark recovery; never an exactly-once API claim."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import tempfile
from pathlib import Path

from code_agent.verification import repository_fingerprint


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def checkpoint_scope(cases, root, policy, config_hash) -> str:
    sources = {}
    package = Path(__file__).parent
    for path in sorted(package.glob("*.py")):
        sources[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    for path in sorted((package.parent / "evals" / "experiments").glob("code_context*.json")):
        sources[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    repositories = {}
    for name in sorted({case.repository for case in cases}):
        repository = (root / name).resolve()
        repository.relative_to(root)
        if not repository.is_dir():
            raise ValueError("checkpoint repository is missing")
        repositories[name] = repository_fingerprint(repository, policy.max_files, policy.max_repository_bytes)
    versions = {}
    for name in ("langchain-core", "langchain-openai", "langchain-groq", "langgraph", "pydantic",
                 "openai", "groq", "tree-sitter", "tree-sitter-language-pack", "sentence-transformers"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "unavailable"
    # Bind backend configuration without persisting endpoint URLs or credentials.
    endpoints = {name: hashlib.sha256(os.getenv(name, "").encode()).hexdigest()
                 for name in ("OPENAI_BASE_URL", "OPENAI_API_BASE", "OPENAI_ORGANIZATION", "OPENAI_PROJECT", "GROQ_API_BASE")}
    retrieval = {name: value for name, value in os.environ.items() if name.startswith("CODE_CONTEXT_")}
    configured_fusion = os.getenv("CODE_CONTEXT_FUSION_ARTIFACT", "").strip()
    if configured_fusion:
        artifact = Path(configured_fusion)
        if not artifact.is_file() or artifact.stat().st_size > 10_000_000:
            raise ValueError("configured retrieval fusion artifact is missing or oversized")
        sources["configured-fusion"] = hashlib.sha256(artifact.read_bytes()).hexdigest()
    return digest({"root": str(root), "repositories": repositories, "sources": sources,
                   "interpreter": [platform.python_implementation(), platform.python_version(), platform.system(), platform.machine()],
                   "versions": versions, "endpoints": endpoints, "retrieval": digest(retrieval), "config": config_hash})


def image_identities(images: set[str]) -> dict[str, str]:
    identities = {}
    for image in sorted(images):
        result = subprocess.run(["docker", "image", "inspect", "--format", "{{.Id}}", image],
                                capture_output=True, text=True, timeout=15, check=False)
        identity = result.stdout.strip()
        if result.returncode or not identity.startswith("sha256:") or len(identity) != 71:
            raise ValueError("checkpoint requires an inspectable prebuilt Docker image")
        identities[image] = identity
    return identities


class BenchmarkCheckpoint:
    """Hash-bound local receipts, OS-released exclusive lock, atomic fsync writes.

    This detects accidental drift/corruption, not malicious tampering by the host
    owner. Artifacts can contain repository data and require private storage.
    """

    def __init__(self, directory: Path, *, resume: bool):
        self.directory = directory.absolute()
        self.resume = resume
        self.lock = None
        self.state = {}

    def safe_path(self, relative: str) -> Path:
        path = self.directory / relative
        path.resolve().relative_to(self.directory.resolve())
        for part in (path, *path.parents):
            if part.is_symlink():
                raise ValueError("checkpoint symlinks are not supported")
        return path

    def __enter__(self):
        self.safe_path("checkpoint.json")
        if self.resume:
            if not self.directory.is_dir():
                raise ValueError("checkpoint does not exist")
        else:
            self.directory.mkdir(parents=True, exist_ok=False)
        self.lock = self.safe_path(".checkpoint.lock").open("a+b")
        try:
            self.lock.seek(0, os.SEEK_END)
            if self.lock.tell() == 0:
                self.lock.write(b"0")
                self.lock.flush()
            self.lock.seek(0)
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(self.lock.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            self.lock.close()
            self.lock = None
            raise ValueError("checkpoint is already in use") from None
        return self

    def __exit__(self, *args):
        if self.lock:
            self.lock.close()  # OS releases lock even after a crashed process.
            self.lock = None

    def bind(self, scope: str, run_id: str, *, initial_metadata: dict | None = None):
        if initial_metadata is not None and (self.resume or set(initial_metadata) & {"schema", "scope", "run_id", "attempts"}):
            raise ValueError("initial checkpoint metadata cannot replace core bindings or change on resume")
        path = self.safe_path("checkpoint.json")
        if self.resume:
            if path.stat().st_size > 100_000_000:
                raise ValueError("checkpoint exceeds size limit")
            envelope = json.loads(path.read_text(encoding="utf-8"))
            state = envelope["state"]
            if envelope["sha256"] != digest(state) or state["scope"] != scope or state["schema"] != "1":
                raise ValueError("checkpoint integrity or task/runtime/image binding changed")
            if not isinstance(state.get("attempts"), dict):
                raise ValueError("invalid checkpoint attempts")
            self.state = state
            # Check EVERY retained attempt, not only the latest or successful one.
            for attempts in state["attempts"].values():
                for attempt in attempts:
                    for relative, expected in attempt.get("artifacts", {}).items():
                        artifact = self.safe_path(relative)
                        if artifact.stat().st_size != expected["bytes"] or hashlib.sha256(artifact.read_bytes()).hexdigest() != expected["sha256"]:
                            raise ValueError("checkpoint artifact integrity changed")
        else:
            self.state = {"schema": "1", "scope": scope, "run_id": run_id, "attempts": {}}
            self.state.update(initial_metadata or {})
            self.save()
        return self.state["run_id"]

    def save(self):
        destination = self.safe_path("checkpoint.json")
        descriptor, temporary = tempfile.mkstemp(prefix=".checkpoint-", dir=self.directory)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                json.dump({"state": self.state, "sha256": digest(self.state)}, stream, sort_keys=True)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, destination)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)

    def previous(self, case_id: str, *, retry_provider_hold: bool, retry_interrupted: bool):
        attempts = self.state["attempts"].get(case_id, [])
        if not attempts:
            return None
        latest = attempts[-1]
        if latest["status"] == "running" and not retry_interrupted:
            raise ValueError("interrupted case requires --retry-interrupted; its prior provider usage is unknown")
        if latest["status"] == "held" and not retry_provider_hold:
            raise ValueError("provider-held case requires --retry-provider-hold after restoring provider access")
        return latest if latest["status"] == "complete" else None

    def start(self, case_id: str) -> Path:
        attempts = self.state["attempts"].setdefault(case_id, [])
        number = sum(len(items) for items in self.state["attempts"].values()) + 1
        attempts.append({"status": "running", "number": number})
        self.save()  # Durable BEFORE any model request.
        directory = self.safe_path(f"attempts/{number}")
        directory.mkdir(parents=True, exist_ok=False)
        return directory

    def finish(self, case_id: str, outcome, prediction, *, held: bool):
        artifacts = {}
        for relative in (outcome.patch_path, outcome.trajectory_path):
            if relative:
                path = self.safe_path(relative)
                artifacts[relative] = {"sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "bytes": path.stat().st_size}
        attempt = self.state["attempts"][case_id][-1]
        attempt.update(status="held" if held else "complete", outcome=outcome.model_dump(mode="json"),
                       prediction=prediction, artifacts=artifacts)
        self.save()  # Called only after sandbox cleanup completes.
