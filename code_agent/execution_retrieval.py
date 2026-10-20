"""Local execution-guided retrieval; log text never becomes retrieval instructions.

Only known repository paths in recognizable failure locations are extracted.
No model call, remote embedding, package installation, or command is added.
"""

from __future__ import annotations

import hashlib
import json
import re
import time

from code_agent.intelligence import CodeIntelligenceIndex
from code_agent.models import ExecutionRetrievalReceipt, SandboxPolicy, ToolObservation
from code_agent.retrieval_backends import RetrievalConfig
from code_agent.security_policy import classify_untrusted_content

_LOCATION = re.compile(
    r'File "([^"\n]{1,500})", line \d+|'
    r"(?<![\w./\\:-])([\w./\\-]+\.(?:py|pyi|ts|tsx|js|jsx|java|go|rs)):\d+|"
    r"FAILED ([\w./\\-]+\.(?:py|pyi|ts|tsx|js|jsx|java|go|rs))::"
)


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def failure_paths(output: str, known_paths: set[str]) -> list[str]:
    """Resolve bounded location tokens, not arbitrary paths or prose in a log.

    Absolute paths outside Docker's /workspace, traversal, and ambiguous
    basenames are ignored. JVM-style basename locations must be unique.
    """
    paths = []
    for match in _LOCATION.finditer(output[:20_000]):
        path = next(value for value in match.groups() if value).replace("\\", "/")
        if path.startswith("/workspace/"):
            path = path[len("/workspace/"):]
        elif path.startswith("/") or ":" in path:
            continue
        if ".." in path.split("/"):
            continue
        path = path.removeprefix("./")
        if path not in known_paths and "/" not in path:
            matches = [item for item in known_paths if item.rsplit("/", 1)[-1] == path]
            path = matches[0] if len(matches) == 1 else ""
        if path in known_paths and path not in paths:
            paths.append(path)
        if len(paths) >= 8:
            break
    return paths


class ExecutionRetriever:
    """One per stage. Unchanged observations consume no new refresh.

    Local hashing embeddings + feature reranking deliberately avoid environment
    configured neural downloads/providers. Source snippets remain untrusted.
    Any write/delete invalidates the old pack even if the refresh cap is reached.
    """

    def __init__(self, policy: SandboxPolicy, stage: str):
        self.policy = policy
        self.stage = stage
        self.index = None
        self.context = ""
        self.receipts: list[ExecutionRetrievalReceipt] = []
        self.refreshes = 0
        self.last_key = None
        self.config = RetrievalConfig(
            embedding_backend="hashing", reranker_backend="feature", fusion_artifact="",
            prefer_tree_sitter=False, packing_policy="balanced_v1",
        )

    def refresh(self, workspace, observations: list[ToolObservation]) -> str:
        tests = [item for item in observations if item.action in {"test", "finish"}]
        if not tests or tests[-1].ok:
            self.context = ""
            return ""
        failure = tests[-1]
        edits = [item.iteration for item in observations
                 if item.ok and item.action in {"write", "edit", "delete"}]
        revision = max(edits, default=-1)
        failure_hash = _sha(failure.output)
        key = (failure.iteration, failure_hash, revision)
        if key == self.last_key:
            return self.context
        self.last_key = key
        self.context = ""  # Never return pre-edit source after a new edit.
        trigger = ("workspace_changed" if revision > failure.iteration else
                   "baseline_failure" if failure.iteration == 0 else "test_failure")
        receipt = ExecutionRetrievalReceipt(
            iteration=max((item.iteration for item in observations), default=0),
            stage=self.stage, trigger=trigger, status="error", failure_sha256=failure_hash,
        )
        started = time.perf_counter()
        try:
            if self.refreshes >= self.policy.execution_retrieval_max_refreshes:
                receipt.status = "budget_exhausted"
                return ""
            self.refreshes += 1
            self.index = CodeIntelligenceIndex.build(
                workspace, previous=self.index, config=self.config,
                max_files=min(self.policy.max_files, 1000),
                max_file_chars=min(self.policy.max_file_bytes, 250_000),
            )
            receipt.parsed_files = self.index.stats.parsed_files
            receipt.reused_files = self.index.stats.reused_files
            receipt.index_sha256 = _sha(json.dumps(
                {path: item.sha256 for path, item in sorted(self.index.files.items())},
                sort_keys=True,
            ))
            anchors = failure_paths(failure.output, set(self.index.files))
            receipt.anchor_paths = anchors
            if not anchors:
                receipt.status = "no_anchors"
                return ""
            # No exception message, assertion value, or arbitrary log prose enters the query.
            query = "failure locations " + " ".join(anchors) + " imports calls dependencies tests"
            receipt.query_sha256 = _sha(query)
            pack = self.index.select(
                query, top_k=6, max_tokens=self.policy.execution_retrieval_max_tokens,
                strategy="hybrid_rerank", packing_policy="balanced_v1",
            )
            # Preserve admission checks, and do not introduce newly retrieved secret material.
            if "secret_material" in classify_untrusted_content(pack.prompt_context):
                receipt.status = "unsafe_context"
                return ""
            self.context = pack.prompt_context
            receipt.context_sha256 = _sha(self.context)
            receipt.selected_file_sha256 = {
                item.path: item.sha256 for item in pack.receipt.selected_files
            }
            receipt.estimated_tokens = pack.receipt.estimated_tokens
            receipt.status = "selected"
            return self.context
        except (OSError, ValueError, RuntimeError, KeyError, TypeError):
            # Local retrieval is optional; never leak the exception body or mask provider failures.
            return ""
        finally:
            receipt.latency_ms = (time.perf_counter() - started) * 1000
            self.receipts.append(receipt)
