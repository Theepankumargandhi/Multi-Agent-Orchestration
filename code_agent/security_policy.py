"""Deterministic complete-mediation policy for coding-agent tool actions.

The model can propose actions, but this module is the authorization authority.
It deliberately uses no LLM and persists only hashes/labels for untrusted input.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from collections.abc import Iterable
from pathlib import PurePosixPath

from code_agent.models import (
    CodeAction,
    CodeTask,
    SecurityArtifact,
    SecurityPolicyEvent,
    SecuritySource,
    SecuritySummary,
    ToolObservation,
)

POLICY_VERSION = "agentforge-security-v1"

_INJECTION_PATTERNS = (
    re.compile(r"(?i)ignore (?:all |any )?(?:previous|prior|system) instructions?"),
    re.compile(r"(?i)(?:system|developer)\s*(?:message|prompt)\s*:"),
    re.compile(r"(?i)do not tell (?:the )?(?:user|reviewer|operator)"),
    re.compile(r"(?i)(?:override|bypass|disable) (?:the )?(?:policy|sandbox|review|safety)"),
    re.compile(r"(?i)<\/?(?:system|assistant|tool)>"),
)
_SECRET_PATTERNS = (
    re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    re.compile(r"\b(?:sk|ghp|github_pat)-[A-Za-z0-9_-]{16,}\b"),
    re.compile(r"(?i)(?:api[_-]?key|password|secret|token)\s*[:=]\s*['\"][^'\"]{12,}['\"]"),
)
_UNSAFE_EXECUTION = (
    re.compile(r"\bos\.system\s*\("),
    re.compile(r"\bsubprocess\.(?:run|call|Popen)\s*\([^\n]{0,300}\bshell\s*=\s*True"),
    re.compile(r"\b(?:eval|exec)\s*\("),
    re.compile(r"\bpickle\.loads?\s*\("),
    re.compile(r"\byaml\.load\s*\((?![^\n]*SafeLoader)"),
    re.compile(r"\bverify\s*=\s*False\b"),
)
_NETWORK_SINKS = re.compile(
    r"(?i)(?:requests\.(?:get|post|put)|urllib\.request|socket\.create_connection|"
    r"https?://|\bcurl\b|\bwget\b)"
)
_SENSITIVE_SOURCES = re.compile(
    r"(?i)(?:os\.environ|getenv\s*\(|credentials?|api[_-]?key|password|secret|token|\.env)"
)
_TEST_TAMPERING = (
    re.compile(r"(?i)pytestmark\s*=\s*pytest\.mark\.skip"),
    re.compile(r"(?i)pytest_collection_modifyitems[^\n]{0,500}(?:skip|deselect)"),
    re.compile(r"(?i)(?:assert\s+True|return\s+True)\s*#\s*(?:bypass|skip|disable)"),
    re.compile(r"(?i)--(?:ignore|deselect|maxfail=0)"),
)
_RESOURCE_ABUSE = (
    re.compile(r":\(\)\s*\{\s*:\|:\s*&\s*\}\s*;\s*:"),
    re.compile(r"(?i)while\s+true\s*;\s*do"),
    re.compile(r"(?i)for\s*\(\s*;;\s*\)"),
)
_SENSITIVE_PARTS = {
    ".git",
    ".github",
    ".agentforge",
    ".ssh",
    "secrets",
    "credentials",
}
_SECRET_NAMES = {".env", ".npmrc", ".pypirc", "id_rsa", "id_ed25519"}


def _canonical_hash(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _matches(patterns: Iterable[re.Pattern[str]], value: str) -> bool:
    return any(pattern.search(value) for pattern in patterns)


def classify_untrusted_content(content: str) -> list[str]:
    """Return non-secret taint labels for untrusted data."""
    taints: list[str] = []
    if _matches(_INJECTION_PATTERNS, content):
        taints.append("instruction_injection")
    if _matches(_SECRET_PATTERNS, content):
        taints.append("secret_material")
    if _NETWORK_SINKS.search(content):
        taints.append("network_sink")
    if _SENSITIVE_SOURCES.search(content):
        taints.append("sensitive_source")
    return taints


def artifact_for_observation(observation: ToolObservation) -> SecurityArtifact | None:
    if not observation.output:
        return None
    source = "repository" if observation.action in {"read", "search", "list"} else "tool_output"
    return SecurityArtifact(
        source=source,
        trusted=False,
        sha256=hashlib.sha256(observation.output.encode("utf-8")).hexdigest(),
        taints=classify_untrusted_content(observation.output),
    )


def artifact_for_content(content: str, source: SecuritySource) -> SecurityArtifact | None:
    """Create a content-free provenance label for a prompt input."""
    if not content:
        return None
    return SecurityArtifact(
        source=source,
        trusted=False,
        sha256=hashlib.sha256(content.encode("utf-8")).hexdigest(),
        taints=classify_untrusted_content(content),
    )


class SecurityPolicyEngine:
    """Policy-as-code authorization boundary for every model-proposed action."""

    def __init__(self, *, enabled: bool = True, version: str = POLICY_VERSION):
        self.enabled = enabled
        self.version = version

    def fingerprint(self) -> str:
        rules = {
            "version": self.version,
            "enabled": self.enabled,
            "rules": [
                "AGENT-PATH-001",
                "AGENT-SECRET-001",
                "AGENT-EXEC-001",
                "AGENT-EXFIL-001",
                "AGENT-TEST-001",
                "AGENT-DOS-001",
                "AGENT-INJECTION-001",
            ],
        }
        return _canonical_hash(rules)

    def evaluate(
        self,
        action: CodeAction,
        task: CodeTask,
        *,
        stage: str = "implementation",
        artifacts: list[SecurityArtifact] | None = None,
    ) -> SecurityPolicyEvent:
        started = time.perf_counter()
        artifacts = artifacts or []
        rules: list[str] = []
        categories: list[str] = []
        severity = "none"
        source_taints = sorted({taint for item in artifacts for taint in item.taints})
        path = action.path.strip().replace("\\", "/")
        payload = "\n".join((action.content, action.rationale, action.pattern))

        if self.enabled and action.kind in {"read", "write", "delete"}:
            pure = PurePosixPath(path)
            unsafe_path = (
                not path
                or path.startswith("/")
                or bool(re.match(r"^[A-Za-z]:/", path))
                or ".." in pure.parts
                or any(part.lower() in _SENSITIVE_PARTS for part in pure.parts)
                or pure.name.lower() in _SECRET_NAMES
                or pure.suffix.lower() in {".pem", ".key", ".p12", ".pfx"}
            )
            if unsafe_path:
                rules.append("AGENT-PATH-001")
                categories.append("identity_privilege_abuse")
                severity = "critical"

        if self.enabled and action.kind == "write":
            if _matches(_SECRET_PATTERNS, action.content):
                rules.append("AGENT-SECRET-001")
                categories.append("sensitive_information_disclosure")
                severity = "critical"
            if _matches(_UNSAFE_EXECUTION, action.content):
                rules.append("AGENT-EXEC-001")
                categories.append("unexpected_code_execution")
                severity = "critical"
            if _NETWORK_SINKS.search(action.content) and _SENSITIVE_SOURCES.search(action.content):
                rules.append("AGENT-EXFIL-001")
                categories.append("data_exfiltration")
                severity = "critical"
            if _matches(_TEST_TAMPERING, action.content):
                rules.append("AGENT-TEST-001")
                categories.append("evaluation_tampering")
                severity = "high" if severity == "none" else severity
            if _matches(_RESOURCE_ABUSE, action.content):
                rules.append("AGENT-DOS-001")
                categories.append("unbounded_consumption")
                severity = "high" if severity == "none" else severity

        injection_in_action = _matches(_INJECTION_PATTERNS, payload)
        privileged_action = action.kind in {"write", "delete", "test"}
        if self.enabled and injection_in_action and privileged_action:
            rules.append("AGENT-INJECTION-001")
            categories.append("agent_goal_hijack")
            severity = "high" if severity == "none" else severity

        allowed = not rules
        requires_approval = bool(
            allowed
            and self.enabled
            and privileged_action
            and "instruction_injection" in source_taints
        )
        if requires_approval:
            # Suspicious context is recorded for the existing owner-approval boundary;
            # content/path-specific high-confidence rules still block immediately.
            severity = "medium"
            categories.append("untrusted_context_influence")

        decision_payload = {
            "policy": self.version,
            "action": action.kind,
            "path": path,
            "stage": stage,
            "allowed": allowed,
            "rules": sorted(set(rules)),
            "categories": sorted(set(categories)),
            "source_hashes": sorted(item.sha256 for item in artifacts),
        }
        return SecurityPolicyEvent(
            decision_id=_canonical_hash(decision_payload),
            policy_version=self.version,
            action=action.kind,
            path=path,
            stage=stage,
            allowed=allowed,
            severity=severity,
            rule_ids=sorted(set(rules)),
            categories=sorted(set(categories)),
            source_taints=source_taints,
            requires_human_approval=requires_approval,
            latency_ms=(time.perf_counter() - started) * 1000,
        )


def summarize_security(events: list[SecurityPolicyEvent], version: str) -> SecuritySummary:
    payload = [event.model_dump(mode="json", exclude={"latency_ms"}) for event in events]
    return SecuritySummary(
        policy_version=version,
        evaluated_actions=len(events),
        blocked_actions=sum(not event.allowed for event in events),
        warned_actions=sum(event.requires_human_approval for event in events),
        tainted_decisions=sum(bool(event.source_taints) for event in events),
        events=events,
        fingerprint=_canonical_hash(payload),
    )
