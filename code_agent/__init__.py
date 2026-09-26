"""Sandboxed software-engineering agent primitives with lazy public imports."""

from __future__ import annotations

from importlib import import_module

__all__ = [
    "CodeAgentResult",
    "CodeContextFile",
    "CodeContextReceipt",
    "CodeIntelligenceIndex",
    "CodeParser",
    "CodeTask",
    "ContextPack",
    "RetrievalConfig",
    "CodingAgent",
    "DockerSandbox",
    "SandboxPolicy",
    "SecurityPolicyEngine",
    "VerifiedPRAgent",
    "build_coding_model",
    "build_repository_map",
    "build_verified_pr_agent",
]

_EXPORTS = {
    "CodeAgentResult": ("code_agent.models", "CodeAgentResult"),
    "CodeContextFile": ("code_agent.models", "CodeContextFile"),
    "CodeContextReceipt": ("code_agent.models", "CodeContextReceipt"),
    "CodeIntelligenceIndex": ("code_agent.intelligence", "CodeIntelligenceIndex"),
    "CodeParser": ("code_agent.code_parsing", "CodeParser"),
    "CodeTask": ("code_agent.models", "CodeTask"),
    "ContextPack": ("code_agent.intelligence", "ContextPack"),
    "RetrievalConfig": ("code_agent.retrieval_backends", "RetrievalConfig"),
    "CodingAgent": ("code_agent.agent", "CodingAgent"),
    "DockerSandbox": ("code_agent.sandbox", "DockerSandbox"),
    "SandboxPolicy": ("code_agent.models", "SandboxPolicy"),
    "SecurityPolicyEngine": ("code_agent.security_policy", "SecurityPolicyEngine"),
    "VerifiedPRAgent": ("code_agent.verified_pr", "VerifiedPRAgent"),
    "build_coding_model": ("code_agent.agent", "build_coding_model"),
    "build_repository_map": ("code_agent.repository_map", "build_repository_map"),
    "build_verified_pr_agent": ("code_agent.verified_pr", "build_verified_pr_agent"),
}


def __getattr__(name: str):
    try:
        module_name, attribute = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(name) from exc
    value = getattr(import_module(module_name), attribute)
    globals()[name] = value
    return value
