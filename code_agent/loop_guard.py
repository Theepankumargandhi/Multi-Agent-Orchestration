"""Bounded cycle evidence for action loops; no diagnosis from failure prose."""

from __future__ import annotations

import re

from code_agent.models import AgentLoopReceipt
from code_agent.patch_minimization import file_manifest
from code_agent.regression_challenges import fingerprint

# Normalize only pytest's terminal timing summary, not assertion values or paths.
_PYTEST_TIME = re.compile(r"(?m)^(=+\s*)?((?:\d+ (?:passed|failed|skipped|deselected|xfailed|xpassed|error|errors)(?:, )?)+) in \d+(?:\.\d+)?s(?: \([^\n]*\))?(\s*=+)?$")
NOTICE = ("The controller detected two identical tool/source cycles. Inspect different evidence "
          "or make a justified change; repeating this cycle reaches the configured stop limit. "
          "This is a repetition heuristic, not a root-cause diagnosis. Do not weaken tests or permissions.")


def event_digest(action, observation, source_sha256, test_result=None):
    output = observation.output
    if action.kind in {"test", "finish"}:
        output = _PYTEST_TIME.sub(lambda match: match.group(2), output)
    arguments = ({"path": action.path, "content_sha256": fingerprint(action.content)} if action.kind == "write"
                 else {"path": action.path, "expected_sha256": action.expected_sha256,
                       "old_sha256": fingerprint(action.old_text), "new_sha256": fingerprint(action.new_text)} if action.kind == "edit"
                 else {"path": action.path, "start_line": action.start_line, "end_line": action.end_line} if action.kind == "read_lines"
                 else {"path": action.path} if action.kind in {"read", "delete"}
                 else {"pattern_sha256": fingerprint(action.pattern)} if action.kind == "search" else {})
    return fingerprint({"action": action.kind, "arguments": arguments, "ok": observation.ok,
                        "output_sha256": fingerprint(output), "source_sha256": source_sha256,
                        "exit_code": test_result.exit_code if test_result is not None else None,
                        "timed_out": test_result.timed_out if test_result is not None else None})


def repeated_suffix(history, max_period, limit):
    """Prefer the shortest repeated suffix; retain at most the configured limit."""
    for period in range(1, max_period + 1):
        block = history[-period:]
        if len(block) != period:
            continue
        repeats = 1
        while repeats < limit and len(history) >= period * (repeats + 1):
            if history[-period * (repeats + 1):-period * repeats] != block:
                break
            repeats += 1
        if repeats >= 2:
            return period, repeats
    return 0, 0


class AgentLoopGuard:
    def __init__(self, policy, stage):
        self.policy, self.stage = policy, stage
        self.history = []
        self.receipts = []
        self.notice = ""
        self._emitted = set()

    def observe(self, workspace, action, observation, test_result=None):
        """Called after security enforcement and tool execution. Never skips a gate."""
        self.notice = ""
        try:
            source_sha = fingerprint(file_manifest(workspace, self.policy))
        except (OSError, ValueError, RuntimeError):
            self.history.clear()
            self._record(AgentLoopReceipt(stage=self.stage, iteration=observation.iteration,
                mode=self.policy.loop_guard_mode, disposition="unavailable", reason="source_manifest_unavailable"))
            return False
        self.history.append(event_digest(action, observation, source_sha, test_result))
        bound = self.policy.loop_guard_max_cycle_length * self.policy.loop_guard_repetition_limit
        self.history = self.history[-bound:]
        period, repeats = repeated_suffix(self.history, self.policy.loop_guard_max_cycle_length,
                                         self.policy.loop_guard_repetition_limit)
        if not period:
            return False
        disposition = ("warning" if repeats < self.policy.loop_guard_repetition_limit else
                       "stopped" if self.policy.loop_guard_mode == "enforce" else "shadow_limit")
        witness = self.history[-period * repeats:]
        key = fingerprint([disposition, witness])
        if key not in self._emitted:
            self._emitted.add(key)
            self._record(AgentLoopReceipt(stage=self.stage, iteration=observation.iteration,
                mode=self.policy.loop_guard_mode, disposition=disposition, reason="repeated_tool_source_cycle",
                cycle_length=period, repetitions=repeats, source_sha256=source_sha, witness_sha256=witness))
        if self.policy.loop_guard_mode == "enforce":
            self.notice = NOTICE
        return disposition == "stopped"

    def _record(self, receipt):
        receipt.repetition_limit = self.policy.loop_guard_repetition_limit
        receipt.max_cycle_length = self.policy.loop_guard_max_cycle_length
        receipt.fingerprint = fingerprint(receipt.model_dump(mode="json", exclude={"fingerprint"}))
        self.receipts.append(receipt)


def valid_loop_receipt(receipt):
    """Self-consistency only: digests do not authenticate sandbox outcomes."""
    if receipt.fingerprint != fingerprint(receipt.model_dump(mode="json", exclude={"fingerprint"})):
        return False
    if receipt.disposition == "unavailable":
        return (receipt.reason == "source_manifest_unavailable" and not receipt.witness_sha256
                and not receipt.source_sha256 and receipt.cycle_length == receipt.repetitions == 0)
    if (receipt.reason != "repeated_tool_source_cycle" or not receipt.cycle_length or receipt.repetitions < 2
            or receipt.cycle_length > receipt.max_cycle_length
            or len(receipt.witness_sha256) != receipt.cycle_length * receipt.repetitions
            or receipt.iteration < len(receipt.witness_sha256)
            or any(not re.fullmatch(r"[a-f0-9]{64}", value) for value in [receipt.source_sha256, *receipt.witness_sha256])
            or (receipt.disposition == "stopped" and receipt.mode != "enforce")
            or (receipt.disposition == "shadow_limit" and receipt.mode != "shadow")
            or (receipt.disposition == "warning" and receipt.repetitions >= receipt.repetition_limit)
            or (receipt.disposition != "warning" and receipt.repetitions != receipt.repetition_limit)):
        return False
    block = receipt.witness_sha256[:receipt.cycle_length]
    return receipt.witness_sha256 == block * receipt.repetitions
