"""Bounded receipt revalidation across document revisions; no implicit persistence or retries."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Annotated, Literal

from pydantic import Field, field_validator

from agentforge_sdk.document_changes import (
    CalculationReplay,
    ChangeConfig,
    DocumentRevision,
    Reader,
    ReceiptImpact,
    _context,
    _inner,
)
from agentforge_sdk.evidence_review import ReviewedAnswer, ReviewedCalculation
from agentforge_sdk.models import Contract, SDKError, fingerprint
from agentforge_sdk.multimodal import _seal
from agentforge_sdk.rag import _normalize_question
from agentforge_sdk.spatial import Calculation, SpatialAnswer

_HASH = r"^[a-f0-9]{64}$"
_ID = r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,119}$"
_MAX_CHECKPOINT_BYTES = 16_000_000


class SavedAnswer(Contract):
    kind: Literal["answer"] = "answer"
    item_id: str = Field(pattern=_ID)
    question: str = Field(min_length=1, max_length=4000, repr=False)
    receipt: SpatialAnswer | ReviewedAnswer = Field(repr=False)

    @field_validator("question")
    @classmethod
    def valid_question(cls, value):
        return _normalize_question(value)


class SavedCalculation(Contract):
    kind: Literal["calculation"] = "calculation"
    item_id: str = Field(pattern=_ID)
    receipt: Calculation | ReviewedCalculation = Field(repr=False)


SavedReceipt = Annotated[SavedAnswer | SavedCalculation, Field(discriminator="kind")]


class RevalidationConfig(Contract):
    # Refresh is an explicit execution decision. Assessment alone calls no backends.
    refresh_answers: bool = False
    replay_calculations: bool = False
    max_items_per_step: int = Field(default=8, ge=1, le=64)
    max_answer_attempts: int = Field(default=8, ge=0, le=64)
    answer_top_k: int = Field(default=3, ge=1, le=20)


class RevalidationEntry(Contract):
    item_id: str = Field(pattern=_ID)
    input_fingerprint: str = Field(pattern=_HASH)
    impact: ReceiptImpact = Field(repr=False)
    status: Literal["current", "refresh_required", "refreshed", "blocked", "abstained", "failed"]
    reason: Literal[
        "verified_current", "refresh_not_enabled", "answer_budget_exhausted", "source_unverifiable",
        "revision_held", "alignment_unavailable", "fresh_answer", "tool_replayed", "no_fresh_evidence",
        "tool_abstained", "answer_execution_failed",
    ]
    answer_attempted: bool = False
    answer: SpatialAnswer | ReviewedAnswer | None = Field(default=None, repr=False)
    replay: CalculationReplay | None = Field(default=None, repr=False)
    result_changed: bool | None = None
    # SDK answer attempts are not provider requests, token usage, or a dollar cap.
    provider_cost_usd: float | None = Field(default=0.0, ge=0)
    review_required: bool
    fingerprint: str = Field(pattern=_HASH)


class RevalidationCheckpoint(Contract):
    protocol: Literal["agentforge-revalidation-v1"] = "agentforge-revalidation-v1"
    batch_fingerprint: str = Field(pattern=_HASH)
    entries: tuple[RevalidationEntry, ...] = Field(max_length=64, repr=False)
    total_items: int = Field(ge=1, le=64)
    complete: bool
    answer_attempts: int = Field(ge=0, le=64)
    provider_cost_usd: float | None = Field(ge=0)
    fingerprint: str = Field(pattern=_HASH)


class RevalidationReview(Contract):
    """Content-minimal application queue: no questions, excerpts, or numeric values."""

    item_id: str = Field(pattern=_ID)
    entry_fingerprint: str = Field(pattern=_HASH)
    category: Literal["blocked", "changed_result", "refresh_required", "abstained", "failed"]
    reason: str


class CheckpointError(SDKError):
    """A checkpoint is malformed, belongs to another batch, or is no longer valid."""


class RevalidationBatch:
    """Revalidate up to 64 saved receipts, in stable input order and bounded steps.

    Checkpoints are integrity-checked application data, NOT signed execution logs.
    The application owns storage and synchronization. Persist a returned checkpoint
    before continuing. A crash during an answer call can lose that call's result;
    resuming the previous checkpoint can repeat it. There is no exactly-once claim.
    """

    def __init__(self, before: Reader, after: Reader, items: Sequence[SavedAnswer | SavedCalculation], *,
                 config: RevalidationConfig | None = None, change_config: ChangeConfig | None = None):
        if not isinstance(items, (list, tuple)) or not 1 <= len(items) <= 64:
            raise ValueError("revalidation requires between 1 and 64 saved receipts")
        validated = []
        for item in items:
            if not isinstance(item, (SavedAnswer, SavedCalculation)):
                raise ValueError("revalidation requires typed saved receipts")
            validated.append(type(item).model_validate(item.model_dump()))
        self._items = tuple(validated)
        if len({i.item_id for i in self._items}) != len(self._items):
            raise ValueError("saved receipt IDs must be unique")
        if sum(len(i.model_dump_json().encode("utf-8")) for i in self._items) > 2_000_000:
            raise ValueError("saved receipt payload exceeds the batch byte budget")
        self._config = RevalidationConfig.model_validate((config or RevalidationConfig()).model_dump())
        self._revision = DocumentRevision(before, after, config=change_config)
        self._after = after
        self._input_hashes = tuple(fingerprint(i.model_dump(mode="json")) for i in self._items)
        self._fingerprint = fingerprint(dict(
            before=_context(before), after=_context(after), changes=self._revision.changes.fingerprint,
            config=self._config.model_dump(mode="json"), inputs=self._input_hashes,
        ))

    @property
    def fingerprint(self) -> str:
        return self._fingerprint

    def _impact(self, item):
        return (self._revision.assess_answer(item.receipt, question=item.question) if isinstance(item, SavedAnswer)
                else self._revision.assess_calculation(item.receipt))

    @staticmethod
    def _blocked(impact):
        if impact.status == "invalid_receipt":
            return "source_unverifiable"
        if any(r in impact.reasons for r in (
            "incomplete_extraction", "comparison_budget", "target_not_authorized", "extraction_profile_changed",
        )):
            return "revision_held"
        return None

    def _entry(self, index, impact, *, attempted=False, answer=None, replay=None, failed=False, budget=False):
        item = self._items[index]
        status, reason, changed, cost = "blocked", self._blocked(impact), None, 0.0
        if attempted:
            cost = None
        if failed:
            status, reason = "failed", "answer_execution_failed"
        elif reason is not None:
            pass
        elif answer is not None:
            result, old = _inner(answer), _inner(item.receipt)
            status = "refreshed" if result.status == "evidence_found" else "abstained"
            reason = "fresh_answer" if status == "refreshed" else "no_fresh_evidence"
            changed = (old.status, old.answer) != (result.status, result.answer)
            cost = result.text_receipt.provider_cost_usd if result.text_receipt else 0.0
        elif replay is not None:
            status = "refreshed" if replay.status == "recalculated" else "abstained"
            reason = "tool_replayed" if status == "refreshed" else "tool_abstained"
            changed = replay.value_changed
        elif impact.status == "dependencies_unchanged" and not impact.requires_reissue:
            status, reason = "current", "verified_current"
        elif isinstance(item, SavedCalculation) and impact.status == "needs_review":
            reason = "alignment_unavailable"
        else:
            status, reason = "refresh_required", "answer_budget_exhausted" if budget else "refresh_not_enabled"
        review = status not in {"current", "refreshed"} or changed is True or impact.status in {"dependencies_changed", "needs_review"}
        return _seal(RevalidationEntry, dict(
            item_id=item.item_id, input_fingerprint=self._input_hashes[index], impact=impact,
            status=status, reason=reason, answer_attempted=attempted, answer=answer, replay=replay,
            result_changed=changed, provider_cost_usd=cost, review_required=review,
        ))

    def _checkpoint(self, entries):
        costs = [e.provider_cost_usd for e in entries]
        return _seal(RevalidationCheckpoint, dict(
            batch_fingerprint=self._fingerprint, entries=tuple(entries), total_items=len(self._items),
            complete=len(entries) == len(self._items), answer_attempts=sum(e.answer_attempted for e in entries),
            provider_cost_usd=math.fsum(costs) if all(c is not None for c in costs) else None,
        ))

    def _validate(self, checkpoint):
        checkpoint = RevalidationCheckpoint.model_validate(checkpoint.model_dump())
        if checkpoint.batch_fingerprint != self._fingerprint or len(checkpoint.entries) > len(self._items):
            raise ValueError("checkpoint batch mismatch")
        attempts = 0
        for index, entry in enumerate(checkpoint.entries):
            item, impact = self._items[index], self._impact(self._items[index])
            if entry.impact != impact:
                raise ValueError("checkpoint evidence or authorization changed")
            executable = (self._blocked(impact) is None
                          and (impact.requires_reissue or impact.status != "dependencies_unchanged"))
            if entry.answer is not None and entry.replay is not None:
                raise ValueError("checkpoint contains conflicting outputs")
            if entry.answer is not None:
                if (not isinstance(item, SavedAnswer) or not entry.answer_attempted
                        or not self._after.verify_answer(entry.answer, question=item.question)):
                    raise ValueError("checkpoint answer is unverifiable")
            if entry.replay is not None:
                if (not isinstance(item, SavedCalculation) or not self._config.replay_calculations
                        or not self._revision.verify_recalculation(entry.replay, item.receipt)):
                    raise ValueError("checkpoint replay is unverifiable")
            failed = entry.status == "failed"
            if entry.answer_attempted:
                if (not isinstance(item, SavedAnswer) or not self._config.refresh_answers
                        or (entry.answer is None and not failed) or entry.replay is not None
                        or (not executable and not failed)):
                    raise ValueError("checkpoint attempt is inconsistent")
                attempts += 1
            if failed and (not entry.answer_attempted or entry.answer is not None):
                raise ValueError("checkpoint failure is inconsistent")
            budget = entry.reason == "answer_budget_exhausted"
            if budget and (not isinstance(item, SavedAnswer) or not self._config.refresh_answers
                           or attempts < self._config.max_answer_attempts):
                raise ValueError("checkpoint budget is inconsistent")
            if executable and isinstance(item, SavedAnswer) and self._config.refresh_answers:
                if not entry.answer_attempted and not budget:
                    raise ValueError("checkpoint skipped an enabled answer operation")
            if executable and isinstance(item, SavedCalculation) and self._config.replay_calculations:
                if impact.status != "needs_review" and entry.replay is None:
                    raise ValueError("checkpoint skipped an enabled tool operation")
            if entry.replay is not None and not executable:
                raise ValueError("checkpoint replay was unnecessary or blocked")
            expected = self._entry(index, impact, attempted=entry.answer_attempted, answer=entry.answer,
                                   replay=entry.replay, failed=failed, budget=budget)
            if expected != entry:
                raise ValueError("checkpoint entry is inconsistent")
        if attempts > self._config.max_answer_attempts or self._checkpoint(checkpoint.entries) != checkpoint:
            raise ValueError("checkpoint summary is inconsistent")
        return checkpoint

    def verify_checkpoint(self, checkpoint: RevalidationCheckpoint) -> bool:
        """Recheck evidence and outputs without rerunning any answer backends."""
        try:
            self._validate(checkpoint)
            return True
        except (ValueError, TypeError, AttributeError, KeyError, SDKError):
            return False

    def restore(self, serialized: str | bytes) -> RevalidationCheckpoint:
        """Bounded JSON admission; callers must still protect checkpoint storage."""
        try:
            if not isinstance(serialized, (str, bytes)) or len(serialized) > _MAX_CHECKPOINT_BYTES:
                raise ValueError("checkpoint size exceeded")
            if isinstance(serialized, str) and len(serialized.encode("utf-8")) > _MAX_CHECKPOINT_BYTES:
                raise ValueError("checkpoint size exceeded")
            return self._validate(RevalidationCheckpoint.model_validate_json(serialized))
        except (ValueError, TypeError, AttributeError, KeyError, SDKError):
            raise CheckpointError("checkpoint is invalid or its evidence is no longer authorized") from None

    def advance(self, checkpoint: RevalidationCheckpoint | None = None) -> RevalidationCheckpoint:
        """Process one bounded step. No retries, background tasks, or implicit disk writes."""
        if checkpoint is not None:
            try:
                checkpoint = self._validate(checkpoint)
            except (ValueError, TypeError, AttributeError, KeyError, SDKError):
                raise CheckpointError("checkpoint is invalid or its evidence is no longer authorized") from None
        entries = list(checkpoint.entries) if checkpoint is not None else []
        attempts = sum(e.answer_attempted for e in entries)
        stop = min(len(self._items), len(entries) + self._config.max_items_per_step)
        for index in range(len(entries), stop):
            item, impact = self._items[index], self._impact(self._items[index])
            kwargs = {}
            if self._blocked(impact) is None and (impact.requires_reissue or impact.status != "dependencies_unchanged"):
                if isinstance(item, SavedAnswer) and self._config.refresh_answers:
                    if attempts >= self._config.max_answer_attempts:
                        kwargs["budget"] = True
                    else:
                        attempts += 1
                        kwargs["attempted"] = True
                        try:
                            answer = self._after.answer(item.question, top_k=self._config.answer_top_k)
                            impact = self._impact(item)
                            if self._blocked(impact) or not self._after.verify_answer(answer, question=item.question):
                                raise SDKError("answer authorization changed")
                            kwargs["answer"] = answer
                        except SDKError:
                            kwargs["failed"] = True
                elif (isinstance(item, SavedCalculation) and self._config.replay_calculations
                      and impact.status != "needs_review"):
                    kwargs["replay"] = self._revision.recalculate(item.receipt)
                    impact = self._impact(item)
            entries.append(self._entry(index, impact, **kwargs))
        result = self._checkpoint(entries)
        if len(result.model_dump_json().encode("utf-8")) > _MAX_CHECKPOINT_BYTES:
            raise CheckpointError("checkpoint exceeds its serialized byte budget")
        # Authority or source bytes may change while a later item is running.
        # Never return earlier results as current without one final no-provider check.
        if not self.verify_checkpoint(result):
            raise CheckpointError("evidence changed during revalidation; retain the last valid checkpoint")
        return result

    def review_queue(self, checkpoint: RevalidationCheckpoint) -> tuple[RevalidationReview, ...]:
        """Return only outstanding processed items; unprocessed work is not a clean result."""
        if not self.verify_checkpoint(checkpoint):
            raise CheckpointError("cannot build a queue from an invalid checkpoint")
        queue = []
        for entry in checkpoint.entries:
            if entry.review_required:
                category = "changed_result" if entry.status == "refreshed" else entry.status
                queue.append(RevalidationReview(item_id=entry.item_id, entry_fingerprint=entry.fingerprint,
                                                category=category, reason=entry.reason))
        priority = {"blocked": 0, "failed": 1, "abstained": 2, "changed_result": 3, "refresh_required": 4}
        return tuple(sorted(queue, key=lambda q: (priority[q.category], q.item_id)))
