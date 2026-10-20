"""Bounded natural-language planning over literal evidence and registered numeric tools."""

from __future__ import annotations

import hashlib
import math
import re
from typing import Literal, Protocol

from pydantic import Field, model_validator

from agentforge_sdk.models import (
    Citation,
    Contract,
    DocumentPage,
    ExternalProcessingDenied,
    RAGConfig,
    SDKError,
    SearchResult,
    fingerprint,
)
from agentforge_sdk.multimodal import _seal
from agentforge_sdk.rag import DocumentRAG, _normalize_question, _quotes
from agentforge_sdk.reconciliation import EvidenceSet, FactCalculation, FactQuery, FactReconciliation

_HASH = r"^[a-f0-9]{64}$"
_ABSTAIN = "No verified document answer is available."


class PlannerUsage(Contract):
    input_tokens: int | None = Field(default=None, ge=0, le=10_000_000)
    output_tokens: int | None = Field(default=None, ge=0, le=10_000_000)
    provider_requests: int | None = Field(default=None, ge=0, le=100)
    provider_cost_usd: float | None = Field(default=None, ge=0)


class ReasoningPlan(Contract):
    action: Literal["quote", "lookup", "calculate", "abstain"]
    candidate_ids: tuple[str, ...] = Field(default=(), max_length=16)
    operation: Literal["sum", "difference", "ratio", "percent_change"] | None = None

    @model_validator(mode="after")
    def valid_shape(self):
        if (len(set(self.candidate_ids)) != len(self.candidate_ids)
                or any(not re.fullmatch(_HASH, c) for c in self.candidate_ids)):
            raise ValueError("plan candidates require unique evidence IDs")
        count = len(self.candidate_ids)
        if self.action == "abstain":
            valid = count == 0 and self.operation is None
        elif self.action == "calculate":
            valid = self.operation is not None and (1 <= count <= 16 if self.operation == "sum" else count == 2)
        else:
            valid = self.operation is None and (count == 1 if self.action == "lookup" else 1 <= count <= 3)
        if not valid:
            raise ValueError("plan action has invalid operation or arity")
        return self


class ReasoningCard(Contract):
    candidate_id: str = Field(pattern=_HASH)
    kind: Literal["quote", "fact"]
    text: str = Field(min_length=1, max_length=1000, repr=False)
    evidence_id: str | None = None
    fact_query: FactQuery | None = Field(default=None, repr=False)
    source: str | None = Field(default=None, max_length=500)
    page: int | None = Field(default=None, ge=1)


class PlannerRequest(Contract):
    question: str = Field(min_length=1, max_length=4000, repr=False)
    candidates: tuple[ReasoningCard, ...] = Field(max_length=40, repr=False)
    previous_failure: Literal["invalid_plan", "candidate_unavailable"] | None = None
    attempts_remaining: int = Field(ge=0, le=2)


class PlannerOutput(Contract):
    plan_json: str = Field(min_length=2, max_length=8000, repr=False)
    usage: PlannerUsage = Field(default_factory=PlannerUsage)


class ReasoningPlanner(Protocol):
    name: str
    external_processing: bool

    def plan(self, request: PlannerRequest) -> PlannerOutput: ...


class ReasoningConfig(Contract):
    max_planner_calls: int = Field(default=2, ge=1, le=3)
    retrieval_top_k: int = Field(default=6, ge=1, le=10)
    recovery_top_k: int = Field(default=12, ge=1, le=20)
    max_prompt_chars: int = Field(default=24_000, ge=2000, le=64_000)
    max_fact_candidates: int = Field(default=1000, ge=1, le=1000)
    allow_external_processing: bool = False


class ReasoningAttempt(Contract):
    number: int = Field(ge=1, le=3)
    request: PlannerRequest = Field(repr=False)
    text_retrieval: SearchResult = Field(repr=False)
    fact_retrieval: SearchResult = Field(repr=False)
    plan: ReasoningPlan | None = None
    outcome: Literal["answered", "invalid_plan", "candidate_unavailable", "planner_failed", "model_abstained", "unresolved_fact", "tool_abstained"]
    usage: PlannerUsage


class ReasoningResult(Contract):
    protocol: Literal["agentforge-document-reasoning-v1"] = "agentforge-document-reasoning-v1"
    agent_fingerprint: str = Field(pattern=_HASH)
    query_sha256: str = Field(pattern=_HASH)
    status: Literal["answered", "abstained"]
    reason: str
    answer: str = Field(repr=False)
    attempts: tuple[ReasoningAttempt, ...] = Field(max_length=3, repr=False)
    citations: tuple[Citation, ...] = Field(default=(), max_length=3, repr=False)
    fact: FactReconciliation | None = Field(default=None, repr=False)
    calculation: FactCalculation | None = Field(default=None, repr=False)
    planner_calls: int = Field(ge=0, le=3)
    provider_cost_usd: float | None = Field(ge=0)
    fingerprint: str = Field(pattern=_HASH)


class DocumentReasoningAgent:
    """Model proposals are untrusted; only literal quotes and checked tools can be released.

    Verification proves evidence/tool lineage, NOT that the model interpreted the
    question correctly. Configured scopes are application authority, not model output.
    """

    def __init__(self, documents: DocumentRAG, planner: ReasoningPlanner, *, evidence: EvidenceSet | None = None,
                 config: ReasoningConfig | None = None):
        self._config = ReasoningConfig.model_validate((config or ReasoningConfig()).model_dump())
        if not isinstance(documents, DocumentRAG) or documents.config.allow_external_processing:
            raise ValueError("reasoning retrieval requires a local-only DocumentRAG")
        if evidence is not None and not isinstance(evidence, EvidenceSet):
            raise ValueError("numeric reasoning requires an explicit EvidenceSet")
        self._documents, self._planner, self._evidence = documents, planner, evidence
        self._planner_name = self._backend()
        queries = evidence.query_catalog() if evidence is not None else ()
        if len(queries) > self._config.max_fact_candidates:
            raise ValueError("registered fact catalog exceeds its budget; select a smaller corpus")
        self._queries = {fingerprint(q.model_dump(mode="json")): q for q in queries}
        self._facts = DocumentRAG([DocumentPage(document_id="f:" + key, source="registered-fact",
            text=f"{q.scope} | {q.row_label} | {q.column_label}") for key, q in self._queries.items()],
            config=RAGConfig(chunk_size=1000, chunk_overlap=0, max_pages=1000, max_chunks=1000,
                             max_context_chars=32_000, max_quote_chars=600))
        self._fingerprint = fingerprint(dict(documents=documents.corpus_fingerprint,
            evidence=evidence.fingerprint if evidence is not None else None,
            fact_catalog=self._facts.corpus_fingerprint, planner=self._planner_name,
            config=self._config.model_dump(mode="json")))

    def _backend(self):
        name, external = getattr(self._planner, "name", None), getattr(self._planner, "external_processing", None)
        if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9_.:/-]{1,120}", name) or type(external) is not bool:
            raise ValueError("planner must declare a bounded name and external_processing boolean")
        if external and not self._config.allow_external_processing:
            raise ExternalProcessingDenied("reasoning planner requires explicit external-processing consent")
        return name

    @property
    def fingerprint(self) -> str:
        return self._fingerprint

    def preview(self, question: str) -> PlannerRequest:
        """Expose the same initial bounded candidates to a comparison baseline; no planner call."""
        question = _normalize_question(question)
        if self._evidence is not None:
            self._evidence.query_catalog()
        top_k = self._config.retrieval_top_k
        text, facts = self._documents.search(question, top_k=top_k), self._facts.search(question, top_k=top_k)
        return PlannerRequest(question=question, candidates=self._cards(question, text, facts),
                              attempts_remaining=self._config.max_planner_calls - 1)

    def _cards(self, question, text, facts):
        cards = []
        for item in facts.evidence:
            query = self._queries[item.document_id[2:]]
            payload = dict(kind="fact", text=item.text, fact_query=query, evidence_id=None)
            cards.append(ReasoningCard(candidate_id=fingerprint(dict(agent=self._fingerprint, query=query.model_dump(mode="json"))), **payload))
        by_id = {item.evidence_id: item for item in text.evidence}
        for selection in _quotes(question, text.evidence, 600):
            item = by_id[selection.evidence_id]
            payload = dict(kind="quote", text=selection.quote, evidence_id=item.evidence_id, fact_query=None)
            payload.update(source=item.source, page=item.page)
            cards.append(ReasoningCard(candidate_id=fingerprint(dict(agent=self._fingerprint, evidence=item.evidence_id, quote=selection.quote)), **payload))
        packed, used = [], len(question)
        for card in cards:
            size = len(card.model_dump_json())
            if used + size <= self._config.max_prompt_chars:
                packed.append(card)
                used += size
        return tuple(packed)

    def _execute(self, plan, cards, retrieval):
        by_id = {c.candidate_id: c for c in cards}
        if plan.action == "abstain":
            return "model_abstained", {}
        if any(key not in by_id for key in plan.candidate_ids):
            return "candidate_unavailable", {}
        chosen = [by_id[key] for key in plan.candidate_ids]
        if plan.action == "quote":
            if any(c.kind != "quote" for c in chosen):
                return "invalid_plan", {}
            actual = {e.evidence_id: e for e in retrieval.evidence}
            citations = []
            for card in chosen:
                item = actual[card.evidence_id]
                start = item.text.index(card.text)
                citation = Citation(evidence_id=item.evidence_id, document_id=item.document_id,
                    source=item.source, version=item.version, page=item.page, input_text_sha256=item.input_text_sha256,
                    chunk_sha256=item.chunk_sha256, quote=card.text, chunk_start=start, chunk_end=start + len(card.text))
                if not self._documents.verify_citation(citation):
                    return "candidate_unavailable", {}
                citations.append(citation)
            return "answered", dict(citations=tuple(citations), answer="\n".join(f'"{c.quote}" [{i}]' for i, c in enumerate(citations, 1)))
        if self._evidence is None or any(c.kind != "fact" for c in chosen):
            return "invalid_plan", {}
        facts = [self._evidence.lookup(c.fact_query) for c in chosen]
        if not all(self._evidence.verify_fact(f) for f in facts):
            return "unresolved_fact", {}
        if plan.action == "lookup":
            fact = facts[0]
            return ("answered", dict(fact=fact, answer=f"{fact.value} {fact.unit}")) if fact.status == "agreed" else ("unresolved_fact", dict(fact=fact))
        calculation = self._evidence.calculate(plan.operation, facts)
        if not self._evidence.verify_calculation(calculation):
            return "tool_abstained", {}
        return (("answered", dict(calculation=calculation, answer=f"{calculation.value} {calculation.unit}"))
                if calculation.status == "calculated" else ("tool_abstained", dict(calculation=calculation)))

    def _result(self, question, attempts, outcome, data):
        costs = [a.usage.provider_cost_usd for a in attempts]
        return _seal(ReasoningResult, dict(agent_fingerprint=self._fingerprint,
            query_sha256=hashlib.sha256(question.encode()).hexdigest(), status="answered" if outcome == "answered" else "abstained",
            reason=outcome, answer=data.get("answer", _ABSTAIN), attempts=tuple(attempts),
            planner_calls=len(attempts), provider_cost_usd=math.fsum(costs) if all(c is not None for c in costs) else None,
            **{k: v for k, v in data.items() if k != "answer"}))

    def run(self, question: str) -> ReasoningResult:
        question, attempts, previous = _normalize_question(question), [], None
        for number in range(1, self._config.max_planner_calls + 1):
            if self._backend() != self._planner_name:
                raise SDKError("planner identity changed")
            # The numeric catalog itself is reviewed data. Do not send stale labels externally.
            if self._evidence is not None:
                try:
                    current = self._evidence.query_catalog()
                    if {fingerprint(q.model_dump(mode="json")) for q in current} != set(self._queries):
                        raise SDKError("catalog changed")
                except SDKError:
                    if attempts:
                        raise SDKError("reasoning source authority changed during recovery") from None
                    return self._result(question, attempts, "source_not_authorized", {})
            top_k = self._config.retrieval_top_k if number == 1 else self._config.recovery_top_k
            text, facts = self._documents.search(question, top_k=top_k), self._facts.search(question, top_k=top_k)
            cards = self._cards(question, text, facts)
            if not cards:
                return self._result(question, attempts, attempts[-1].outcome if attempts else "no_candidates", {})
            request = PlannerRequest(question=question, candidates=cards, previous_failure=previous,
                                     attempts_remaining=self._config.max_planner_calls - number)
            plan, usage, data = None, PlannerUsage(), {}
            try:
                if self._backend() != self._planner_name:
                    raise SDKError("planner identity changed before execution")
                output = self._planner.plan(request)
                output = PlannerOutput.model_validate(output.model_dump())
                usage = output.usage
            except Exception:
                outcome = "planner_failed"
            else:
                try:
                    plan = ReasoningPlan.model_validate_json(output.plan_json)
                    outcome, data = self._execute(plan, cards, text)
                except (ValueError, KeyError, TypeError, SDKError):
                    outcome = "invalid_plan" if plan is None else "unresolved_fact"
            attempts.append(ReasoningAttempt(number=number, request=request, text_retrieval=text,
                fact_retrieval=facts, plan=plan, outcome=outcome, usage=usage))
            if outcome not in {"invalid_plan", "candidate_unavailable"} or number == self._config.max_planner_calls:
                result = self._result(question, attempts, outcome, data)
                if not self.verify_result(result, question=question):
                    raise SDKError("reasoning evidence changed during execution")
                return result
            previous = outcome
        raise AssertionError("bounded reasoning loop exhausted without a result")

    def verify_result(self, result: ReasoningResult, *, question: str) -> bool:
        """Verify provenance/execution without planner calls; not semantic answer correctness."""
        try:
            question = _normalize_question(question)
            result = ReasoningResult.model_validate(result.model_dump())
            if result.agent_fingerprint != self._fingerprint or result.planner_calls != len(result.attempts):
                return False
            previous, data, outcome = None, {}, result.reason
            for number, attempt in enumerate(result.attempts, 1):
                if (attempt.number != number or attempt.request.question != question
                        or attempt.request.previous_failure != previous
                        or attempt.request.attempts_remaining != self._config.max_planner_calls - number
                        or not self._documents.verify_search(attempt.text_retrieval, question=question)
                        or not self._facts.verify_search(attempt.fact_retrieval, question=question)
                        or attempt.request.candidates != self._cards(question, attempt.text_retrieval, attempt.fact_retrieval)):
                    return False
                outcome, data = (attempt.outcome, {}) if attempt.plan is None else self._execute(attempt.plan, attempt.request.candidates, attempt.text_retrieval)
                if outcome != attempt.outcome or (attempt.plan is None and outcome not in {"planner_failed", "invalid_plan"}):
                    return False
                if number < len(result.attempts) and outcome not in {"invalid_plan", "candidate_unavailable"}:
                    return False
                previous = outcome
            if not result.attempts and result.reason not in {"no_candidates", "source_not_authorized"}:
                return False
            return result == self._result(question, list(result.attempts), outcome, data)
        except (ValueError, KeyError, TypeError, AttributeError, SDKError):
            return False
