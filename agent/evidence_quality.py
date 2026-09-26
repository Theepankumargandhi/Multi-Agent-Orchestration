"""Pre-synthesis evidence quality, independence, freshness, and conflict controls."""

from __future__ import annotations

import hashlib
import hmac
import json
import re
from datetime import date, datetime, timedelta, timezone
from typing import Literal
from urllib.parse import urlparse

from pydantic import BaseModel, Field

from agent.grounding import EvidenceItem

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+|\n+")
_DATE_RE = re.compile(r"(?<!\d)(20\d{2})[-/](0[1-9]|1[0-2])[-/](0[1-9]|[12]\d|3[01])(?!\d)")
_NUMBER_RE = re.compile(r"(?<![a-z])[-+]?\d+(?:\.\d+)?%?", re.IGNORECASE)
_NEGATION_RE = re.compile(r"\b(?:not|no|never|neither|cannot|can't|isn't|wasn't|without)\b", re.I)
_INJECTION_RE = re.compile(
    r"(?i)(?:ignore\s+(?:all\s+)?(?:previous|prior|system)\s+instructions|"
    r"reveal\s+(?:the\s+)?system\s+prompt|developer\s+message|"
    r"execute\s+(?:this\s+)?(?:command|tool)|call\s+(?:the\s+)?tool|"
    r"you\s+are\s+now\s+(?:the|an?)\s+)"
)
_STOP = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has",
    "have", "in", "is", "it", "of", "on", "or", "that", "the", "this", "to",
    "was", "were", "will", "with", "not", "no", "never", "cannot", "without",
}


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()


def _fingerprint(value: object, key: bytes | None = None) -> str:
    payload = _canonical(value)
    return (
        hmac.new(key, payload, hashlib.sha256).hexdigest()
        if key
        else hashlib.sha256(payload).hexdigest()
    )


def _tokens(value: str) -> set[str]:
    return {item for item in _TOKEN_RE.findall(value.casefold()) if item not in _STOP}


def _jaccard(left: set[str], right: set[str]) -> float:
    return len(left & right) / len(left | right) if left or right else 1.0


def _domain(item: EvidenceItem) -> str:
    for url in item.urls:
        host = (urlparse(url).hostname or "").casefold().removeprefix("www.")
        if host:
            return host
    return f"local:{item.source_type}"


def _published_at(text: str) -> date | None:
    match = _DATE_RE.search(text)
    if not match:
        return None
    try:
        return date(int(match.group(1)), int(match.group(2)), int(match.group(3)))
    except ValueError:
        return None


class EvidenceQualityPolicy(BaseModel):
    version: str = "evidence-quality-v1"
    duplicate_similarity: float = Field(default=0.82, ge=0.5, le=1)
    min_web_independent_sources: int = Field(default=2, ge=1, le=10)
    min_local_independent_sources: int = Field(default=1, ge=1, le=10)
    max_conflicts: int = Field(default=0, ge=0, le=20)
    min_fresh_coverage: float = Field(default=0.5, ge=0, le=1)
    authority_domains: set[str] = Field(
        default_factory=lambda: {"openai.com", "github.com", "python.org"}
    )


class EvidenceConflict(BaseModel):
    conflict_type: Literal["numeric", "negation"]
    left_evidence_id: str
    right_evidence_id: str
    proposition_fingerprint: str
    left_value: str
    right_value: str


class EvidenceQualityReport(BaseModel):
    schema_version: str = "1.0"
    policy_version: str
    route: str
    action: Literal["pass", "degraded", "abstain", "not_required"]
    total_evidence: int
    usable_evidence: int
    quarantined_evidence_ids: list[str]
    duplicate_clusters: list[list[str]]
    independent_source_count: int
    source_domains: list[str]
    dated_evidence_count: int
    fresh_evidence_count: int
    fresh_coverage: float
    authoritative_source_count: int
    conflicts: list[EvidenceConflict]
    reason_codes: list[str]
    usable_evidence_fingerprints: list[str]
    report_fingerprint: str = ""


class EvidenceAdjudication(BaseModel):
    report: EvidenceQualityReport
    usable_evidence: list[EvidenceItem]


def _authority_score(domain: str, source_type: str, trusted: set[str]) -> int:
    if source_type in {"math", "knowledge_graph", "rag"}:
        return 3
    if domain.endswith(".gov") or domain.endswith(".edu"):
        return 3
    if any(domain == item or domain.endswith(f".{item}") for item in trusted):
        return 3
    return 1


def _duplicate_clusters(
    items: list[EvidenceItem], threshold: float
) -> list[list[EvidenceItem]]:
    clusters: list[list[EvidenceItem]] = []
    for item in items:
        item_tokens = _tokens(item.text)
        for cluster in clusters:
            if any(_jaccard(item_tokens, _tokens(other.text)) >= threshold for other in cluster):
                cluster.append(item)
                break
        else:
            clusters.append([item])
    return clusters


def _statement_records(items: list[EvidenceItem]) -> list[tuple[EvidenceItem, str, set[str]]]:
    records = []
    for item in items:
        for statement in _SENTENCE_RE.split(item.text):
            clean = statement.strip()
            tokens = _tokens(_NUMBER_RE.sub("", clean))
            if len(tokens) >= 3:
                records.append((item, clean, tokens))
    return records[:200]


def _conflicts(items: list[EvidenceItem]) -> list[EvidenceConflict]:
    records = _statement_records(items)
    conflicts: list[EvidenceConflict] = []
    seen: set[tuple[str, str, str]] = set()
    for index, (left_item, left, left_tokens) in enumerate(records):
        for right_item, right, right_tokens in records[index + 1 :]:
            if left_item.evidence_id == right_item.evidence_id:
                continue
            if _jaccard(left_tokens, right_tokens) < 0.6:
                continue
            left_numbers = set(_NUMBER_RE.findall(left))
            right_numbers = set(_NUMBER_RE.findall(right))
            left_negated = bool(_NEGATION_RE.search(left))
            right_negated = bool(_NEGATION_RE.search(right))
            conflict_type = ""
            left_value = right_value = ""
            if left_numbers and right_numbers and left_numbers != right_numbers:
                conflict_type = "numeric"
                left_value, right_value = ",".join(sorted(left_numbers)), ",".join(sorted(right_numbers))
            elif left_negated != right_negated:
                conflict_type = "negation"
                left_value, right_value = str(left_negated), str(right_negated)
            if not conflict_type:
                continue
            pair = tuple(sorted((left_item.evidence_id, right_item.evidence_id)))
            if pair[0] != left_item.evidence_id:
                left_value, right_value = right_value, left_value
            key = (pair[0], pair[1], conflict_type)
            if key in seen:
                continue
            seen.add(key)
            conflicts.append(
                EvidenceConflict(
                    conflict_type=conflict_type,
                    left_evidence_id=pair[0],
                    right_evidence_id=pair[1],
                    proposition_fingerprint=_fingerprint(sorted(left_tokens & right_tokens)),
                    left_value=left_value,
                    right_value=right_value,
                )
            )
    return conflicts[:50]


def adjudicate_evidence(
    evidence: list[EvidenceItem],
    *,
    route: str,
    recency_days: int = 0,
    as_of: datetime | None = None,
    policy: EvidenceQualityPolicy | None = None,
    integrity_key: bytes | None = None,
) -> EvidenceAdjudication:
    policy = policy or EvidenceQualityPolicy()
    normalized_route = route.casefold()
    if normalized_route not in {"web", "hybrid", "rag", "kg", "math"}:
        report = EvidenceQualityReport(
            policy_version=policy.version,
            route=normalized_route,
            action="not_required",
            total_evidence=len(evidence),
            usable_evidence=len(evidence),
            quarantined_evidence_ids=[],
            duplicate_clusters=[],
            independent_source_count=len(evidence),
            source_domains=sorted({_domain(item) for item in evidence}),
            dated_evidence_count=0,
            fresh_evidence_count=0,
            fresh_coverage=1,
            authoritative_source_count=0,
            conflicts=[],
            reason_codes=["route_not_evidence_bound"],
            usable_evidence_fingerprints=[item.fingerprint for item in evidence],
        )
        report.report_fingerprint = _fingerprint(
            report.model_dump(mode="json", exclude={"report_fingerprint"}), integrity_key
        )
        return EvidenceAdjudication(report=report, usable_evidence=evidence)

    quarantined = [item for item in evidence if _INJECTION_RE.search(item.text)]
    clean = [item for item in evidence if item not in quarantined]
    clusters = _duplicate_clusters(clean, policy.duplicate_similarity)
    now = (as_of or datetime.now(timezone.utc)).date()

    def rank(item: EvidenceItem) -> tuple[int, int, str]:
        domain = _domain(item)
        published = _published_at(item.text)
        freshness = published.toordinal() if published else 0
        return (_authority_score(domain, item.source_type, policy.authority_domains), freshness, item.evidence_id)

    usable = [max(cluster, key=rank) for cluster in clusters]
    domains = sorted({_domain(item) for item in usable})
    independent_count = min(len(domains), len(usable))
    dates = [_published_at(item.text) for item in usable]
    dated = [item for item in dates if item is not None]
    cutoff = now - timedelta(days=recency_days) if recency_days > 0 else None
    fresh = [item for item in dated if cutoff is None or item >= cutoff]
    fresh_coverage = len(fresh) / len(usable) if usable else 0.0
    authoritative = sum(
        _authority_score(_domain(item), item.source_type, policy.authority_domains) >= 3
        for item in usable
    )
    conflicts = _conflicts(usable)
    minimum_sources = (
        policy.min_web_independent_sources
        if normalized_route in {"web", "hybrid"}
        else policy.min_local_independent_sources
    )
    reasons: list[str] = []
    if quarantined:
        reasons.append("prompt_injection_quarantined")
    if any(len(cluster) > 1 for cluster in clusters):
        reasons.append("near_duplicate_evidence_collapsed")
    if independent_count < minimum_sources:
        reasons.append("insufficient_independent_sources")
    if recency_days > 0 and fresh_coverage < policy.min_fresh_coverage:
        reasons.append("insufficient_fresh_evidence")
    if len(conflicts) > policy.max_conflicts:
        reasons.append("unresolved_evidence_conflict")
    hard_failures = {
        "insufficient_independent_sources",
        "insufficient_fresh_evidence",
        "unresolved_evidence_conflict",
    }
    if not usable or hard_failures.intersection(reasons):
        action = "abstain"
    elif reasons:
        action = "degraded"
    else:
        action = "pass"
    report = EvidenceQualityReport(
        policy_version=policy.version,
        route=normalized_route,
        action=action,
        total_evidence=len(evidence),
        usable_evidence=len(usable),
        quarantined_evidence_ids=sorted(item.evidence_id for item in quarantined),
        duplicate_clusters=[
            sorted(item.evidence_id for item in cluster)
            for cluster in clusters
            if len(cluster) > 1
        ],
        independent_source_count=independent_count,
        source_domains=domains,
        dated_evidence_count=len(dated),
        fresh_evidence_count=len(fresh),
        fresh_coverage=round(fresh_coverage, 6),
        authoritative_source_count=authoritative,
        conflicts=conflicts,
        reason_codes=reasons or ["evidence_quality_passed"],
        usable_evidence_fingerprints=[item.fingerprint for item in usable],
    )
    report.report_fingerprint = _fingerprint(
        report.model_dump(mode="json", exclude={"report_fingerprint"}), integrity_key
    )
    return EvidenceAdjudication(report=report, usable_evidence=usable)


def verify_report(report: EvidenceQualityReport, integrity_key: bytes | None = None) -> bool:
    expected = _fingerprint(
        report.model_dump(mode="json", exclude={"report_fingerprint"}), integrity_key
    )
    return hmac.compare_digest(report.report_fingerprint, expected)
