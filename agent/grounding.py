"""Claim-level grounding verification and fail-closed answer repair."""

from __future__ import annotations

import hashlib
import hmac
import json
import math
import re
from collections import Counter
from typing import Literal, Mapping

from pydantic import BaseModel, Field

GroundingAction = Literal["pass", "repair", "abstain", "not_required"]

_URL_RE = re.compile(r"https?://[^\s)\]>]+")
_MARKDOWN_LINK_RE = re.compile(r"\[[^\]]+\]\((https?://[^)]+)\)")
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+|\n+")
_TOKEN_RE = re.compile(r"[a-z0-9]+")
_STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has", "have",
    "in", "is", "it", "of", "on", "or", "that", "the", "this", "to", "was", "were",
    "will", "with", "you", "your",
}
_HIGH_RISK_RE = re.compile(
    r"(?i)(?:\b\d+(?:\.\d+)?%?\b|\$\s*\d|\b(?:today|currently|latest|always|never|"
    r"guaranteed|approved|certified|safe|vulnerable|legal|medical|diagnos|dose|revenue)\b)"
)
_NON_CLAIM_PREFIXES = (
    "sources:",
    "source:",
    "please ",
    "try ",
    "i could not",
    "i couldn't",
    "i cannot",
    "i don't have enough verified evidence",
)
# Includes words that negate by meaning: "insufficiently independent" ~ "not independent".
_NEGATION_CUES = {
    "not", "no", "never", "neither", "nor", "none", "cannot", "without",
    "insufficient", "insufficiently", "lack", "lacks", "lacking", "unable", "missing", "absent",
}
# Words skipped when finding what a negation governs, as in "is not even available".
_NEGATION_FILLERS = _STOP_WORDS | {
    "be", "been", "being", "do", "does", "did", "had", "can", "could", "would", "should",
    "may", "might", "must", "ever", "even", "yet", "any", "also", "always", "only",
}
_NUMBER_RE = re.compile(r"\d+(?:[.,]\d+)*")
# Citation markers and retrieval metadata carry page numbers and scores, not facts:
# "\u3010source=README.md, page=22\u3011" and "source=x, page=3, fused=0.03, bm25=6.6:".
_CITATION_MARKER_RE = re.compile(r"\u3010[^\u3011]*\u3011|\b(?:source|page|fused|bm25|vector|relevance|distance)=[^\s,:]+[,:]?")
# Clause boundaries; "1,000" and "10:30" stay intact.
_CLAUSE_RE = re.compile(r"[,;]\s+|:\s+|\s+(?:but|while|whereas|although|though|however)\s+", re.I)
# A claim that uses one side while the matching evidence uses only the other
# side states the opposite of that evidence.
_OPPOSITE_FAMILIES = (
    ({"rise", "rises", "rose", "risen", "rising", "increase", "increases", "increased", "grew", "higher"},
     {"fall", "falls", "fell", "fallen", "falling", "decrease", "decreases", "decreased", "declined",
      "dropped", "lower"}),
    ({"pass", "passes", "passed", "succeeded", "succeeds"}, {"fail", "fails", "failed"}),
    ({"enable", "enables", "enabled"}, {"disable", "disables", "disabled"}),
    ({"available"}, {"unavailable"}),
    ({"supported"}, {"unsupported"}),
    ({"allowed"}, {"blocked", "forbidden", "disallowed"}),
    ({"valid"}, {"invalid"}),
    ({"safe"}, {"unsafe"}),
    ({"secure"}, {"insecure"}),
    ({"true"}, {"false"}),
    ({"accepted", "approved"}, {"rejected", "denied"}),
)
_POLARITY_WORDS = {word for left, right in _OPPOSITE_FAMILIES for word in left | right} | {
    "not", "no", "never", "neither", "nor", "none", "cannot", "without",
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


def _tokens(value: str) -> list[str]:
    return [item for item in _TOKEN_RE.findall(value.casefold()) if item not in _STOP_WORDS]


def _features(value: str) -> dict[int, float]:
    tokens = _tokens(value)
    features = tokens + [f"{left}:{right}" for left, right in zip(tokens, tokens[1:])]
    counts: Counter[int] = Counter()
    for feature in features:
        index = int.from_bytes(hashlib.sha256(feature.encode()).digest()[:4], "big") % 512
        counts[index] += 1
    norm = math.sqrt(sum(count * count for count in counts.values())) or 1
    return {index: count / norm for index, count in counts.items()}


def _cosine(left: dict[int, float], right: dict[int, float]) -> float:
    return sum(value * right.get(index, 0) for index, value in left.items())


def _content_tokens(value: str) -> set[str]:
    return {item for item in _tokens(value) if not item.isdigit() and item not in _POLARITY_WORDS}


def _numbers(value: str) -> set[str]:
    return {item.replace(",", "") for item in _NUMBER_RE.findall(_URL_RE.sub(" ", value))}


def _clauses(text: str) -> list[str]:
    return [item for item in _CLAUSE_RE.split(text) if item and item.strip()]


def _stem(word: str) -> str:
    # Enough to match "use"/"uses"/"used" and "store"/"stored"/"storing".
    if len(word) > 3:
        for suffix in ("ing", "ed", "es", "s"):
            if word.endswith(suffix) and not word.endswith("ss") and len(word) - len(suffix) >= 2:
                word = word[: -len(suffix)]
                break
    return word[:-1] if len(word) >= 3 and word.endswith("e") else word


def _negation(text: str) -> tuple[bool, set[str]]:
    """Return whether text contains a negation and the words its negations govern."""
    normalized = re.sub(r"n't\b", " not", text.casefold().replace(chr(0x2019), "'"))
    words = _TOKEN_RE.findall(normalized)
    negated, governed = False, set()
    for index, word in enumerate(words):
        if word not in _NEGATION_CUES:
            continue
        negated = True
        for following in words[index + 1:index + 4]:
            if following not in _NEGATION_FILLERS and following not in _NEGATION_CUES:
                governed.add(following)
                break
    return negated, governed


def _best_match(content: set[str], candidates: list[str]) -> tuple[str, float]:
    return max(
        ((item, len(content & _content_tokens(item)) / len(content)) for item in candidates),
        key=lambda pair: pair[1],
        default=("", 0.0),
    )


def _part_contradiction(part: str, evidence_sentences: list[str]) -> str:
    content = _content_tokens(part)
    # Too short to judge, e.g. a markdown fragment such as "`code_agent/`".
    if not content or len(_tokens(part)) < 3:
        return ""
    sentence, overlap = _best_match(content, evidence_sentences)
    # Only evidence about the same subject can contradict this part of the claim.
    if overlap < 0.5:
        return ""
    # Numbers are read across the whole sentence ("In 2024, revenue rose by 5%").
    part_numbers, sentence_numbers = _numbers(part), _numbers(sentence)
    if part_numbers and sentence_numbers and not part_numbers <= sentence_numbers:
        return "numeric"
    # A negation contradicts only when the other side is not negated and contains the
    # word it governs: "not available" against "available", but "runs tests, not
    # shell commands" still supports "runs tests".
    part_negated, part_governed = _negation(part)
    sentence_negated, sentence_governed = _negation(sentence)
    if part_negated != sentence_negated:
        governed, plain = (part_governed, sentence) if part_negated else (sentence_governed, part)
        if {_stem(word) for word in governed} & {_stem(word) for word in _TOKEN_RE.findall(plain.casefold())}:
            return "negation"
    part_words, sentence_words = set(_tokens(part)), set(_tokens(sentence))
    for left, right in _OPPOSITE_FAMILIES:
        # A part that uses both sides ("failed before, passed after") is not one-sided.
        if part_words & left and part_words & right:
            continue
        for mine, other in ((left, right), (right, left)):
            if part_words & mine and sentence_words & other and not sentence_words & mine:
                return "opposite"
    return ""


def _contradiction(claim: str, evidence_text: str) -> str:
    """Name how evidence about the same subject states the opposite of the claim, if it does.

    Word overlap alone scores "X is not available" as strong support for
    "X is available"; this checks numbers, negation and opposite words for each
    clause of the claim.
    """
    evidence_sentences = [item for item in _SENTENCE_RE.split(_CITATION_MARKER_RE.sub(" ", evidence_text)) if item.strip()]
    for part in _clauses(_CITATION_MARKER_RE.sub(" ", _MARKDOWN_LINK_RE.sub(" ", claim))):
        found = _part_contradiction(part, evidence_sentences)
        if found:
            return found
    return ""


class GroundingPolicy(BaseModel):
    version: str = "grounding-policy-v3"
    required_routes: set[str] = Field(
        default_factory=lambda: {"web", "hybrid", "rag", "kg", "math"}
    )
    min_claim_score: float = Field(default=0.22, ge=0, le=1)
    min_high_risk_claim_score: float = Field(default=0.35, ge=0, le=1)
    min_coverage: float = Field(default=0.8, ge=0, le=1)
    fail_closed: bool = True


class EvidenceItem(BaseModel):
    evidence_id: str
    source_type: Literal["web", "rag", "knowledge_graph", "math"]
    text: str
    urls: list[str] = Field(default_factory=list)
    fingerprint: str


class ClaimAssessment(BaseModel):
    claim_index: int
    claim: str
    claim_fingerprint: str
    best_evidence_id: str = ""
    support_score: float = 0
    supported: bool = False
    high_risk: bool = False
    contradiction: Literal["", "numeric", "negation", "opposite"] = ""
    citation_urls: list[str] = Field(default_factory=list)
    invalid_citation_urls: list[str] = Field(default_factory=list)


class GroundingReport(BaseModel):
    schema_version: str = "1.0"
    policy_version: str
    route: str
    verification_required: bool
    passed: bool
    action: GroundingAction
    confidence: float
    claim_coverage: float
    citation_precision: float
    total_claims: int
    supported_claims: int
    unsupported_high_risk_claims: int
    contradicted_claims: int = 0
    evidence_count: int
    evidence_fingerprints: list[str]
    claims: list[ClaimAssessment]
    reason: str
    report_fingerprint: str = ""


def _split_evidence(source_type: str, value: str) -> list[EvidenceItem]:
    text = (value or "").strip()
    if not text or "not required for" in text.casefold() or "retrieval failed:" in text.casefold():
        return []
    chunks = [item.strip() for item in re.split(r"\n\s*\n|(?=\n-\s)", text) if item.strip()]
    if not chunks:
        chunks = [text]
    result: list[EvidenceItem] = []
    normalized_type = "knowledge_graph" if source_type == "kg" else source_type
    for index, chunk in enumerate(chunks[:50]):
        bounded = chunk[:4000]
        fingerprint = _fingerprint({"source_type": normalized_type, "text": bounded})
        result.append(
            EvidenceItem(
                evidence_id=f"ev_{normalized_type}_{index}_{fingerprint[:10]}",
                source_type=normalized_type,
                text=bounded,
                urls=sorted(set(_URL_RE.findall(bounded))),
                fingerprint=fingerprint,
            )
        )
    return result


def evidence_from_state(state: Mapping[str, object]) -> list[EvidenceItem]:
    if state.get("evidence_quality_report"):
        raw = state.get("adjudicated_evidence") or []
        if isinstance(raw, list):
            return [EvidenceItem.model_validate(item) for item in raw]
    evidence: list[EvidenceItem] = []
    for source_type, field in (
        ("web", "web_notes"),
        ("rag", "rag_notes"),
        ("kg", "kg_notes"),
        ("math", "math_result"),
    ):
        evidence.extend(_split_evidence(source_type, str(state.get(field) or "")))
    return evidence


def extract_claims(answer: str) -> list[str]:
    clean = re.sub(r"```.*?```", "", answer or "", flags=re.DOTALL)
    claims: list[str] = []
    for fragment in _SENTENCE_RE.split(clean):
        claim = re.sub(r"^\s*(?:[-*]|\d+[.)])\s*", "", fragment).strip()
        if not claim or len(claim) < 12 or claim.endswith("?"):
            continue
        if claim.casefold().startswith(_NON_CLAIM_PREFIXES):
            continue
        if not re.search(r"[A-Za-z]", claim):
            continue
        claims.append(claim[:1000])
    return claims[:50]


def _support_score(claim: str, evidence: EvidenceItem) -> float:
    claim_tokens = set(_tokens(_MARKDOWN_LINK_RE.sub("", claim)))
    evidence_tokens = set(_tokens(evidence.text))
    if not claim_tokens or not evidence_tokens:
        return 0
    lexical = len(claim_tokens & evidence_tokens) / len(claim_tokens)
    semantic = max(0.0, _cosine(_features(claim), _features(evidence.text)))
    return min(1.0, 0.7 * lexical + 0.3 * semantic)


def _assess_claim(
    index: int,
    claim: str,
    evidence: list[EvidenceItem],
    allowed_urls: set[str],
    policy: GroundingPolicy,
    integrity_key: bytes | None = None,
) -> ClaimAssessment:
    scored = sorted(
        ((item, _support_score(claim, item)) for item in evidence),
        key=lambda pair: (-pair[1], pair[0].evidence_id),
    )
    best_item, best_score = scored[0] if scored else (None, 0.0)
    claim_urls = _MARKDOWN_LINK_RE.findall(claim)
    claim_invalid = sorted(set(claim_urls) - allowed_urls)
    high_risk = bool(_HIGH_RISK_RE.search(_MARKDOWN_LINK_RE.sub("", claim)))
    threshold = policy.min_high_risk_claim_score if high_risk else policy.min_claim_score
    # Any qualifying evidence that states the opposite blocks the claim, even
    # if another item appears to support it.
    contradiction = next(
        (found for item, score in scored if score >= threshold
         for found in [_contradiction(claim, item.text)] if found),
        "",
    )
    supported = best_score >= threshold and not claim_invalid and not contradiction
    return ClaimAssessment(
        claim_index=index,
        claim=claim,
        claim_fingerprint=_fingerprint(claim, integrity_key),
        best_evidence_id=best_item.evidence_id if best_item else "",
        support_score=round(best_score, 6),
        supported=supported,
        high_risk=high_risk,
        contradiction=contradiction,
        citation_urls=claim_urls,
        invalid_citation_urls=claim_invalid,
    )


def take_complete_sentences(buffer: str) -> tuple[list[str], str]:
    """Split finished sentences off the front of a streaming buffer.

    Sentences end where extract_claims splits, so a fragment checked while streaming holds the
    same claims the full-answer verifier sees. Each piece keeps its trailing whitespace, and a
    fenced code block stays whole until it closes.
    """
    pieces: list[str] = []
    start = 0
    for match in _SENTENCE_RE.finditer(buffer):
        # Separator at the very end may still grow, or the next token may be a new list item.
        if match.end() == len(buffer):
            break
        if buffer[start:match.end()].count("```") % 2:
            continue
        pieces.append(buffer[start:match.end()])
        start = match.end()
    return pieces, buffer[start:]


def fragment_is_supported(
    fragment: str,
    evidence: list[EvidenceItem],
    policy: GroundingPolicy | None = None,
) -> bool:
    """Whether part of a draft may be shown before the full answer is verified.

    Stricter than the release policy: every claim in the fragment must be supported, free of
    contradiction and cite only retrieved URLs. Text with no factual claim passes.
    """
    policy = policy or GroundingPolicy()
    allowed_urls = {url for item in evidence for url in item.urls}
    return all(
        _assess_claim(index, claim, evidence, allowed_urls, policy).supported
        for index, claim in enumerate(extract_claims(fragment))
    )


def verify_grounding(
    *,
    route: str,
    answer: str,
    evidence: list[EvidenceItem],
    policy: GroundingPolicy | None = None,
    integrity_key: bytes | None = None,
) -> GroundingReport:
    policy = policy or GroundingPolicy()
    normalized_route = (route or "general").casefold()
    required = normalized_route in policy.required_routes
    claims = extract_claims(answer)
    if not required:
        report = GroundingReport(
            policy_version=policy.version,
            route=normalized_route,
            verification_required=False,
            passed=True,
            action="not_required",
            confidence=1.0,
            claim_coverage=1.0,
            citation_precision=1.0,
            total_claims=len(claims),
            supported_claims=len(claims),
            unsupported_high_risk_claims=0,
            evidence_count=len(evidence),
            evidence_fingerprints=[item.fingerprint for item in evidence],
            claims=[],
            reason="route_not_evidence_bound",
        )
        report.report_fingerprint = _fingerprint(
            report.model_dump(mode="json", exclude={"report_fingerprint"}), integrity_key
        )
        return report

    allowed_urls = {url for item in evidence for url in item.urls}
    cited_urls = _MARKDOWN_LINK_RE.findall(answer or "")
    invalid_urls = sorted(set(cited_urls) - allowed_urls)
    citation_precision = (
        sum(url in allowed_urls for url in cited_urls) / len(cited_urls) if cited_urls else 1.0
    )
    assessments = [
        _assess_claim(index, claim, evidence, allowed_urls, policy, integrity_key)
        for index, claim in enumerate(claims)
    ]

    total_weight = sum(2 if item.high_risk else 1 for item in assessments)
    supported_weight = sum(
        (2 if item.high_risk else 1) for item in assessments if item.supported
    )
    coverage = supported_weight / total_weight if total_weight else 1.0
    supported_count = sum(item.supported for item in assessments)
    unsupported_high_risk = sum(item.high_risk and not item.supported for item in assessments)
    contradicted = sum(bool(item.contradiction) for item in assessments)
    # A claim that states the opposite of the evidence is never averaged away by coverage.
    passed = coverage >= policy.min_coverage and not invalid_urls and not contradicted
    if passed:
        action: GroundingAction = "pass"
        reason = "claim_coverage_and_citations_passed"
    elif policy.fail_closed and (not evidence or coverage < 0.5):
        action = "abstain"
        reason = "insufficient_support_for_majority_claims"
    else:
        # Repair keeps only supported claims, so an unsupported high-risk or contradicted
        # claim is never released; the verified rest of the answer still is.
        action = "repair"
        reason = (
            "remove_contradicted_claims" if contradicted
            else "remove_unsupported_high_risk_claims" if unsupported_high_risk
            else "remove_unsupported_claims_or_invalid_citations"
        )
    confidence = max(0.0, min(1.0, coverage * citation_precision))
    report = GroundingReport(
        policy_version=policy.version,
        route=normalized_route,
        verification_required=True,
        passed=passed,
        action=action,
        confidence=round(confidence, 6),
        claim_coverage=round(coverage, 6),
        citation_precision=round(citation_precision, 6),
        total_claims=len(assessments),
        supported_claims=supported_count,
        unsupported_high_risk_claims=unsupported_high_risk,
        contradicted_claims=contradicted,
        evidence_count=len(evidence),
        evidence_fingerprints=[item.fingerprint for item in evidence],
        claims=assessments,
        reason=reason,
    )
    report.report_fingerprint = _fingerprint(
        report.model_dump(mode="json", exclude={"report_fingerprint"}), integrity_key
    )
    return report


def verify_report(report: GroundingReport, integrity_key: bytes | None = None) -> bool:
    expected = _fingerprint(
        report.model_dump(mode="json", exclude={"report_fingerprint"}), integrity_key
    )
    return hmac.compare_digest(report.report_fingerprint, expected)


def _keep_supported_fragments(answer: str, supported: set[str]) -> str:
    # Same fragments the stream released, so the repaired answer keeps the draft's layout.
    pieces, rest = take_complete_sentences(answer)
    kept = [
        piece for piece in [*pieces, rest]
        if all(claim in supported for claim in extract_claims(piece))
    ]
    return "".join(kept).strip()


def repair_answer(
    report: GroundingReport, evidence: list[EvidenceItem], answer: str = ""
) -> str:
    if report.action == "abstain":
        return (
            "I don’t have enough verified evidence to answer this reliably. "
            "I’d rather leave the claim unresolved than present unsupported information."
        )
    supported = [item.claim for item in report.claims if item.supported]
    if not supported:
        return (
            "I don’t have enough verified evidence to answer this reliably. "
            "Please provide another source or broaden the retrieval scope."
        )
    repaired = _keep_supported_fragments(answer, set(supported)) if answer else ""
    repaired = repaired or " ".join(supported)
    allowed_urls = sorted({url for item in evidence for url in item.urls})
    cited = [url for url in _MARKDOWN_LINK_RE.findall(repaired) if url in allowed_urls]
    if report.route in {"web", "hybrid"} and allowed_urls and not cited:
        repaired += "\n\nVerified sources: " + ", ".join(
            f"[source {index}]({url})" for index, url in enumerate(allowed_urls[:3], start=1)
        )
    return repaired + "\n\nI omitted claims that could not be verified against the retrieved evidence."
