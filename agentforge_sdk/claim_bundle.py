"""Portable numeric replay over untrusted extraction snapshots, not source authentication."""

import json
from collections.abc import Sequence
from typing import Literal

from pydantic import Field, model_validator

from agentforge_sdk.claim_audit import NumericClaim, NumericClaimCheck, check_numeric_claim
from agentforge_sdk.evidence_review import ReviewedRAG
from agentforge_sdk.models import Contract, SDKError, fingerprint
from agentforge_sdk.multimodal import MultimodalRAG
from agentforge_sdk.reconciliation import (
    EvidenceSet,
    EvidenceSource,
    FactQuery,
    ReconciliationConfig,
)
from agentforge_sdk.spatial import MultimodalDocument

MAX_BUNDLE_BYTES = 3_000_000


class ClaimBundleError(SDKError):
    """Content-free rejection; no submitted evidence is included in the error."""


class ClaimReplayRecipe(Contract):
    claim: NumericClaim
    queries: tuple[FactQuery, ...] = Field(min_length=1, max_length=8, repr=False)
    operation: Literal["sum", "difference", "ratio", "percent_change"] | None = None

    @model_validator(mode="after")
    def valid_recipe(self):
        if len({fingerprint(q.model_dump(mode="json")) for q in self.queries}) != len(self.queries):
            raise ValueError("replay queries must be unique")
        if self.operation is None and len(self.queries) != 1:
            raise ValueError("a direct fact recipe requires one query")
        if self.operation not in {None, "sum"} and len(self.queries) != 2:
            raise ValueError("this calculation requires two ordered queries")
        return self


class ClaimReplayEntry(Contract):
    recipe: ClaimReplayRecipe
    check: NumericClaimCheck = Field(repr=False)

    @model_validator(mode="after")
    def same_assertion(self):
        if self.recipe.claim != self.check.claim:
            raise ValueError("recipe and check assertions must match")
        return self


class ClaimBundleSource(Contract):
    source_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,119}$")
    scope: str = Field(min_length=1, max_length=200)
    document: MultimodalDocument = Field(repr=False)


class ClaimReplayBundle(Contract):
    protocol: Literal["agentforge-claim-replay-bundle-v1"] = "agentforge-claim-replay-bundle-v1"
    config: ReconciliationConfig
    sources: tuple[ClaimBundleSource, ...] = Field(min_length=1, max_length=5, repr=False)
    collection_fingerprint: str = Field(pattern=r"^[a-f0-9]{64}$")
    entries: tuple[ClaimReplayEntry, ...] = Field(min_length=1, max_length=10, repr=False)
    fingerprint: str = Field(pattern=r"^[a-f0-9]{64}$")

    @model_validator(mode="after")
    def bounded_snapshot(self):
        ids = [s.source_id for s in self.sources]
        if ids != sorted(set(ids)):
            raise ValueError("bundle sources require unique canonical order")
        if len({e.recipe.claim.claim_id for e in self.entries}) != len(self.entries):
            raise ValueError("bundle claim identifiers must be unique")
        regions = [r for s in self.sources for p in s.document.pages for r in p.regions]
        if len(regions) > 2000 or sum(len(r.text) for r in regions) > 500_000:
            raise ValueError("bundle extraction budget exceeded")
        if any(p.method == "reviewed" for s in self.sources for p in s.document.pages):
            raise ValueError("live reviewed authority cannot be represented by a plain snapshot")
        if self.fingerprint != fingerprint(self.model_dump(mode="json", exclude={"fingerprint"})):
            raise ValueError("bundle binding is invalid")
        return self


class ClaimBundleVerification(Contract):
    replay_verified: Literal[True] = True
    bundle_fingerprint: str
    fingerprint_pinned: bool
    total: int
    supported: int
    refuted: int
    unresolved: int
    all_numeric_assertions_supported: bool
    # Even a pinned snapshot is not proof of extraction accuracy or current authority.
    source_files_authenticated: Literal[False] = False
    extraction_reexecuted: Literal[False] = False
    release_authorized: Literal[False] = False
    model_calls: Literal[0] = 0


def load_claim_replay_bundle(raw: bytes | str) -> ClaimReplayBundle:
    """Parse bounded plain JSON. No archives, source files, imports or URLs are resolved."""

    def unique_keys(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate JSON key")
            result[key] = value
        return result

    try:
        if not isinstance(raw, (bytes, str)):
            raise ValueError("expected JSON bytes or text")
        data = raw.encode("utf-8") if isinstance(raw, str) else raw
        if not 0 < len(data) <= MAX_BUNDLE_BYTES:
            raise ValueError("bundle byte budget exceeded")
        # Pydantic's JSON mode accepts JSON arrays for strict tuple contracts.
        # Check duplicate keys separately so no parser-dependent last-key-wins is allowed.
        json.loads(data.decode("utf-8"), object_pairs_hook=unique_keys)
        return ClaimReplayBundle.model_validate_json(data)
    except (ValueError, TypeError, RecursionError):
        raise ClaimBundleError("claim replay bundle is invalid or exceeds its budget") from None


def _collection(bundle):
    return EvidenceSet(
        [EvidenceSource(s.source_id, s.scope, MultimodalRAG(s.document)) for s in bundle.sources],
        config=bundle.config,
    )


def verify_claim_replay_bundle(
    bundle: ClaimReplayBundle, *, expected_fingerprint: str | None = None
) -> ClaimBundleVerification:
    """Recompute every recipe and original check; success is not numerical support.

    A separately obtained fingerprint can pin an expected snapshot. Unkeyed
    digests cannot establish publisher identity, accurate extraction, current
    review authority or truth against an author who rewrites the whole bundle.
    """
    try:
        bundle = load_claim_replay_bundle(bundle.model_dump_json())
        if expected_fingerprint is not None and expected_fingerprint != bundle.fingerprint:
            raise ValueError("unexpected bundle fingerprint")
        evidence = _collection(bundle)
        if evidence.fingerprint != bundle.collection_fingerprint:
            raise ValueError("collection mismatch")
        facts = {}
        counts = dict(supported=0, refuted=0, unresolved=0)
        for entry in bundle.entries:
            inputs = []
            for query in entry.recipe.queries:
                key = fingerprint(query.model_dump(mode="json"))
                if key not in facts:
                    facts[key] = evidence.lookup(query)
                inputs.append(facts[key])
            receipt = (
                evidence.calculate(entry.recipe.operation, inputs)
                if entry.recipe.operation is not None
                else inputs[0]
            )
            fresh = check_numeric_claim(evidence, entry.recipe.claim, receipt, facts=inputs)
            if fresh != entry.check or fresh.reason == "evidence_unverifiable":
                raise ValueError("check cannot be replayed")
            counts[fresh.status] += 1
        return ClaimBundleVerification(
            bundle_fingerprint=bundle.fingerprint,
            fingerprint_pinned=expected_fingerprint is not None,
            total=len(bundle.entries),
            **counts,
            all_numeric_assertions_supported=counts["supported"] == len(bundle.entries),
        )
    except (ValueError, TypeError, AttributeError, KeyError, RecursionError, SDKError):
        raise ClaimBundleError("claim replay bundle could not be verified") from None


def build_claim_replay_bundle(
    evidence: EvidenceSet, entries: Sequence[ClaimReplayEntry]
) -> ClaimReplayBundle:
    """Freeze a plain extracted evidence set and all explicitly selected numeric checks.

    Manifests include all extraction text, not only cited excerpts. ReviewedRAG
    is refused rather than dropping HMAC/expiry/revocation authority on export.
    No files, images, credentials, owner notes or executable tools are included.
    """
    try:
        if not isinstance(entries, (list, tuple)) or not 1 <= len(entries) <= 10:
            raise ValueError("bounded typed entries required")
        if any(isinstance(s.reader, ReviewedRAG) for s in evidence._sources):
            raise ValueError("live review authority cannot be exported")
        payload = dict(
            protocol="agentforge-claim-replay-bundle-v1",
            config=evidence._config.model_dump(mode="json"),
            sources=[
                dict(
                    source_id=s.source_id,
                    scope=s.scope,
                    document=evidence._documents[s.source_id].model_dump(mode="json"),
                )
                for s in evidence._sources
            ],
            collection_fingerprint=evidence.fingerprint,
            entries=[e.model_dump(mode="json") for e in entries],
        )
        bundle = ClaimReplayBundle.model_validate_json(
            json.dumps({**payload, "fingerprint": fingerprint(payload)})
        )
        verify_claim_replay_bundle(bundle)
        return bundle
    except (ValueError, TypeError, AttributeError, KeyError, RecursionError, SDKError):
        raise ClaimBundleError("claim replay bundle cannot be created from this evidence") from None
