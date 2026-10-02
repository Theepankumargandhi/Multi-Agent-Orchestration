"""Content-free Bradley–Terry preferences; never a replacement for release gates."""

from __future__ import annotations

import hashlib
import hmac
import json
import math
import random
import statistics
from collections import Counter
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from agent.execution_replay import ExecutionReplayStore, Observation, ReviewLabel, digest
from agent.prospective_validation import SignedRecord

FEATURES = ["confidence", "log_output_tokens", "log_latency", "confidence_x_length"]
HEX = r"^[a-f0-9]{64}$"


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class RankingCandidate(StrictModel):
    candidate_id: str = Field(min_length=1, max_length=160)
    confidence: float = Field(ge=0, le=1)
    token_count: int = Field(ge=0, le=1000000)
    latency_ms: float = Field(ge=0, le=86400000)


def features(candidate: RankingCandidate) -> list[float]:
    length = math.log1p(candidate.token_count) / math.log1p(8192)
    latency = math.log1p(candidate.latency_ms) / math.log1p(120000)
    return [candidate.confidence, length, latency, candidate.confidence * length]


class ReviewedCandidate(StrictModel):
    observation: Observation
    label: ReviewLabel

    def ranking_view(self) -> RankingCandidate:
        return RankingCandidate(
            candidate_id=self.observation.candidate,
            confidence=self.observation.confidence,
            token_count=self.observation.estimated_output_tokens,
            latency_ms=self.observation.latency_ms,
        )

    @property
    def preferred(self) -> bool:
        return self.label.verdict == "correct" and not self.label.unsafe


class PreferencePool(StrictModel):
    request_group: str = Field(pattern=HEX)
    task_family: str = Field(pattern=HEX)
    family_fingerprint: str = Field(pattern=HEX)
    first_seen: float = Field(ge=0)
    split: Literal["train", "test"]
    route: str
    high_risk: bool
    candidates: list[ReviewedCandidate] = Field(min_length=2, max_length=5)


class PreferenceCohort(SignedRecord):
    schema_version: Literal["1.0"] = "1.0"
    tenant: str = Field(pattern=HEX)
    cutoff: float = Field(ge=0)
    embargo_seconds: float = Field(ge=0)
    frozen_at: float = Field(ge=0)
    simulation: bool = False
    pools: list[PreferencePool]
    exclusions: dict[str, int]

    def verify(self, key: bytes) -> None:
        super().verify(key)
        if self.frozen_at <= self.cutoff + self.embargo_seconds:
            raise ValueError("invalid preference time window")
        families, groups = set(), set()
        for pool in self.pools:
            if pool.task_family in families or pool.request_group in groups:
                raise ValueError("preference family/request leakage")
            families.add(pool.task_family)
            groups.add(pool.request_group)
            ids = set()
            for row in pool.candidates:
                obs, label = row.observation, row.label
                # Verify the original payloads as well as the enclosing cohort.
                for record in (obs, label):
                    expected = digest(key, type(record).__name__, record.model_dump(exclude={"fingerprint"}))
                    if not hmac.compare_digest(record.fingerprint, expected):
                        raise ValueError("preference source integrity failed")
                if (
                    obs.tenant != self.tenant
                    or obs.request_group != pool.request_group
                    or obs.route != pool.route
                    or obs.high_risk != pool.high_risk
                    or obs.event_id in ids
                    or not obs.grounded
                    or label.event_id != obs.event_id
                    or label.observation_fingerprint != obs.fingerprint
                    or label.verdict == "ambiguous"
                    or label.reviewed_at < obs.observed_at
                    or label.reviewed_at > self.frozen_at
                    or (obs.origin != "runtime" and not self.simulation)
                    or pool.first_seen > obs.observed_at
                ):
                    raise ValueError("invalid preference source binding")
                ids.add(obs.event_id)
                if pool.split == "train":
                    if max(obs.observed_at, label.reviewed_at) > self.cutoff:
                        raise ValueError("training label arrived after cutoff")
                elif pool.first_seen <= self.cutoff + self.embargo_seconds:
                    raise ValueError("held-out family violates embargo")


def freeze_preferences(
    store: ExecutionReplayStore,
    tenant: str,
    *,
    cutoff: float,
    embargo_seconds: float,
    simulation: bool = False,
) -> PreferenceCohort:
    """Fix the earliest request per family BEFORE looking at review coverage."""
    frozen_at = store.clock()
    rows, assignments = store.snapshot(tenant)
    requests: dict[str, list[tuple[Observation, ReviewLabel | None]]] = {}
    for obs, label in rows:
        if obs.observed_at <= frozen_at:
            requests.setdefault(obs.request_group, []).append((obs, label))
    families: dict[str, list[str]] = {}
    excluded = Counter()
    for group in requests:
        assignment = assignments.get(group)
        if not assignment or not assignment.task_family:
            excluded["missing_preassigned_family"] += 1
            continue
        families.setdefault(assignment.task_family, []).append(group)
    pools = []
    for family, groups in sorted(families.items()):
        group = min(groups, key=lambda item: (min(row[0].observed_at for row in requests[item]), item))
        excluded["repeated_family_requests"] += len(groups) - 1
        siblings = requests[group]
        first_seen = min(row[0].observed_at for row in siblings)
        split = "train" if first_seen <= cutoff else "test"
        if split == "test" and first_seen <= cutoff + embargo_seconds:
            excluded["embargo"] += 1
            continue
        # No substitution of reviewed candidates: incomplete/ambiguous pools are held.
        if len(siblings) < 2 or any(label is None or label.verdict == "ambiguous" for _, label in siblings):
            excluded["incomplete_review_pool"] += 1
            continue
        if any(label.reviewed_at > (cutoff if split == "train" else frozen_at) for _, label in siblings):
            excluded["late_label"] += 1
            continue
        if any(not obs.grounded or (obs.origin != "runtime" and not simulation) for obs, _ in siblings):
            excluded["ungrounded_or_synthetic"] += 1
            continue
        routes = {(obs.route, obs.high_risk) for obs, _ in siblings}
        if len(routes) != 1:
            raise ValueError("inconsistent request scope")
        route, risk = next(iter(routes))
        pools.append(
            PreferencePool(
                request_group=group,
                task_family=family,
                family_fingerprint=assignments[group].fingerprint,
                first_seen=first_seen,
                split=split,
                route=route,
                high_risk=risk,
                candidates=[
                    ReviewedCandidate(observation=obs, label=label)
                    for obs, label in sorted(siblings, key=lambda row: row[0].candidate)
                ],
            )
        )
    cohort = PreferenceCohort(
        tenant=digest(store.key, "tenant", tenant),
        cutoff=cutoff,
        embargo_seconds=embargo_seconds,
        frozen_at=frozen_at,
        simulation=simulation,
        pools=pools,
        exclusions=dict(excluded),
    ).seal(store.key)
    cohort.verify(store.key)
    return cohort


def validate_preference_lineage(cohort: PreferenceCohort, store: ExecutionReplayStore, tenant: str) -> None:
    cohort.verify(store.key)
    if cohort.tenant != digest(store.key, "tenant", tenant):
        raise ValueError("preference tenant mismatch")
    rows, assignments = store.snapshot(tenant)
    current = {obs.event_id: (obs, label) for obs, label in rows}
    for pool in cohort.pools:
        assignment = assignments.get(pool.request_group)
        if (
            not assignment
            or assignment.fingerprint != pool.family_fingerprint
            or assignment.task_family != pool.task_family
        ):
            raise ValueError("preference family lineage revoked")
        family_rows = [
            obs
            for obs, _ in rows
            if assignments.get(obs.request_group)
            and assignments[obs.request_group].task_family == pool.task_family
            and obs.observed_at <= cohort.frozen_at
        ]
        if not family_rows:
            raise ValueError("preference source lineage revoked")
        first_group = min(family_rows, key=lambda obs: (obs.observed_at, obs.request_group)).request_group
        if (
            first_group != pool.request_group
            or min(obs.observed_at for obs in family_rows) != pool.first_seen
        ):
            raise ValueError("preference family representative changed")
        live_ids = {obs.event_id for obs, _ in rows if obs.request_group == pool.request_group}
        if live_ids != {row.observation.event_id for row in pool.candidates}:
            raise ValueError("preference candidate pool changed")
        for row in pool.candidates:
            obs, label = current.get(row.observation.event_id, (None, None))
            if (
                not obs
                or not label
                or obs.fingerprint != row.observation.fingerprint
                or label.fingerprint != row.label.fingerprint
            ):
                raise ValueError("preference source lineage revoked")


class FeatureSupport(StrictModel):
    route: str
    high_risk: bool
    families: int = Field(ge=20)
    lower: list[float] = Field(min_length=4, max_length=4)
    upper: list[float] = Field(min_length=4, max_length=4)

    @model_validator(mode="after")
    def valid_bounds(self):
        if any(lo > hi for lo, hi in zip(self.lower, self.upper, strict=True)):
            raise ValueError("invalid ranker support bounds")
        return self


class PreferenceArtifact(SignedRecord):
    schema_version: Literal["1.0"] = "1.0"
    model_type: Literal["bootstrapped-bradley-terry-metadata"] = "bootstrapped-bradley-terry-metadata"
    feature_names: list[str] = Field(default_factory=lambda: list(FEATURES))
    feature_scales: list[float] = Field(min_length=4, max_length=4)
    tenant: str = Field(pattern=HEX)
    simulation: bool
    training_families: int = Field(ge=20)
    pair_count: int = Field(ge=20)
    training_fingerprint: str = Field(pattern=HEX)
    members: list[list[float]] = Field(min_length=5, max_length=15)
    support: list[FeatureSupport] = Field(min_length=1)
    risk_multiplier: float = Field(default=2.0, ge=1, le=5)
    minimum_margin: float = Field(default=0.1, gt=0, le=2)

    def verify(self, key: bytes) -> None:
        if len(key) < 32:
            raise ValueError("preference artifact requires a key of at least 32 bytes")
        try:
            super().verify(key)
        except ValueError:
            # Schema 1.0 originally emitted the float default as integer 2.
            # Accept ONLY its authentic legacy representation, not other changes.
            payload = self.model_dump(exclude={"fingerprint"})
            payload["risk_multiplier"] = 2
            if self.risk_multiplier != 2.0 or not hmac.compare_digest(
                self.fingerprint, digest(key, type(self).__name__, payload)
            ):
                raise

    @model_validator(mode="after")
    def valid_features(self):
        if self.feature_names != FEATURES or any(
            len(w) != len(FEATURES) or any(not math.isfinite(x) or abs(x) > 100 for x in w)
            for w in self.members
        ):
            raise ValueError("invalid preference model features/weights")
        if any(not math.isfinite(scale) or not 0.001 <= scale <= 10 for scale in self.feature_scales):
            raise ValueError("invalid preference feature scales")
        if len({(scope.route, scope.high_risk) for scope in self.support}) != len(self.support):
            raise ValueError("duplicate ranker scope")
        return self


def _pairs(pool: PreferencePool) -> list[list[float]]:
    good = [features(row.ranking_view()) for row in pool.candidates if row.preferred]
    bad = [features(row.ranking_view()) for row in pool.candidates if not row.preferred]
    return [[x - y for x, y in zip(left, right, strict=True)] for left in good for right in bad]


def train_preferences(
    cohort: PreferenceCohort, replay_key: bytes, artifact_key: bytes, tenant: str
) -> PreferenceArtifact:
    cohort.verify(replay_key)
    if (
        len(artifact_key) < 32
        or artifact_key == replay_key
        or cohort.tenant != digest(replay_key, "tenant", tenant)
    ):
        raise ValueError("invalid preference training identity/key")
    training = [pool for pool in cohort.pools if pool.split == "train" and _pairs(pool)]
    counts = Counter((pool.route, pool.high_risk) for pool in training)
    scopes = {scope for scope, count in counts.items() if count >= 20}
    training = [pool for pool in training if (pool.route, pool.high_risk) in scopes]
    if len(training) < 20:
        raise ValueError("preference training requires 20 contrasting reviewed families per scope")
    vectors = [features(row.ranking_view()) for pool in training for row in pool.candidates]
    # Scale using training candidates only, never held-out distribution statistics.
    scales = [max(0.05, statistics.pstdev(vector[i] for vector in vectors)) for i in range(4)]
    pairs = [
        [[value / scale for value, scale in zip(pair, scales, strict=True)] for pair in _pairs(pool)]
        for pool in training
    ]
    members = []
    for seed in range(7):
        rng = random.Random(1700 + seed)
        sample = [pairs[rng.randrange(len(pairs))] for _ in pairs]
        weights = [0.0] * len(FEATURES)
        for _ in range(450):
            grad = [0.0] * len(FEATURES)
            for family_pairs in sample:
                for difference in family_pairs:
                    score = sum(w * x for w, x in zip(weights, difference, strict=True))
                    error = 1 / (1 + math.exp(max(-40, min(40, score))))
                    for index, value in enumerate(difference):
                        # Every family has equal weight, irrespective of sibling count.
                        grad[index] += error * value / len(family_pairs) / len(sample)
            weights = [w + 0.4 * (g - 0.01 * w) for w, g in zip(weights, grad, strict=True)]
        members.append([round(w, 10) for w in weights])
    support = []
    for route, risk in sorted(scopes):
        vectors = [
            features(row.ranking_view())
            for pool in training
            if (pool.route, pool.high_risk) == (route, risk)
            for row in pool.candidates
        ]
        support.append(
            FeatureSupport(
                route=route,
                high_risk=risk,
                families=counts[(route, risk)],
                lower=[min(v[i] for v in vectors) for i in range(4)],
                upper=[max(v[i] for v in vectors) for i in range(4)],
            )
        )
    # Test labels, selection flags, identifiers, and timestamps never enter features.
    train_hash = digest(
        replay_key, "preference-training", [pool.model_dump(mode="json") for pool in training]
    )
    return PreferenceArtifact(
        tenant=digest(artifact_key, "preference-tenant", tenant),
        simulation=cohort.simulation,
        training_families=len(training),
        pair_count=sum(map(len, pairs)),
        training_fingerprint=train_hash,
        members=members,
        support=support,
        feature_scales=scales,
    ).seal(artifact_key)


class RankingDecision(StrictModel):
    artifact_fingerprint: str
    status: Literal["reranked", "fallback"]
    reason: str
    baseline_candidate_id: str
    selected_candidate_id: str
    improvement_lcb: float = 0
    candidate_scores: dict[str, float] = Field(default_factory=dict)


class PreferenceRanker:
    def __init__(
        self, artifact: PreferenceArtifact, key: bytes, tenant: str, *, allow_simulation: bool = False
    ):
        artifact.verify(key)
        if artifact.tenant != digest(key, "preference-tenant", tenant):
            raise ValueError("preference ranker tenant mismatch")
        if artifact.simulation and not allow_simulation:
            raise ValueError("synthetic preference artifacts cannot be activated")
        self.artifact = artifact.model_copy(deep=True)
        self.tenant = tenant

    @classmethod
    def load(cls, path: str | Path, key: bytes, tenant: str):
        return cls(
            PreferenceArtifact.model_validate_json(Path(path).read_text(encoding="utf-8")), key, tenant
        )

    def rank(
        self, candidates: list[RankingCandidate], baseline_id: str, *, route: str, high_risk: bool
    ) -> RankingDecision:
        artifact = self.artifact
        decision = RankingDecision(
            artifact_fingerprint=artifact.fingerprint,
            status="fallback",
            reason="insufficient_candidates",
            baseline_candidate_id=baseline_id,
            selected_candidate_id=baseline_id,
        )
        if len({row.candidate_id for row in candidates}) != len(candidates) or baseline_id not in {
            row.candidate_id for row in candidates
        }:
            raise ValueError("invalid ranker candidate pool")
        if len(candidates) < 2:
            return decision
        scope = next(
            (scope for scope in artifact.support if scope.route == route and scope.high_risk == high_risk),
            None,
        )
        vectors = {row.candidate_id: features(row) for row in candidates}
        if scope is None or any(
            any(
                value < lo - 1e-9 or value > hi + 1e-9
                for value, lo, hi in zip(vector, scope.lower, scope.upper, strict=True)
            )
            for vector in vectors.values()
        ):
            decision.reason = "unsupported_scope_or_feature_ood"
            return decision
        scores = {
            name: [
                sum(
                    w * x / scale for w, x, scale in zip(member, vector, artifact.feature_scales, strict=True)
                )
                for member in artifact.members
            ]
            for name, vector in vectors.items()
        }
        decision.candidate_scores = {
            name: round(statistics.mean(values), 6) for name, values in scores.items()
        }
        winner = max(scores, key=lambda name: (statistics.mean(scores[name]), name))
        delta = [left - right for left, right in zip(scores[winner], scores[baseline_id], strict=True)]
        lcb = statistics.mean(delta) - artifact.risk_multiplier * statistics.stdev(delta)
        decision.improvement_lcb = round(lcb, 6)
        decision.reason = "no_supported_preference_margin"
        if winner != baseline_id and lcb >= artifact.minimum_margin:
            decision.status, decision.reason, decision.selected_candidate_id = (
                "reranked",
                "conservative_preference_margin",
                winner,
            )
        return decision


def artifact_identity(artifact: PreferenceArtifact) -> str:
    """Stable identity for reservation before any held-out scoring."""
    return hashlib.sha256(json.dumps(artifact.model_dump(mode="json"), sort_keys=True).encode()).hexdigest()
