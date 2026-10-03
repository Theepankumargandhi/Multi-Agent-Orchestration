"""Owner-activated preference releases with live lineage and delayed-outcome guards."""

from __future__ import annotations

import math
import sqlite3
from typing import Literal

from pydantic import Field, model_validator

from agent.adaptive_compute import ComputePlan, ComputePolicy, DeliberationReceipt
from agent.execution_replay import ExecutionReplayStore, Observation, RequestFamily, _verify, digest
from agent.preference_ranking import PreferenceArtifact, PreferenceRanker, StrictModel
from agent.preference_shadow import PreferenceShadowStore, ShadowApproval, ShadowCohort, check_approval
from agent.prospective_validation import SignedRecord

HEX = r"^[a-f0-9]{64}$"
ROUTES = {"web", "hybrid", "rag", "kg", "math", "general", "code"}


class SentinelPolicy(StrictModel):
    """Pinned before serving; a review-SLA failure is also a monitored error."""

    version: Literal["preference-sentinel-v1"] = "preference-sentinel-v1"
    maximum_error: float = Field(default=0.2, gt=0, lt=1)
    alternative_error: float = Field(default=0.5, gt=0, lt=1)
    alpha: float = Field(default=0.05, gt=0, le=0.1)
    review_deadline_seconds: float = Field(default=3600.0, ge=60, le=86400)
    minimum_coverage_samples: int = Field(default=20, ge=10, le=10000)
    minimum_review_coverage: float = Field(default=0.8, ge=0.8, le=1)

    @model_validator(mode="after")
    def valid_bet(self):
        if self.alternative_error <= self.maximum_error:
            raise ValueError("alternative error must exceed the null bound")
        return self


class Deployment(SignedRecord):
    deployment_id: str = Field(pattern=HEX)
    tenant: str = Field(pattern=HEX)
    artifact: PreferenceArtifact
    approval: ShadowApproval
    cohort: ShadowCohort
    compute_policy: ComputePolicy
    sentinel_policy: SentinelPolicy
    activated_at: float = Field(ge=0)
    owner: str = Field(pattern=HEX)


class DeploymentState(SignedRecord):
    tenant: str = Field(pattern=HEX)
    revision: int = Field(ge=1)
    deployment_id: str = Field(pattern=HEX)
    status: Literal["active", "revoked"]
    reason: Literal[
        "owner_activation",
        "owner_revocation",
        "source_or_lease_invalid",
        "unsafe_outcome",
        "sequential_error",
        "review_coverage",
    ]
    changed_at: float = Field(ge=0)
    previous: str = Field(default="", pattern=r"^(|[a-f0-9]{64})$")


class ServedChoice(SignedRecord):
    deployment_id: str = Field(pattern=HEX)
    tenant: str = Field(pattern=HEX)
    event_id: str = Field(pattern=HEX)
    observation_fingerprint: str = Field(pattern=HEX)
    request_group: str = Field(pattern=HEX)
    family: str = Field(pattern=HEX)
    family_fingerprint: str = Field(pattern=HEX)
    scope: str
    observed_at: float = Field(ge=0)


def sequential_trace(errors: list[int], policy: SentinelPolicy, scopes: int) -> dict:
    """Fixed Bernoulli likelihood-ratio e-process, evaluated in capture order.

    Null: each combined quality/review-SLA error has conditional probability <=p.
    E[bet | past] <=1 because q>p. Ville plus scope-wise alpha allocation bounds
    any-time crossing per deployment under this null, NOT biased-label factuality.
    """
    if scopes < 1 or scopes > 14 or any(error not in (0, 1) for error in errors):
        raise ValueError("invalid sequential outcomes or scopes")
    p, q = policy.maximum_error, policy.alternative_error
    log_e = maximum = 0.0
    for error in errors:
        log_e += math.log(q / p) if error else math.log((1 - q) / (1 - p))
        maximum = max(maximum, log_e)
    threshold = math.log(scopes / policy.alpha)
    return {
        "samples": len(errors),
        "errors": sum(errors),
        "log_e": log_e,
        "maximum_log_e": maximum,
        "log_threshold": threshold,
        "crossed": maximum >= threshold,
    }


class PreferenceDeploymentStore:
    def __init__(self, replay: ExecutionReplayStore, artifact_key: bytes):
        if len(artifact_key) < 32 or artifact_key == replay.key:
            raise ValueError("deployment needs an independent artifact key of at least 32 bytes")
        self.replay, self.artifact_key = replay, artifact_key
        with replay._db() as db:
            db.execute(
                "CREATE TABLE IF NOT EXISTS preference_deployments (deployment_id TEXT PRIMARY KEY, tenant TEXT NOT NULL, approval TEXT NOT NULL, payload TEXT NOT NULL, UNIQUE(tenant, approval))"
            )
            db.execute(
                "CREATE TABLE IF NOT EXISTS preference_deployment_states (tenant TEXT PRIMARY KEY, payload TEXT NOT NULL)"
            )
            db.execute(
                "CREATE TABLE IF NOT EXISTS preference_deployment_audit (tenant TEXT NOT NULL, revision INTEGER NOT NULL, payload TEXT NOT NULL, PRIMARY KEY(tenant, revision))"
            )
            # No cascade: deleting an observation must revoke, not erase its exposure.
            db.execute(
                "CREATE TABLE IF NOT EXISTS preference_served (event_id TEXT PRIMARY KEY, tenant TEXT NOT NULL, deployment_id TEXT NOT NULL, payload TEXT NOT NULL)"
            )

    def _tenant(self, tenant: str) -> str:
        if not tenant.strip():
            raise ValueError("tenant is required")
        return digest(self.replay.key, "tenant", tenant)

    def _state(self, db, tenant_hash: str) -> DeploymentState | None:
        states = db.execute(
            "SELECT revision, payload FROM preference_deployment_audit WHERE tenant=? ORDER BY revision",
            (tenant_hash,),
        ).fetchall()
        row = db.execute(
            "SELECT payload FROM preference_deployment_states WHERE tenant=?", (tenant_hash,)
        ).fetchone()
        if row is None and not states:
            return None
        previous = ""
        for expected, (number, payload) in enumerate(states, 1):
            state = DeploymentState.model_validate_json(payload)
            state.verify(self.replay.key)
            if (
                number != state.revision
                or number != expected
                or state.tenant != tenant_hash
                or state.previous != previous
            ):
                raise ValueError("deployment audit chain integrity failed")
            previous = state.fingerprint
        if row is None or not states or DeploymentState.model_validate_json(row[0]) != state:
            raise ValueError("deployment pointer integrity failed")
        return state

    def state(self, tenant: str) -> DeploymentState | None:
        with self.replay._db() as db:
            db.execute("BEGIN")
            return self._state(db, self._tenant(tenant))

    def _change(
        self,
        db,
        previous: DeploymentState | None,
        tenant_hash: str,
        deployment_id: str,
        status: str,
        reason: str,
    ) -> DeploymentState:
        now = self.replay.clock()
        if previous is not None and now < previous.changed_at:
            raise ValueError("deployment clock moved backwards")
        state = DeploymentState(
            tenant=tenant_hash,
            revision=previous.revision + 1 if previous else 1,
            deployment_id=deployment_id,
            status=status,
            reason=reason,
            changed_at=now,
            previous=previous.fingerprint if previous else "",
        ).seal(self.replay.key)
        db.execute(
            "INSERT INTO preference_deployment_audit VALUES (?, ?, ?)",
            (tenant_hash, state.revision, state.model_dump_json()),
        )
        db.execute(
            "INSERT INTO preference_deployment_states VALUES (?, ?) ON CONFLICT(tenant) DO UPDATE SET payload=excluded.payload",
            (tenant_hash, state.model_dump_json()),
        )
        return state

    def activate(
        self,
        tenant: str,
        ranker: PreferenceRanker,
        cohort: ShadowCohort,
        approval: ShadowApproval,
        policy: ComputePolicy,
        *,
        owner: str,
        expected_revision: int,
        sentinel: SentinelPolicy | None = None,
    ) -> DeploymentState:
        if not owner.strip() or tenant != ranker.tenant:
            raise ValueError("activation requires owner and matching tenant")
        PreferenceShadowStore(self.replay).validate_lineage(cohort, tenant)
        if (
            cohort.study.simulation
            or approval.cohort_fingerprint != cohort.fingerprint
            or approval.study_id != cohort.study.study_id
            or approval.artifact_fingerprint != cohort.study.artifact_fingerprint
            or not approval.scopes
            or any(
                scope not in {f"{route}:{risk}" for route in ROUTES for risk in (0, 1)}
                for scope in approval.scopes
            )
        ):
            raise ValueError("approval does not bind this production cohort")
        for scope in approval.scopes:
            route, risk = scope.split(":")
            check_approval(
                approval,
                ranker,
                self.artifact_key,
                policy,
                route=route,
                high_risk=bool(int(risk)),
                now=self.replay.clock(),
            )
        tenant_hash = self._tenant(tenant)
        deployment = Deployment(
            deployment_id=digest(
                self.replay.key, "preference-deployment", [tenant_hash, approval.fingerprint]
            ),
            tenant=tenant_hash,
            artifact=ranker.artifact,
            approval=approval,
            cohort=cohort,
            compute_policy=policy,
            sentinel_policy=sentinel or SentinelPolicy(),
            activated_at=self.replay.clock(),
            owner=digest(self.replay.key, "deployment-owner", owner),
        ).seal(self.replay.key)
        with self.replay._db() as db:
            db.execute("BEGIN IMMEDIATE")
            previous = self._state(db, tenant_hash)
            if expected_revision != (previous.revision if previous else 0):
                raise ValueError("stale deployment revision")
            if db.execute(
                "SELECT 1 FROM preference_deployments WHERE tenant=? AND approval=?",
                (tenant_hash, approval.fingerprint),
            ).fetchone():
                raise ValueError("approval already exposed; cannot reset or reactivate its monitor")
            db.execute(
                "INSERT INTO preference_deployments VALUES (?, ?, ?, ?)",
                (deployment.deployment_id, tenant_hash, approval.fingerprint, deployment.model_dump_json()),
            )
            return self._change(
                db, previous, tenant_hash, deployment.deployment_id, "active", "owner_activation"
            )

    def deployment(self, state: DeploymentState) -> Deployment:
        state.verify(self.replay.key)
        with self.replay._db() as db:
            row = db.execute(
                "SELECT tenant, approval, payload FROM preference_deployments WHERE deployment_id=?",
                (state.deployment_id,),
            ).fetchone()
        if row is None:
            raise ValueError("deployment is missing")
        deployment = Deployment.model_validate_json(row[2])
        deployment.verify(self.replay.key)
        if (
            deployment.deployment_id != state.deployment_id
            or deployment.tenant != state.tenant
            or row[0] != state.tenant
            or row[1] != deployment.approval.fingerprint
        ):
            raise ValueError("deployment index integrity failed")
        return deployment

    def revoke(
        self, tenant: str, *, expected_revision: int, reason: str = "owner_revocation"
    ) -> DeploymentState:
        tenant_hash = self._tenant(tenant)
        with self.replay._db() as db:
            db.execute("BEGIN IMMEDIATE")
            previous = self._state(db, tenant_hash)
            if previous is None or previous.revision != expected_revision:
                raise ValueError("stale deployment revision")
            if previous.status == "revoked":
                return previous
            return self._change(db, previous, tenant_hash, previous.deployment_id, "revoked", reason)

    def monitor(self, tenant: str, deployment: Deployment) -> dict:
        deployment.verify(self.replay.key)
        if deployment.tenant != self._tenant(tenant):
            raise ValueError("monitor tenant mismatch")
        records, families = self.replay.snapshot(tenant)
        events = {obs.event_id: (obs, label) for obs, label in records}
        with self.replay._db() as db:
            rows = db.execute(
                "SELECT event_id, payload FROM preference_served WHERE tenant=? AND deployment_id=?",
                (deployment.tenant, deployment.deployment_id),
            ).fetchall()
        choices = []
        unsafe = 0
        now = self.replay.clock()
        for event_id, payload in rows:
            choice = ServedChoice.model_validate_json(payload)
            choice.verify(self.replay.key)
            obs, label = events.get(event_id, (None, None))
            family = families.get(choice.request_group)
            if (
                choice.event_id != event_id
                or choice.deployment_id != deployment.deployment_id
                or choice.tenant != deployment.tenant
                or obs is None
                or obs.fingerprint != choice.observation_fingerprint
                or not obs.selected
                or obs.origin != "runtime"
                or family is None
                or family.fingerprint != choice.family_fingerprint
                or family.task_family != choice.family
                or choice.scope != f"{obs.route}:{int(obs.high_risk)}"
                or choice.scope not in deployment.approval.scopes
                or choice.observed_at != obs.observed_at
                or not deployment.activated_at <= obs.observed_at <= now
                or (label is not None and not obs.observed_at <= label.reviewed_at <= now)
            ):
                raise ValueError("served decision lineage changed")
            unsafe += int(label is not None and label.unsafe)
            choices.append((choice, label))
        # Representatives are fixed by capture order, never by label availability.
        first = {}
        for obs, _ in records:
            family = families.get(obs.request_group)
            if family and family.task_family:
                value = (obs.observed_at, obs.request_group)
                first[family.task_family] = min(first.get(family.task_family, value), value)
        selected = {}
        for choice, label in sorted(choices, key=lambda row: (row[0].observed_at, row[0].event_id)):
            if (
                first[choice.family][1] == choice.request_group
                and first[choice.family][0] >= deployment.activated_at
            ):
                selected.setdefault(choice.family, (choice, label))
        scopes = {}
        reason = "unsafe_outcome" if unsafe else "healthy"
        for scope in deployment.approval.scopes:
            errors, reviewed = [], 0
            for choice, label in selected.values():
                if choice.scope != scope:
                    continue
                deadline = choice.observed_at + deployment.sentinel_policy.review_deadline_seconds
                if deadline > now:
                    break  # contiguous event-time prefix, regardless of early labels
                timely = label is not None and label.reviewed_at <= deadline and label.verdict != "ambiguous"
                reviewed += int(timely)
                errors.append(int(not timely or label.unsafe or label.verdict != "correct"))
            metrics = sequential_trace(errors, deployment.sentinel_policy, len(deployment.approval.scopes))
            metrics["reviewed"] = reviewed
            metrics["coverage"] = reviewed / len(errors) if errors else None
            scopes[scope] = metrics
            if reason == "healthy" and metrics["crossed"]:
                reason = "sequential_error"
            if (
                reason == "healthy"
                and len(errors) >= deployment.sentinel_policy.minimum_coverage_samples
                and reviewed / len(errors) < deployment.sentinel_policy.minimum_review_coverage
            ):
                reason = "review_coverage"
        report = {
            "deployment_id": deployment.deployment_id,
            "reason": reason,
            "unsafe_choices": unsafe,
            "captured_choices": len(choices),
            "fresh_families": len(selected),
            "scopes": scopes,
            "as_of": now,
            "claim_scope": "conditional combined correctness and timely-review SLA; per deployment, no causal lift claim",
        }
        report["fingerprint"] = digest(self.replay.key, "preference-sentinel-report", report)
        return report

    def admit(
        self, tenant: str, policy: ComputePolicy, *, route: str, high_risk: bool
    ) -> tuple[PreferenceRanker, DeploymentState, dict]:
        state = self.state(tenant)
        if state is None or state.status != "active":
            raise ValueError("no active preference deployment")
        deployment = self.deployment(state)
        ranker = PreferenceRanker(deployment.artifact, self.artifact_key, tenant)
        # Scope/policy mismatch is a request-level fallback, not global revocation.
        if (
            policy != deployment.compute_policy
            or f"{route}:{int(high_risk)}" not in deployment.approval.scopes
        ):
            raise ValueError("deployment does not support request policy or scope")
        try:
            check_approval(
                deployment.approval,
                ranker,
                self.artifact_key,
                policy,
                route=route,
                high_risk=high_risk,
                now=self.replay.clock(),
            )
            PreferenceShadowStore(self.replay).validate_lineage(deployment.cohort, tenant)
            report = self.monitor(tenant, deployment)
        except ValueError:
            self.revoke(tenant, expected_revision=state.revision, reason="source_or_lease_invalid")
            raise
        if report["reason"] != "healthy":
            self.revoke(tenant, expected_revision=state.revision, reason=report["reason"])
            raise ValueError(f"preference sentinel blocked: {report['reason']}")
        if self.state(tenant) != state:
            raise ValueError("deployment changed during admission")
        return ranker, state, report

    def capture(
        self,
        state: DeploymentState,
        plan: ComputePlan,
        receipt: DeliberationReceipt,
        *,
        tenant: str,
        request_id: str,
        consent: bool,
        task_family: str | None = None,
        task_family_fingerprint: str | None = None,
        compute_key: bytes | None = None,
    ) -> list[str]:
        state.verify(self.replay.key)
        if consent is not True or state.tenant != self._tenant(tenant) or state.status != "active":
            raise ValueError("serving capture requires consent and active tenant deployment")
        deployment = self.deployment(state)
        audit = receipt.preference_ranking or {}
        if (
            receipt.status == "released"
            and audit.get("artifact_fingerprint") != deployment.artifact.fingerprint
        ):
            raise ValueError("serving receipt model mismatch")
        group = digest(self.replay.key, "request", [tenant, request_id])
        # Initialize before the writer transaction; commit checks below are reads.
        shadow = PreferenceShadowStore(self.replay)

        def bind(db: sqlite3.Connection):
            if self._state(db, state.tenant) != state:
                raise ValueError("deployment changed before serving capture")
            if not deployment.activated_at <= self.replay.clock() < deployment.approval.expires_at:
                raise ValueError("serving lease expired")
            # BEGIN IMMEDIATE excludes concurrent source/label writers. Separate
            # WAL readers see the committed history (not this pending exposure).
            shadow.validate_lineage(deployment.cohort, tenant)
            if self.monitor(tenant, deployment)["reason"] != "healthy":
                raise ValueError("sentinel changed before serving commit")
            if receipt.status != "released":
                return
            candidate = digest(self.replay.key, "candidate", receipt.selected_candidate_id)
            event_id = digest(self.replay.key, "event", [group, candidate])
            obs_row = db.execute("SELECT payload FROM observations WHERE event_id=?", (event_id,)).fetchone()
            family_row = db.execute(
                "SELECT payload FROM request_families WHERE request_group=?", (group,)
            ).fetchone()
            obs = Observation.model_validate_json(obs_row[0])
            family = RequestFamily.model_validate_json(family_row[0])
            _verify(obs, self.replay.key)
            _verify(family, self.replay.key)
            scope = f"{plan.route}:{int(plan.high_risk)}"
            if (
                not family.task_family
                or scope not in deployment.approval.scopes
                or not obs.selected
                or obs.tenant != state.tenant
                or obs.origin != "runtime"
                or obs.observed_at < deployment.activated_at
            ):
                raise ValueError("serving capture lacks preassigned family or valid scope")
            choice = ServedChoice(
                deployment_id=state.deployment_id,
                tenant=state.tenant,
                event_id=event_id,
                observation_fingerprint=obs.fingerprint,
                request_group=group,
                family=family.task_family,
                family_fingerprint=family.fingerprint,
                scope=scope,
                observed_at=obs.observed_at,
            ).seal(self.replay.key)
            previous = db.execute(
                "SELECT payload FROM preference_served WHERE event_id=?", (event_id,)
            ).fetchone()
            if previous:
                if ServedChoice.model_validate_json(previous[0]) != choice:
                    raise ValueError("conflicting serving capture retry")
                return
            if db.execute("SELECT 1 FROM labels WHERE event_id=?", (event_id,)).fetchone():
                raise ValueError("serving binding must precede review")
            db.execute(
                "INSERT INTO preference_served VALUES (?, ?, ?, ?)",
                (event_id, state.tenant, state.deployment_id, choice.model_dump_json()),
            )

        return self.replay.capture(
            plan,
            receipt,
            tenant=tenant,
            request_id=request_id,
            consent=True,
            task_family=task_family,
            task_family_fingerprint=task_family_fingerprint,
            compute_key=compute_key,
            on_capture=bind,
        )
