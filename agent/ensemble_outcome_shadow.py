"""Non-serving final-outcome shadows for independently reviewed STEP ensembles."""

from __future__ import annotations

from collections import Counter
from typing import Literal

from pydantic import Field

from agent.execution_replay import digest
from agent.preference_ranking import StrictModel
from agent.reviewed_verifier_uncertainty import ReviewedStepEnsemble, ReviewedStepScorer, StepEstimate
from agent.verifier_shadow import (
    VerifierChoice,
    VerifierCohort,
    VerifierComparison,
    VerifierMember,
    VerifierShadowStore,
    VerifierStudy,
)


class StepToAnswerPolicy(StrictModel):
    version: Literal["all-reviewed-steps-min-lower-v1"] = "all-reviewed-steps-min-lower-v1"
    minimum_correct_probability: float = Field(default=0.5, ge=0.5, le=0.9)


class StepAnswerDecision(StrictModel):
    allowed: bool
    quality_score: float = Field(ge=0, le=1)
    reasons: list[str]
    estimates: list[StepEstimate] = Field(min_length=1, max_length=64)


def aggregate_steps(steps, estimates, model_policy, aggregation):
    """Quality uses P(correct), NOT confidence in either binary class.

    This minimum is a ranking heuristic, not P(the answer is correct). Every
    step must pass; at least one verification step and a terminal answer exist.
    """
    if not steps or len(steps) != len(estimates):
        raise ValueError("step aggregation requires aligned nonempty workflow estimates")
    reasons = []
    if not any(step.kind == "verify" for step in steps) or steps[-1].kind != "answer":
        reasons.append("missing_required_verify_or_answer")
    if any(not step.policy_allowed or step.error for step in steps):
        reasons.append("policy_or_execution_failure")
    if any(not estimate.feature_supported for estimate in estimates):
        reasons.append("unfamiliar_features")
    if any(not estimate.accepted for estimate in estimates):
        reasons.append("step_prediction_deferred")
    if any(
        estimate.mean < aggregation.minimum_correct_probability or not estimate.predicted_correct
        for estimate in estimates
    ):
        reasons.append("predicted_incorrect_step")
    quality = min(
        max(0.0, estimate.mean - model_policy.spread_penalty * estimate.spread) for estimate in estimates
    )
    return StepAnswerDecision(
        allowed=not reasons, quality_score=quality, reasons=reasons, estimates=estimates
    )


class EnsembleOutcomeStudy(VerifierStudy):
    candidate: ReviewedStepEnsemble
    aggregation: StepToAnswerPolicy


class EnsembleOutcomeChoice(VerifierChoice):
    step_decision: StepAnswerDecision


def choose_unguarded(choices, required_consensus):
    return max(
        [row for row in choices if row.eligible and row.consensus >= required_consensus],
        key=lambda row: (
            row.step_decision.quality_score,
            row.snapshot.source.confidence,
            row.consensus,
            -row.snapshot.source.estimated_output_tokens,
            row.tie_rank,
        ),
        default=None,
    )


class EnsembleOutcomeComparison(VerifierComparison):
    choices: list[EnsembleOutcomeChoice] = Field(min_length=1, max_length=5)
    unguarded_event: str = Field(pattern=r"^(|[a-f0-9]{64})$")

    def releasable(self, shadow=False):
        rows = super().releasable(shadow)
        return [row for row in rows if row.step_decision.allowed] if shadow else rows

    def select_unguarded(self, threshold):
        row = next(
            (row for row in self.choices if row.snapshot.source.event_id == self.unguarded_event), None
        )
        return self.unguarded_event if row and row.step_decision.quality_score >= threshold else ""

    def verify(self, key):
        super().verify(key)
        chosen = choose_unguarded(self.releasable(), self.required_consensus)
        if self.unguarded_event != (chosen.snapshot.source.event_id if chosen else ""):
            raise ValueError("unguarded ensemble choice violates frozen release policy")
        if any(
            row.score != (row.step_decision.quality_score if row.step_decision.allowed else 0.0)
            for row in self.choices
        ):
            raise ValueError("ensemble score does not bind its step decision")


class EnsembleOutcomeMember(VerifierMember):
    comparison: EnsembleOutcomeComparison


class EnsembleOutcomeCohort(VerifierCohort):
    study: EnsembleOutcomeStudy
    members: list[EnsembleOutcomeMember]


class EnsembleOutcomeShadowStore(VerifierShadowStore):
    study_schema = EnsembleOutcomeStudy
    choice_schema = EnsembleOutcomeChoice
    comparison_schema = EnsembleOutcomeComparison
    member_schema = EnsembleOutcomeMember
    cohort_schema = EnsembleOutcomeCohort
    study_table = "ensemble_outcome_studies"
    comparison_table = "ensemble_outcome_comparisons"

    def check_study(self, study, tenant):
        key = self.replay.key
        study.verify(key)
        study.trial_policy.check()
        study.candidate.verify(self.model_key)
        StepToAnswerPolicy.model_validate(study.aggregation.model_dump())
        report = study.training_report
        if (
            study.tenant != digest(key, "tenant", tenant)
            or study.candidate.tenant != digest(self.model_key, "reviewed-step-tenant", tenant)
            or study.candidate.cohort_fingerprint != study.training_cohort.fingerprint
            or study.candidate.simulation != study.training_cohort.simulation
            or study.training_cohort.frozen_at > study.registered_at
            or report.get("fingerprint")
            != digest(key, "reviewed-step-report", {k: v for k, v in report.items() if k != "fingerprint"})
            or study.candidate.report_fingerprint != report.get("fingerprint")
            or report.get("cohort_fingerprint") != study.training_cohort.fingerprint
            or report.get("policy") != study.candidate.policy.model_dump()
            or report.get("gate_passed") is not True
            or (not study.candidate.simulation and report.get("candidate_ready_for_review") is not True)
        ):
            raise ValueError("ensemble training provenance or tenant mismatch")
        self.process.validate_lineage(tenant, study.training_cohort)

    def assess_snapshot(self, study, snapshot):
        steps = [step.process_step(i) for i, step in enumerate(snapshot.steps)]
        estimates = ReviewedStepScorer(study.candidate, self.model_key).estimates(
            steps, snapshot.source.high_risk
        )
        decision = aggregate_steps(steps, estimates, study.candidate.policy, study.aggregation)
        return decision.quality_score if decision.allowed else 0.0, {"step_decision": decision}

    def proposal_allowed(self, extras):
        return extras["step_decision"].allowed

    def comparison_extras(self, study, choices):
        required = (
            study.compute_policy.high_risk_min_consensus
            if choices[0].snapshot.source.high_risk
            else study.compute_policy.min_consensus
        )
        chosen = choose_unguarded(choices, required)
        return {"unguarded_event": chosen.snapshot.source.event_id if chosen else ""}

    def review_events(self, study, row):
        return super().review_events(study, row) | {
            row.select_unguarded(t) for t in study.trial_policy.curve_thresholds
        } - {""}

    def report_extras(self, cohort):
        # Reuse identical terminal-outcome metrics, only AFTER ledger reservation.
        from evals.verifier_shadow_evaluation import threshold_metrics

        reasons = Counter(
            reason
            for member in cohort.members
            for row in member.comparison.choices
            for reason in row.step_decision.reasons
        )
        return {
            "pipeline": "reviewed-step-ensemble-to-terminal-outcomes-v1",
            "aggregation_policy": cohort.study.aggregation.model_dump(),
            "step_model_policy": cohort.study.candidate.policy.model_dump(),
            "step_block_reasons": dict(sorted(reasons.items())),
            "guard_blocked_candidates": sum(
                not row.step_decision.allowed
                for member in cohort.members
                for row in member.comparison.choices
            ),
            "unguarded_primary_ablation": threshold_metrics(
                cohort, cohort.study.trial_policy.primary_threshold, unguarded=True
            ),
            "unguarded_risk_coverage_curve": [
                threshold_metrics(cohort, t, unguarded=True)
                for t in cohort.study.trial_policy.curve_thresholds
            ],
            "score_claim": "minimum step P(correct) minus spread penalty is a ranking heuristic, not calibrated answer probability",
        }
