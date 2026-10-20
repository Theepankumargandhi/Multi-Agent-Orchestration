"""Bounded metamorphic workflow compilation. No eval, inferred laws or host application imports."""
from __future__ import annotations

import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Literal

from pydantic import Field, model_validator

from code_agent.upgrade_models import StrictModel, fingerprint
from code_agent.upgrade_sessions import SessionScenario, session_key, validate_session_scope

RELATION_GENERATOR = "owner-laws-ordered-workflows-v1"
RelationKind = Literal["idempotent_retry", "read_noninterference", "rejected_call_rollback", "commutative_pair"]


class RelationSeed(StrictModel):
    scenario: SessionScenario
    relation: RelationKind
    operation_indices: list[int] = Field(min_length=1, max_length=8)
    expected_error: str | None = Field(default=None, pattern=r"^[A-Za-z][A-Za-z0-9_]{0,99}$")

    @model_validator(mode="after")
    def explicit_operations(self):
        observer = self.scenario.steps[-1]
        if observer.args or observer.kwargs:
            raise ValueError("the final seed step must be a no-argument observation")
        if len(set(self.operation_indices)) != len(self.operation_indices):
            raise ValueError("operation indices must be unique")
        if (self.relation == "rejected_call_rollback") != bool(self.expected_error):
            raise ValueError("rollback laws require an exact expected exception class")
        for index in self.operation_indices:
            extra = 1 if self.relation == "commutative_pair" else 0
            if index < 0 or index + extra >= len(self.scenario.steps) - 1:
                raise ValueError("selected operations must precede the final observation")
            if any(step.method == observer.method for step in self.scenario.steps[index:index + 1 + extra]):
                raise ValueError("an observation cannot stand in for an effectful operation")
            if extra and self.scenario.steps[index] == self.scenario.steps[index + 1]:
                raise ValueError("commutativity requires two distinct operation inputs")
        return self


class RelationPolicy(StrictModel):
    seeds: list[RelationSeed] = Field(min_length=1, max_length=8)
    seed: int = Field(default=1729, ge=0, le=2**32 - 1)
    discovery_cases: int = Field(default=4, ge=1, le=8)
    final_cases: int = Field(default=4, ge=1, le=8)
    max_feedback_cases: int = Field(default=2, ge=1, le=3)

    @model_validator(mode="after")
    def bounded(self):
        ids = [row.scenario.scenario_id for row in self.seeds]
        if len(set(ids)) != len(ids):
            raise ValueError("relation seed IDs must be unique")
        if len(self.model_dump_json().encode()) > 40000:
            raise ValueError("relation policy exceeds its byte budget")
        return self


@dataclass(frozen=True)
class RelationCase:
    case_id: str
    relation: str
    scenarios: tuple[SessionScenario, ...]
    before: int
    middle: int
    after: int
    error_index: int | None
    expected_error: str | None

    def payload(self):
        return {"case_id": self.case_id, "relation": self.relation, "scenarios": [row.model_dump() for row in self.scenarios],
            "before": self.before, "middle": self.middle, "after": self.after,
            "error_index": self.error_index, "expected_error": self.expected_error}


@dataclass(frozen=True)
class RelationCampaign:
    discovery: tuple[RelationCase, ...]
    final: tuple[RelationCase, ...]
    commitment: str
    final_fingerprint: str


def _compile(seed, index):
    template = seed.scenario
    prefix, operation, observer = template.steps[:index], template.steps[index], template.steps[-1]
    before, middle = index, index + 2
    if seed.relation == "idempotent_retry":
        variants = [[*prefix, observer, operation, observer, operation, observer]]
    elif seed.relation == "read_noninterference":
        variants = [[*prefix, observer, operation, observer, observer]]
    elif seed.relation == "rejected_call_rollback":
        variants = [[*prefix, observer, operation, observer]]
    else:
        other = template.steps[index + 1]
        variants = [[*prefix, observer, operation, observer, other, observer],
            [*prefix, observer, other, observer, operation, observer]]
    identity = fingerprint({"seed": seed.model_dump(), "index": index})[:20]
    scenarios = tuple(SessionScenario.model_validate({**template.model_dump(), "scenario_id": f"relation-{identity}-{number}",
        "steps": [step.model_dump() for step in steps]}) for number, steps in enumerate(variants))
    return RelationCase("law-" + identity, seed.relation, scenarios, before, middle,
        len(scenarios[0].steps) - 1, index + 1 if seed.expected_error else None, seed.expected_error)


def generate_relations(spec, policy):
    groups = defaultdict(list)
    seen = set()
    for seed in policy.seeds:
        for index in seed.operation_indices:
            case = _compile(seed, index)
            key = fingerprint({"relation": case.relation, "inputs": sorted(session_key(row) for row in case.scenarios),
                "expected_error": case.expected_error})
            if key not in seen:
                groups[case.relation].append(case)
                seen.add(key)
    rng = random.Random(policy.seed)
    for kind in sorted(groups):
        groups[kind].sort(key=lambda row: row.case_id)
        rng.shuffle(groups[kind])
    # Give both sets one example per available law when their budgets permit it.
    discovery, final, discovery_inputs, final_inputs = [], [], set(), set()

    def admit(row, destination, forbidden, known):
        inputs = {session_key(scenario) for scenario in row.scenarios}
        if inputs & forbidden:
            return False
        destination.append(row)
        known.update(inputs)
        return True

    for kind in sorted(groups):
        if len(groups[kind]) >= 2 and len(discovery) < policy.discovery_cases and len(final) < policy.final_cases:
            left, right = groups[kind].pop(), groups[kind].pop()
            if not admit(left, discovery, final_inputs, discovery_inputs):
                raise ValueError("overlapping compiled relation scenarios")
            if not admit(right, final, discovery_inputs, final_inputs):
                raise ValueError("overlapping compiled relation scenarios")
    # Fill remaining asymmetric budgets in deterministic interleaved order.
    pool, round_number = [], 0
    while any(groups.values()):
        kinds = sorted(groups, reverse=bool(round_number % 2))
        for kind in kinds:
            if groups[kind]:
                pool.append(groups[kind].pop())
        round_number += 1
    if len(pool) + len(discovery) + len(final) < policy.discovery_cases + policy.final_cases:
        raise ValueError("insufficient distinct compiled relations for independent discovery and final budgets")
    for row in pool:
        if len(discovery) < policy.discovery_cases and (len(discovery) <= len(final) or len(final) == policy.final_cases):
            admit(row, discovery, final_inputs, discovery_inputs)
        elif len(final) < policy.final_cases:
            admit(row, final, discovery_inputs, final_inputs)
        if len(discovery) == policy.discovery_cases and len(final) == policy.final_cases:
            break
    if len(discovery) != policy.discovery_cases or len(final) != policy.final_cases:
        raise ValueError("insufficient disjoint relation scenarios for the requested budgets")
    final_hash = fingerprint([row.payload() for row in final])
    return RelationCampaign(tuple(discovery), tuple(final), fingerprint({"generator": RELATION_GENERATOR,
        "spec": spec.model_dump(), "policy": policy.model_dump(), "discovery": [row.payload() for row in discovery],
        "final_fingerprint": final_hash}), final_hash)


def validate_relations(repository, spec, policy):
    validate_session_scope(repository, spec, [seed.scenario for seed in policy.seeds])


def assess_relation(case, traces):
    if case.relation not in {"idempotent_retry", "read_noninterference", "rejected_call_rollback", "commutative_pair"}:
        raise ValueError("unsupported metamorphic law")
    required = 2 if case.relation == "commutative_pair" else 1
    if len(traces) != required or len(traces) != len(case.scenarios) or any(len(trace) != len(scenario.steps)
            for trace, scenario in zip(traces, case.scenarios, strict=True)):
        raise ValueError("incomplete metamorphic evidence")
    violations, observations = [], []
    for branch, trace in enumerate(traces):
        for index, row in enumerate(trace):
            result = row["result"]
            if index == case.error_index:
                if result != {"kind": "error", "type": case.expected_error}:
                    violations.append("expected_rejection_not_observed")
            elif result["kind"] != "value":
                violations.append("operation_or_observation_failed")
        indices = (case.before, case.middle, case.after)
        if any(trace[index]["result"].get("kind") != "value" or trace[index]["result"].get("value") is None for index in indices):
            violations.append("uninformative_observation")
            continue
        before, middle, after = (trace[index]["sha256"] for index in indices)
        if case.relation == "rejected_call_rollback":
            if before != after:
                violations.append("rejected_call_changed_state")
        else:
            if before == middle:
                violations.append("operation_effect_missing")
            if case.relation == "commutative_pair":
                if middle == after:
                    violations.append("second_operation_effect_missing")
            elif middle != after:
                violations.append("retry_not_idempotent" if case.relation == "idempotent_retry" else "read_changed_state")
        observations.append({"branch": branch, "before": before, "middle": middle, "after": after})
    if case.relation == "commutative_pair" and len(observations) == 2 and observations[0]["after"] != observations[1]["after"]:
        violations.append("operation_order_changed_result")
    return {"case_id": case.case_id, "relation": case.relation, "passed": not violations,
        "violations": sorted(set(violations)), "trace_sha256": [fingerprint(trace) for trace in traces],
        "observations": observations, "claim": "owner_declared_law_on_bounded_generated_sequences"}


def relation_inventory(cases):
    return dict(sorted(Counter(row.relation for row in cases).items()))


def relation_reserve(cases):
    return 2 * sum(2 + len(scenario.steps) for row in cases for scenario in row.scenarios)
