"""Budgeted verifier-guided Monte Carlo tree search for agent action planning."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from typing import Literal

from pydantic import BaseModel, Field

from agent.offline_rl import ConservativePlanningPolicy, PlanningState
from agent.process_reward import ProcessRewardScorer, ProcessStep
from agent.verifier_ensemble import EnsembleProcessRewardScorer, RewardEstimate
from agent.world_model import AgentWorldModel

SearchAction = Literal["retrieve", "reason", "verify", "answer", "abstain"]

ACTION_TOKEN_COST: dict[SearchAction, int] = {
    "retrieve": 180,
    "reason": 220,
    "verify": 100,
    "answer": 160,
    "abstain": 16,
}


def _hash(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()
    ).hexdigest()


class SearchPolicy(BaseModel):
    version: str = "verifier-mcts-v1"
    iterations: int = Field(default=96, ge=8, le=2048)
    max_nodes: int = Field(default=128, ge=8, le=4096)
    max_depth: int = Field(default=6, ge=2, le=12)
    exploration_constant: float = Field(default=1.2, ge=0, le=5)
    answer_confidence_floor: float = Field(default=0.62, ge=0, le=1)
    cost_penalty: float = Field(default=0.12, ge=0, le=1)


class SearchRequest(BaseModel):
    request_id: str = Field(min_length=2, max_length=160)
    route: str = Field(min_length=2, max_length=80)
    evidence_count: int = Field(ge=0, le=100)
    confidence: float = Field(ge=0, le=1)
    high_risk: bool = False
    initial_reasoned: bool = False
    initial_verified: bool = False
    retrieval_available: bool = True
    verification_available: bool = True
    max_additional_evidence: int = Field(default=2, ge=0, le=8)
    token_budget: int = Field(default=1200, ge=64, le=100_000)


class SearchState(BaseModel):
    evidence_count: int
    confidence: float
    high_risk: bool
    retrieved: int = 0
    reasoned: bool = False
    verified: bool = False
    tokens_used: int = 0
    actions: list[SearchAction] = Field(default_factory=list)
    steps: list[ProcessStep] = Field(default_factory=list)
    terminal: bool = False
    terminal_action: Literal["answer", "abstain", ""] = ""
    world_model_uncertainty: float = 0.0
    world_model_ood: bool = False
    transition_success_lcb: float = 1.0
    world_model_fingerprint: str = ""


class RootActionStat(BaseModel):
    action: SearchAction
    visits: int = Field(ge=0)
    mean_value: float


class SearchPlan(BaseModel):
    schema_version: str = "1.0"
    policy_version: str
    policy_fingerprint: str
    process_reward_fingerprint: str
    request_fingerprint: str
    planned_actions: list[SearchAction]
    terminal_action: Literal["answer", "abstain"]
    predicted_value: float
    process_reward: float = Field(ge=0, le=1)
    verifier_mean: float = Field(default=0, ge=0, le=1)
    verifier_uncertainty: float = Field(default=0, ge=0, le=1)
    risk_adjusted_reward: float = Field(default=0, ge=0, le=1)
    verifier_ood: bool = False
    dynamics_strategy: Literal["heuristic", "learned_world_model"] = "heuristic"
    world_model_fingerprint: str = ""
    world_model_max_uncertainty: float = Field(default=0, ge=0)
    world_model_ood: bool = False
    transition_success_lcb: float = Field(default=1, ge=0, le=1)
    tree_policy_strategy: Literal["uct", "offline_rl_puct"] = "uct"
    offline_rl_policy_fingerprint: str = ""
    offline_rl_max_uncertainty: float = Field(default=0, ge=0)
    offline_rl_ood: bool = False
    tokens_planned: int = Field(ge=0)
    iterations: int = Field(ge=0)
    nodes_expanded: int = Field(ge=0)
    unique_states: int = Field(ge=0)
    unsafe_branches_pruned: int = Field(ge=0)
    budget_branches_pruned: int = Field(ge=0)
    root_action_stats: list[RootActionStat]
    plan_fingerprint: str = ""

    def seal(self) -> None:
        self.plan_fingerprint = _hash(
            self.model_dump(mode="json", exclude={"plan_fingerprint"})
        )

    def verify(self) -> bool:
        return bool(self.plan_fingerprint) and self.plan_fingerprint == _hash(
            self.model_dump(mode="json", exclude={"plan_fingerprint"})
        )


@dataclass
class _Node:
    state: SearchState
    parent: "_Node | None" = None
    action: SearchAction | None = None
    children: dict[SearchAction, "_Node"] = field(default_factory=dict)
    untried: list[SearchAction] = field(default_factory=list)
    visits: int = 0
    value_sum: float = 0.0
    terminal_value: float | None = None
    process_reward: float = 0.0
    verifier_mean: float = 0.0
    verifier_uncertainty: float = 0.0
    risk_adjusted_reward: float = 0.0
    verifier_ood: bool = False
    action_priors: dict[SearchAction, float] = field(default_factory=dict)

    @property
    def mean_value(self) -> float:
        return self.value_sum / self.visits if self.visits else 0.0


class VerifierGuidedMCTS:
    """MCTS whose terminal value combines PRM score, confidence, safety, and cost."""

    def __init__(
        self,
        scorer: ProcessRewardScorer | EnsembleProcessRewardScorer,
        policy: SearchPolicy | None = None,
        world_model: AgentWorldModel | None = None,
        planning_policy: ConservativePlanningPolicy | None = None,
    ) -> None:
        self.scorer = scorer
        self.policy = policy or SearchPolicy()
        self.world_model = world_model
        self.planning_policy = planning_policy
        self._unsafe_pruned = 0
        self._budget_pruned = 0
        self._reward_cache: dict[str, RewardEstimate] = {}
        self._offline_rl_max_uncertainty = 0.0
        self._offline_rl_ood = False

    def _action_priors(
        self,
        state: SearchState,
        request: SearchRequest,
        actions: list[SearchAction],
    ) -> dict[SearchAction, float]:
        if not actions:
            return {}
        if self.planning_policy is None:
            return {action: 1 / len(actions) for action in actions}
        estimate = self.planning_policy.priors(
            route=request.route,
            high_risk=state.high_risk,
            state=PlanningState(
                evidence_count=state.evidence_count,
                confidence=state.confidence,
                reasoned=state.reasoned,
                verified=state.verified,
            ),
            actions=actions,
        )
        self._offline_rl_max_uncertainty = max(
            self._offline_rl_max_uncertainty, estimate.max_uncertainty
        )
        self._offline_rl_ood = self._offline_rl_ood or estimate.out_of_distribution
        return {item.action: item.probability for item in estimate.actions}

    @staticmethod
    def _ordered_actions(
        actions: list[SearchAction], priors: dict[SearchAction, float]
    ) -> list[SearchAction]:
        return sorted(actions, key=lambda action: (-priors.get(action, 0.0), action))

    def _answer_safe(self, state: SearchState) -> bool:
        required_evidence = 2 if state.high_risk else 1
        return (
            state.reasoned
            and state.verified
            and state.evidence_count >= required_evidence
            and state.confidence >= self.policy.answer_confidence_floor
            and not state.world_model_ood
            and (
                self.world_model is None
                or state.transition_success_lcb >= self.world_model.artifact.success_floor
            )
        )

    def _valid_actions(self, state: SearchState, request: SearchRequest) -> list[SearchAction]:
        if state.terminal or len(state.actions) >= self.policy.max_depth:
            return []
        if state.world_model_ood:
            return (
                ["abstain"]
                if state.tokens_used + ACTION_TOKEN_COST["abstain"] <= request.token_budget
                else []
            )
        actions: list[SearchAction] = []
        if (
            request.retrieval_available
            and not state.reasoned
            and state.retrieved < request.max_additional_evidence
            and state.tokens_used + ACTION_TOKEN_COST["retrieve"] <= request.token_budget
        ):
            actions.append("retrieve")
        elif (
            request.retrieval_available
            and not state.reasoned
            and state.retrieved < request.max_additional_evidence
        ):
            self._budget_pruned += 1
        if (
            state.evidence_count > 0
            and not state.reasoned
            and state.tokens_used + ACTION_TOKEN_COST["reason"] <= request.token_budget
        ):
            actions.append("reason")
        elif state.evidence_count > 0 and not state.reasoned:
            self._budget_pruned += 1
        if (
            request.verification_available
            and state.reasoned
            and not state.verified
            and state.tokens_used + ACTION_TOKEN_COST["verify"] <= request.token_budget
        ):
            actions.append("verify")
        elif request.verification_available and state.reasoned and not state.verified:
            self._budget_pruned += 1
        if self._answer_safe(state):
            if state.tokens_used + ACTION_TOKEN_COST["answer"] <= request.token_budget:
                actions.append("answer")
            else:
                self._budget_pruned += 1
        elif state.reasoned:
            self._unsafe_pruned += 1
        if state.tokens_used + ACTION_TOKEN_COST["abstain"] <= request.token_budget:
            actions.append("abstain")
        return actions

    def _transition(
        self, state: SearchState, action: SearchAction, request: SearchRequest
    ) -> SearchState:
        updated = state.model_copy(deep=True)
        updated.actions.append(action)
        updated.tokens_used += ACTION_TOKEN_COST[action]
        confidence_delta = {
            "retrieve": 0.08,
            "reason": 0.1,
            "verify": 0.08,
            "answer": 0.0,
            "abstain": 0.0,
        }[action]
        evidence_delta = 1 if action == "retrieve" else 0
        if self.world_model is not None:
            prediction = self.world_model.predict(
                route=request.route,
                action=action,
                high_risk=state.high_risk,
                evidence_count=state.evidence_count,
                confidence=state.confidence,
                reasoned=state.reasoned,
                verified=state.verified,
            )
            confidence_delta = prediction.confidence_delta_lcb
            evidence_delta = max(0, round(prediction.evidence_delta_lcb))
            updated.world_model_uncertainty = max(
                state.world_model_uncertainty, prediction.epistemic_uncertainty
            )
            updated.world_model_ood = state.world_model_ood or prediction.out_of_distribution
            updated.transition_success_lcb = min(
                state.transition_success_lcb, prediction.success_lcb
            )
            updated.world_model_fingerprint = prediction.model_fingerprint
        updated.confidence = max(
            0.0, min(1.0, updated.confidence + confidence_delta)
        )
        updated.evidence_count = max(0, updated.evidence_count + evidence_delta)
        if action == "retrieve":
            updated.retrieved += 1
            updated.steps.append(
                ProcessStep(
                    step_id=f"retrieve-{updated.retrieved}",
                    kind="retrieve",
                    has_evidence=True,
                    confidence=updated.confidence,
                    tokens=ACTION_TOKEN_COST[action],
                )
            )
        elif action == "reason":
            updated.reasoned = True
            updated.steps.append(
                ProcessStep(
                    step_id="reason",
                    kind="reason",
                    has_evidence=updated.evidence_count > 0,
                    confidence=updated.confidence,
                    tokens=ACTION_TOKEN_COST[action],
                )
            )
        elif action == "verify":
            updated.verified = True
            updated.steps.append(
                ProcessStep(
                    step_id="verify",
                    kind="verify",
                    has_evidence=updated.evidence_count > 0,
                    citation_valid=True,
                    confidence=updated.confidence,
                    tokens=ACTION_TOKEN_COST[action],
                )
            )
        elif action == "answer":
            updated.terminal = True
            updated.terminal_action = "answer"
            updated.steps.append(
                ProcessStep(
                    step_id="answer",
                    kind="answer",
                    has_evidence=updated.evidence_count > 0,
                    citation_valid=updated.verified,
                    confidence=updated.confidence,
                    tokens=ACTION_TOKEN_COST[action],
                )
            )
        else:
            updated.terminal = True
            updated.terminal_action = "abstain"
            updated.steps.append(
                ProcessStep(
                    step_id="abstain",
                    kind="refuse",
                    policy_allowed=True,
                    confidence=max(0.5, 1 - updated.confidence),
                    tokens=ACTION_TOKEN_COST[action],
                )
            )
        return updated

    def _rollout(self, state: SearchState, request: SearchRequest) -> SearchState:
        simulated = state.model_copy(deep=True)
        while not simulated.terminal and len(simulated.actions) < self.policy.max_depth:
            actions = self._valid_actions(simulated, request)
            if not actions:
                break
            preferred = next(
                (
                    action
                    for action in ("retrieve", "reason", "verify", "answer", "abstain")
                    if action in actions
                    and not (action == "retrieve" and simulated.evidence_count >= (2 if simulated.high_risk else 1))
                ),
                actions[0],
            )
            simulated = self._transition(simulated, preferred, request)
        if not simulated.terminal:
            actions = self._valid_actions(simulated, request)
            if "abstain" in actions:
                simulated = self._transition(simulated, "abstain", request)
        return simulated

    def _reward_estimate(self, state: SearchState) -> RewardEstimate:
        cache_key = _hash(
            {
                "steps": [step.model_dump(mode="json") for step in state.steps],
                "high_risk": state.high_risk,
            }
        )
        if cache_key in self._reward_cache:
            return self._reward_cache[cache_key]
        if isinstance(self.scorer, EnsembleProcessRewardScorer):
            estimate = self.scorer.score_steps_with_uncertainty(
                state.steps, state.high_risk
            )
        else:
            score = self.scorer.score_steps(state.steps, state.high_risk)
            estimate = RewardEstimate(
                mean=score,
                standard_deviation=0,
                lower_confidence_bound=score,
                out_of_distribution=False,
                member_scores=[score],
            )
        self._reward_cache[cache_key] = estimate
        return estimate

    def _terminal_value(
        self, state: SearchState, request: SearchRequest
    ) -> tuple[float, RewardEstimate]:
        estimate = self._reward_estimate(state)
        cost = self.policy.cost_penalty * state.tokens_used / request.token_budget
        if state.terminal_action == "answer":
            if not self._answer_safe(state) or estimate.out_of_distribution:
                return -1.0, estimate
            value = (
                0.65 * estimate.lower_confidence_bound
                + 0.35 * state.confidence
                - cost
            )
            return value, estimate
        # Abstention is preferable to an unsafe answer, but worse than a supported answer.
        return 0.12 - cost, estimate

    def _select_child(self, node: _Node) -> _Node:
        if self.planning_policy is not None:
            return max(
                node.children.values(),
                key=lambda child: (
                    child.mean_value
                    + self.policy.exploration_constant
                    * node.action_priors.get(child.action or "abstain", 0.0)
                    * math.sqrt(max(1, node.visits))
                    / (1 + child.visits),
                    str(child.action),
                ),
            )
        log_parent = math.log(max(1, node.visits))
        return max(
            node.children.values(),
            key=lambda child: (
                child.mean_value
                + self.policy.exploration_constant
                * math.sqrt(log_parent / max(1, child.visits)),
                str(child.action),
            ),
        )

    def plan(self, request: SearchRequest) -> SearchPlan:
        self._unsafe_pruned = 0
        self._budget_pruned = 0
        self._reward_cache = {}
        self._offline_rl_max_uncertainty = 0.0
        self._offline_rl_ood = False
        root_state = SearchState(
            evidence_count=request.evidence_count,
            confidence=request.confidence,
            high_risk=request.high_risk,
            reasoned=request.initial_reasoned,
            verified=request.initial_verified,
        )
        root = _Node(state=root_state)
        root_actions = self._valid_actions(root.state, request)
        root.action_priors = self._action_priors(root.state, request, root_actions)
        root.untried = self._ordered_actions(root_actions, root.action_priors)
        nodes = [root]
        terminal_nodes: list[_Node] = []
        unique_states = {_hash(root.state.model_dump(mode="json"))}
        completed_iterations = 0
        for _ in range(self.policy.iterations):
            if len(nodes) >= self.policy.max_nodes:
                break
            node = root
            while not node.untried and node.children and not node.state.terminal:
                node = self._select_child(node)
            if node.untried and len(nodes) < self.policy.max_nodes:
                action = node.untried.pop(0)
                child_state = self._transition(node.state, action, request)
                child = _Node(state=child_state, parent=node, action=action)
                child_actions = self._valid_actions(child_state, request)
                child.action_priors = self._action_priors(
                    child_state, request, child_actions
                )
                child.untried = self._ordered_actions(
                    child_actions, child.action_priors
                )
                node.children[action] = child
                nodes.append(child)
                unique_states.add(_hash(child_state.model_dump(mode="json")))
                node = child
            rollout = self._rollout(node.state, request)
            value, estimate = self._terminal_value(rollout, request)
            if rollout.terminal:
                terminal = _Node(
                    state=rollout,
                    parent=node.parent,
                    action=node.action,
                    visits=1,
                    value_sum=value,
                    terminal_value=value,
                    process_reward=estimate.mean,
                    verifier_mean=estimate.mean,
                    verifier_uncertainty=estimate.standard_deviation,
                    risk_adjusted_reward=estimate.lower_confidence_bound,
                    verifier_ood=estimate.out_of_distribution,
                )
                terminal_nodes.append(terminal)
                unique_states.add(_hash(rollout.model_dump(mode="json")))
            cursor: _Node | None = node
            while cursor is not None:
                cursor.visits += 1
                cursor.value_sum += value
                cursor = cursor.parent
            completed_iterations += 1
        if not terminal_nodes:
            fallback = self._transition(root_state, "abstain", request)
            value, estimate = self._terminal_value(fallback, request)
            terminal_nodes.append(
                _Node(
                    state=fallback,
                    visits=1,
                    value_sum=value,
                    terminal_value=value,
                    process_reward=estimate.mean,
                    verifier_mean=estimate.mean,
                    verifier_uncertainty=estimate.standard_deviation,
                    risk_adjusted_reward=estimate.lower_confidence_bound,
                    verifier_ood=estimate.out_of_distribution,
                )
            )
        best = max(
            terminal_nodes,
            key=lambda item: (
                item.terminal_value if item.terminal_value is not None else -math.inf,
                -item.state.tokens_used,
                tuple(item.state.actions),
            ),
        )
        policy_payload = self.policy.model_dump(mode="json")
        plan = SearchPlan(
            policy_version=self.policy.version,
            policy_fingerprint=_hash(policy_payload),
            process_reward_fingerprint=self.scorer.artifact.artifact_fingerprint,
            request_fingerprint=_hash(request.model_dump(mode="json")),
            planned_actions=best.state.actions,
            terminal_action=best.state.terminal_action or "abstain",
            predicted_value=best.terminal_value or 0.0,
            process_reward=best.process_reward,
            verifier_mean=best.verifier_mean,
            verifier_uncertainty=best.verifier_uncertainty,
            risk_adjusted_reward=best.risk_adjusted_reward,
            verifier_ood=best.verifier_ood,
            dynamics_strategy=(
                "learned_world_model" if self.world_model is not None else "heuristic"
            ),
            world_model_fingerprint=best.state.world_model_fingerprint,
            world_model_max_uncertainty=best.state.world_model_uncertainty,
            world_model_ood=best.state.world_model_ood,
            transition_success_lcb=best.state.transition_success_lcb,
            tree_policy_strategy=(
                "offline_rl_puct" if self.planning_policy is not None else "uct"
            ),
            offline_rl_policy_fingerprint=(
                self.planning_policy.artifact.artifact_fingerprint
                if self.planning_policy is not None
                else ""
            ),
            offline_rl_max_uncertainty=self._offline_rl_max_uncertainty,
            offline_rl_ood=self._offline_rl_ood,
            tokens_planned=best.state.tokens_used,
            iterations=completed_iterations,
            nodes_expanded=len(nodes),
            unique_states=len(unique_states),
            unsafe_branches_pruned=self._unsafe_pruned,
            budget_branches_pruned=self._budget_pruned,
            root_action_stats=[
                RootActionStat(action=action, visits=child.visits, mean_value=child.mean_value)
                for action, child in sorted(root.children.items())
            ],
        )
        plan.seal()
        return plan
