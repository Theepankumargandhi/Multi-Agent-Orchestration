from pathlib import Path

import pytest

from evals.adaptive_router import (
    CostAwareRouter,
    RoutingObservation,
    evaluate_router,
    train_router,
    tune_threshold,
)
from evals.build_seed_dataset import build_cases
from evals.calibration import JudgeDecision, PairwiseCase, calibrate_pairwise_judge
from evals.retrieval_metrics import RetrievalCase, evaluate_rankings, reciprocal_rank_fusion
from evals.review_dataset import export_review_sheet, import_review_sheet


class ContentJudge:
    """Deterministic fake that lets calibration behavior be tested without an API key."""

    async def judge(self, case: PairwiseCase) -> JudgeDecision:
        if "supported" in case.response_a.lower() and "supported" not in case.response_b.lower():
            preference = "A"
        elif "supported" in case.response_b.lower() and "supported" not in case.response_a.lower():
            preference = "B"
        else:
            preference = "tie"
        return JudgeDecision(preference=preference, confidence=0.9, rationale="fixture rubric")


@pytest.mark.asyncio
async def test_pairwise_calibration_checks_accuracy_and_position_bias():
    cases = [
        PairwiseCase(
            id="one",
            input="question",
            response_a="A supported response",
            response_b="A weak response",
            human_preference="A",
            human_reviewed=True,
        ),
        PairwiseCase(
            id="two",
            input="question",
            response_a="weak",
            response_b="supported evidence",
            human_preference="B",
            human_reviewed=True,
        ),
    ]
    report = await calibrate_pairwise_judge(ContentJudge(), cases)
    assert report.accuracy == 1.0
    assert report.position_consistency == 1.0
    assert report.high_confidence_error_rate == 0.0


def test_retrieval_metrics_and_reciprocal_rank_fusion():
    cases = [
        RetrievalCase(id="q1", query="alpha", relevant_ids={"d1"}, tags=["easy"]),
        RetrievalCase(id="q2", query="beta", relevant_ids={"d2", "d3"}, tags=["multi-hop"]),
    ]
    rankings = {"q1": ["d1", "d9"], "q2": ["d8", "d2", "d3"]}
    report = evaluate_rankings(cases, rankings, k=2)
    assert report.hit_rate == 1.0
    assert report.recall_at_k == 0.75
    assert report.mrr == 0.75
    assert set(report.slices) == {"easy", "multi-hop"}
    assert reciprocal_rank_fusion([["a", "b"], ["b", "c"]])[0] == "b"


def _observations() -> list[RoutingObservation]:
    return [
        RoutingObservation(
            id="simple-1",
            query="hello",
            small_success=True,
            strong_success=True,
            small_cost_usd=0.001,
            strong_cost_usd=0.01,
            split="train",
        ),
        RoutingObservation(
            id="simple-2",
            query="calculate 2 + 2",
            small_success=True,
            strong_success=True,
            small_cost_usd=0.001,
            strong_cost_usd=0.01,
            split="train",
        ),
        RoutingObservation(
            id="hard-1",
            query="analyze the multi-step architecture and dependency impact in this codebase",
            small_success=False,
            strong_success=True,
            small_cost_usd=0.001,
            strong_cost_usd=0.01,
            split="train",
        ),
        RoutingObservation(
            id="risk-1",
            query="review this security credential deletion plan",
            small_success=False,
            strong_success=True,
            small_cost_usd=0.001,
            strong_cost_usd=0.01,
            split="train",
            high_risk=True,
        ),
        RoutingObservation(
            id="simple-v",
            query="say hello",
            small_success=True,
            strong_success=True,
            small_cost_usd=0.001,
            strong_cost_usd=0.01,
            split="validation",
        ),
        RoutingObservation(
            id="hard-v",
            query="trace the multi-step API database dependency impact",
            small_success=False,
            strong_success=True,
            small_cost_usd=0.001,
            strong_cost_usd=0.01,
            split="validation",
        ),
    ]


def test_learned_cost_router_tunes_threshold_and_round_trips(tmp_path: Path):
    observations = _observations()
    router = train_router(observations)
    validation = [item for item in observations if item.split == "validation"]
    tuned = tune_threshold(router, validation, minimum_success_rate=1.0)
    assert tuned.success_rate == 1.0
    assert tuned.cost_reduction > 0
    assert router.select("delete production credentials", high_risk=True) == "strong"

    path = tmp_path / "router.json"
    router.save(path)
    restored = CostAwareRouter.load(path)
    assert restored.training_fingerprint == router.training_fingerprint
    assert evaluate_router(restored, validation).success_rate == 1.0


def test_human_review_round_trip_preserves_audit_metadata(tmp_path: Path):
    dataset = tmp_path / "dataset.jsonl"
    dataset.write_text(
        '{"id":"one","input":"hello","expected":{"route":"general"}}\n',
        encoding="utf-8",
    )
    sheet = tmp_path / "review.csv"
    assert export_review_sheet(dataset, sheet) == 1
    contents = sheet.read_text(encoding="utf-8-sig")
    contents = contents.replace(",,,\n", ",approve,reviewer@example.com,looks correct\n")
    sheet.write_text(contents, encoding="utf-8-sig")
    reviewed = tmp_path / "reviewed.jsonl"
    summary = import_review_sheet(dataset, sheet, reviewed)
    assert summary == {"approved": 1, "rejected": 0, "unreviewed": 0}
    payload = reviewed.read_text(encoding="utf-8")
    assert '"review_status":"human_reviewed"' in payload
    assert "reviewer@example.com" in payload


def test_seed_dataset_is_large_split_and_explicitly_unreviewed():
    cases = build_cases()
    assert len(cases) == 120
    assert {case.split for case in cases} == {"train", "validation", "test"}
    assert {case.review_status for case in cases} == {"synthetic_seed"}
    assert sum("safety" in case.tags for case in cases) == 20
