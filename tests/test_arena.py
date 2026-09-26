import json
from pathlib import Path

import pytest

from evals.arena import (
    ArenaConfig,
    ArenaReport,
    DeterministicPolicyAgent,
    JudgeVerdict,
    _hash,
    load_scenarios,
    replay_receipt,
    run_arena,
    run_scenario,
    verify_report,
)

DATASET = Path("evals/datasets/agent_arena_scenarios.jsonl")
CONFIG = Path("evals/experiments/agent_arena.json")


def _config() -> ArenaConfig:
    return ArenaConfig.model_validate_json(CONFIG.read_text(encoding="utf-8"))


@pytest.mark.asyncio
async def test_arena_dataset_and_promotion_are_reproducible():
    scenarios = load_scenarios(DATASET)
    report = await run_arena(_config(), scenarios)
    variants = {item.variant.name: item for item in report.variants}

    assert len(scenarios) == 10
    assert _hash([item.model_dump(mode="json") for item in scenarios]) == (
        "afc0efb84ba3820ff6f25724574d11dfdc8b0815d4b40170638796a7ffb13826"
    )
    assert variants["naive-policy-v1"].pass_rate == 0.3
    assert variants["resilient-policy-v2"].pass_rate == 1.0
    assert variants["resilient-policy-v2"].safety_failure_rate == 0
    assert report.standings[0].variant == "resilient-policy-v2"
    assert report.promotion.approved
    assert verify_report(report)
    repeated = await run_arena(_config(), scenarios)
    assert repeated.arena_id == report.arena_id
    assert repeated.report_fingerprint == report.report_fingerprint


@pytest.mark.asyncio
async def test_adaptive_adversary_follows_the_agents_previous_action():
    scenario = next(
        item for item in load_scenarios(DATASET) if item.id == "adaptive-system-prompt-attack"
    )
    config = _config()
    variants = {item.policy_version: item for item in config.variants}
    agent = DeterministicPolicyAgent()

    resilient = await run_scenario(
        scenario, variants["resilient"], agent, pass_threshold=config.pass_threshold
    )
    baseline = await run_scenario(
        scenario, variants["baseline"], agent, pass_threshold=config.pass_threshold
    )

    assert [item.action for item in resilient.transcript] == ["refuse", "refuse"]
    assert [item.turn for item in resilient.transcript] == [1, 2]
    assert [item.action for item in baseline.transcript] == ["answer", "answer"]
    assert [item.turn for item in baseline.transcript] == [1, 2]
    assert resilient.passed and not resilient.safety_failed
    assert not baseline.passed and baseline.safety_failed


@pytest.mark.asyncio
async def test_cross_turn_memory_and_tool_minimality_are_graded():
    scenario = next(item for item in load_scenarios(DATASET) if item.id == "cross-turn-memory")
    config = _config()
    candidate = next(item for item in config.variants if item.name == config.candidate)

    outcome = await run_scenario(
        scenario,
        candidate,
        DeterministicPolicyAgent(),
        pass_threshold=config.pass_threshold,
    )

    assert outcome.transcript[-1].response.endswith("Orion.")
    assert all(not item.tool_calls for item in outcome.transcript)
    assert outcome.metrics["memory_consistency"].score == 1
    assert outcome.metrics["tool_f1"].score == 1


@pytest.mark.asyncio
async def test_external_judge_is_recorded_as_advisory_evidence():
    class FixedJudge:
        async def judge(self, scenario, outcome):
            return JudgeVerdict(
                judge="calibrated-reviewer",
                judge_type="llm",
                score=0.75,
                confidence=0.8,
                passed=False,
                rationale=f"Advisory review for {scenario.id} with {len(outcome.transcript)} turn(s).",
            )

    scenario = next(item for item in load_scenarios(DATASET) if item.id == "math-then-close")
    config = _config()
    candidate = next(item for item in config.variants if item.name == config.candidate)
    outcome = await run_scenario(
        scenario,
        candidate,
        DeterministicPolicyAgent(),
        pass_threshold=config.pass_threshold,
        external_judges=[FixedJudge()],
    )

    assert len(outcome.judges) == 4
    assert outcome.judges[-1].judge_type == "llm"
    assert outcome.judge_disagreement > 0
    assert outcome.passed


@pytest.mark.asyncio
async def test_tampered_trajectory_or_promotion_receipt_fails_verification():
    report = await run_arena(_config(), load_scenarios(DATASET))
    trajectory_tamper = ArenaReport.model_validate(report.model_dump())
    trajectory_tamper.variants[0].outcomes[0].transcript[0].response = "modified"
    assert not verify_report(trajectory_tamper)

    promotion_tamper = ArenaReport.model_validate(report.model_dump())
    promotion_tamper.promotion.approved = not promotion_tamper.promotion.approved
    assert not verify_report(promotion_tamper)


@pytest.mark.asyncio
async def test_replay_receipt_is_content_addressed_and_rejects_unknown_cases():
    report = await run_arena(_config(), load_scenarios(DATASET))
    receipt = replay_receipt(report, "bounded-coding-delegation")
    assert receipt["report_fingerprint"] == report.report_fingerprint
    assert set(receipt["variants"]) == {"naive-policy-v1", "resilient-policy-v2"}
    assert all(len(item["trajectory_fingerprint"]) == 64 for item in receipt["variants"].values())
    with pytest.raises(KeyError, match="unknown arena scenario"):
        replay_receipt(report, "does-not-exist")


@pytest.mark.asyncio
async def test_arena_report_json_round_trip_is_verifiable(tmp_path):
    original = await run_arena(_config(), load_scenarios(DATASET))
    output = tmp_path / "arena.json"
    output.write_text(original.model_dump_json(indent=2), encoding="utf-8")
    payload = json.loads(output.read_text(encoding="utf-8"))
    restored = ArenaReport.model_validate(payload)
    assert restored.score_source == "deterministic_simulation"
    assert verify_report(restored)
