import pytest

from evals import structured_patch_evaluation as controls


@pytest.mark.asyncio
@pytest.mark.parametrize("trial_count", [0, 4])
async def test_missing_execution_or_wrong_inconclusive_reason_never_passes_a_control(
    monkeypatch, trial_count
):
    async def unavailable(*args, **kwargs):
        return {
            "status": "inconclusive",
            "reason": "docker_unavailable",
            "source_unchanged": True,
            "trials": [{"tests": {"passed": True}} for _ in range(trial_count)],
            "minimizations": [],
            "counterexamples": [],
        }

    monkeypatch.setattr(controls, "verify_patch", unavailable)
    receipts = []
    report = await controls.evaluate(receipts.append)
    assert report["total"] == 6 and report["passed"] == 0 and len(receipts) == 6
    assert not any(row["passed"] for row in receipts)
