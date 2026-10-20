"""Failure injection for recovery/accounting. Simulated usage is not a real provider measurement."""
import asyncio
import json
from pathlib import Path

import pytest

from code_agent.upgrade_cli import private_run_path
from code_agent.upgrade_recovery import MigrationJournal, StudyRunBudget
from code_agent.upgrade_study import run_study
from tests.test_upgrade_study import migration_report, setup_study


@pytest.fixture
def harness(tmp_path, monkeypatch):
    _request, repository, frozen = setup_study(tmp_path, monkeypatch)
    calls, responses = [], []

    async def execute(_repository, spec, **kwargs):
        calls.append(kwargs)
        # The RUNNING marker must exist before even this fake executor is entered.
        state = json.loads((tmp_path / "data/upgradepilot/recovery/checkpoint.json").read_bytes())["state"]
        assert any(attempt["status"] == "running" for rows in state["attempts"].values() for attempt in rows)
        if responses:
            selected = responses.pop(0)
            if isinstance(selected, BaseException):
                raise selected
            return migration_report(spec, **selected)
        count = 2 if kwargs["model"] else 0
        return migration_report(spec, model_calls=count, live_provider_calls=count)
    monkeypatch.setattr("code_agent.upgrade_study.run_migration", execute)
    return {"frozen": frozen, "root": tmp_path, "repository": repository,
        "output": tmp_path / "data/upgradepilot/recovery", "calls": calls, "responses": responses}


async def run(h, **kwargs):
    return await run_study(h["frozen"], h["root"], h["output"], authorize_provider=True, **kwargs)


def summary(h):
    return json.loads((h["output"] / "study.json").read_bytes())


@pytest.mark.asyncio
async def test_completed_quality_holds_are_reused_without_cherry_picking(harness):
    first = await run(harness)
    again = await run(harness, resume=True)
    assert len(harness["calls"]) == 3 and first["outcomes"] == again["outcomes"]
    assert again["complete"] and again["quality_metrics_valid"]
    assert all(row["status"] == "held" for row in again["outcomes"])
    assert set(again["recovery_dispositions"].values()) == {"reused"}
    assert again["resource_accounting"]["known_provider_calls"] == 4
    assert again["arm_metrics"]["existing_repair"]["total_billed_cost_usd"] is None


@pytest.mark.asyncio
async def test_interruption_requires_retry_and_preserves_unknown_reservation(harness):
    harness["responses"].extend([{}, asyncio.CancelledError()])
    budget = StudyRunBudget(max_attempts=4, max_provider_calls=48)
    with pytest.raises(asyncio.CancelledError):
        await run(harness, budget=budget)
    first = summary(harness)
    assert first["recorded_attempts"] == 1 and first["incomplete_attempt_usage"] == "unknown_not_zero"
    assert first["resource_accounting"]["unresolved_provider_call_reservations"] == 16
    with pytest.raises(ValueError, match="retry-interrupted"):
        await run(harness, resume=True)
    assert len(harness["calls"]) == 2
    completed = await run(harness, resume=True, retry_interrupted=True)
    assert len(harness["calls"]) == 4 and completed["complete"]
    assert not completed["quality_metrics_valid"]
    assert completed["resource_accounting"] == {"admitted_attempts": 4, "known_provider_calls": 4,
        "interrupted_attempts": 1, "unresolved_provider_call_reservations": 16, "provider_call_upper_bound_used": 20}
    assert completed["arm_metrics"]["existing_repair"]["billed_cost_per_review_ready_task_usd"] is None
    await run(harness, resume=True)
    assert len(harness["calls"]) == 4


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["calls", "attempts"])
async def test_budget_blocks_before_another_executor_is_entered(harness, kind):
    budget = StudyRunBudget(max_attempts=1 if kind == "attempts" else 3,
        max_provider_calls=100 if kind == "attempts" else 15)
    with pytest.raises(ValueError, match="budget exhausted"):
        await run(harness, budget=budget)
    assert len(harness["calls"]) == 1
    state = summary(harness)
    assert state["recorded_attempts"] == 1 and not state["complete"]
    assert state["resource_accounting"]["admitted_attempts"] == 1
    assert state["incomplete_attempt_usage"] is None  # No request was admitted for the blocked arm.


@pytest.mark.asyncio
async def test_interrupted_usage_is_not_released_to_fund_its_retry(harness):
    harness["responses"].extend([{}, RuntimeError("injected uncertain return")])
    with pytest.raises(RuntimeError):
        await run(harness, budget=StudyRunBudget(max_attempts=4, max_provider_calls=16))
    with pytest.raises(ValueError, match="budget exhausted"):
        await run(harness, resume=True, retry_interrupted=True)
    assert len(harness["calls"]) == 2
    assert summary(harness)["resource_accounting"]["provider_call_upper_bound_used"] == 16


@pytest.mark.asyncio
async def test_resume_uses_stored_budget_and_rejects_silent_limit_changes(harness):
    budget = StudyRunBudget(max_attempts=4, max_provider_calls=64)
    await run(harness, budget=budget)
    assert (await run(harness, resume=True))["run_budget"] == budget.model_dump()
    with pytest.raises(ValueError, match="budget changed"):
        await run(harness, resume=True, budget=StudyRunBudget(max_attempts=5, max_provider_calls=64))
    assert len(harness["calls"]) == 3


@pytest.mark.asyncio
@pytest.mark.parametrize("ready", [False, True])
async def test_complete_saved_receipt_is_adopted_without_reexecution(harness, monkeypatch, ready):
    original = MigrationJournal.finish_report
    if ready:
        harness["responses"].append({"status": "ready_for_owner_review",
            "patch": "--- a/requirements.txt\n+++ b/requirements.txt\n-old\r\n+new\r\n"})

    def crash_before_commit(self, key, report):
        raise RuntimeError("injected crash after fsynced artifacts, before journal commit")
    monkeypatch.setattr(MigrationJournal, "finish_report", crash_before_commit)
    with pytest.raises(RuntimeError):
        await run(harness)
    assert len(harness["calls"]) == 1
    monkeypatch.setattr(MigrationJournal, "finish_report", original)
    recovered = await run(harness, resume=True)
    assert recovered["complete"] and recovered["quality_metrics_valid"]
    assert len(harness["calls"]) == 3  # Only the two remaining arms were executed.
    assert "adopted_saved_receipt" in recovered["recovery_dispositions"].values()
    assert recovered["outcomes"][0]["status"] == ("ready_for_owner_review" if ready else "held")


@pytest.mark.asyncio
async def test_partial_receipt_needs_explicit_retry_and_is_preserved(harness):
    harness["responses"].append(RuntimeError("injected crash"))
    with pytest.raises(RuntimeError):
        await run(harness, budget=StudyRunBudget(max_attempts=4, max_provider_calls=64))
    partial = harness["output"] / "attempts/1/report.json"
    partial.write_bytes(b'{"unfinished":')
    with pytest.raises(ValueError, match="retry-interrupted"):
        await run(harness, resume=True)
    assert len(harness["calls"]) == 1
    completed = await run(harness, resume=True, retry_interrupted=True)
    assert completed["complete"] and not completed["quality_metrics_valid"]
    assert partial.read_bytes() == b'{"unfinished":'


@pytest.mark.asyncio
async def test_completed_receipt_tampering_blocks_even_explicit_retry(harness):
    await run(harness)
    path = harness["output"] / "attempts/1/report.json"
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError, match="artifact integrity"):
        await run(harness, resume=True, retry_interrupted=True)
    assert len(harness["calls"]) == 3


@pytest.mark.asyncio
async def test_non_latest_completed_artifact_is_also_verified(harness):
    harness["responses"].extend([{}, RuntimeError("interrupted model arm")])
    with pytest.raises(RuntimeError):
        await run(harness, budget=StudyRunBudget(max_attempts=4, max_provider_calls=64))
    await run(harness, resume=True, retry_interrupted=True)
    path = harness["output"] / "attempts/1/report.json"
    path.write_text("tampered historical held receipt")
    with pytest.raises(ValueError, match="artifact integrity"):
        await run(harness, resume=True)
    assert len(harness["calls"]) == 4


@pytest.mark.asyncio
@pytest.mark.parametrize("change", ["runtime", "endpoint"])
async def test_runtime_or_endpoint_drift_blocks_before_inference(harness, monkeypatch, change):
    await run(harness)
    if change == "runtime":
        monkeypatch.setattr("code_agent.upgrade_study.checkpoint_scope", lambda *_args: "changed-runtime")
    else:
        monkeypatch.setenv("GROQ_API_BASE", "https://offline.invalid/not-contacted")
    with pytest.raises(ValueError, match="binding"):
        await run(harness, resume=True)
    assert len(harness["calls"]) == 3


@pytest.mark.asyncio
async def test_source_integrity_failure_is_sticky_even_if_files_appear_restored(harness):
    harness["responses"].append({"source_unchanged": False})
    with pytest.raises(ValueError, match="source changed"):
        await run(harness)
    with pytest.raises(ValueError, match="admissions revoked"):
        await run(harness, resume=True)
    assert len(harness["calls"]) == 1


@pytest.mark.asyncio
async def test_exclusive_lock_blocks_another_run_without_executor_calls(harness):
    await run(harness)
    with MigrationJournal(harness["output"], resume=True):
        with pytest.raises(ValueError, match="already in use"):
            await run(harness, resume=True)
    assert len(harness["calls"]) == 3


@pytest.mark.asyncio
@pytest.mark.parametrize("bad", [{"model_calls": True}, {"live_provider_calls": 99}, {"task_id": "other-task"}])
async def test_receipt_shape_and_usage_cannot_release_reserved_calls(harness, bad):
    harness["responses"].append(bad)
    with pytest.raises(ValueError, match="identity|accounting"):
        await run(harness)
    assert len(harness["calls"]) == 1
    with pytest.raises(ValueError, match="retry-interrupted"):
        await run(harness, resume=True)


@pytest.mark.asyncio
async def test_resume_does_not_overwrite_existing_output_or_accept_retry_without_resume(harness):
    with pytest.raises(ValueError, match="requires --resume"):
        await run(harness, retry_interrupted=True)
    assert not harness["output"].exists()
    await run(harness)
    with pytest.raises(FileExistsError):
        await run(harness)
    assert len(harness["calls"]) == 3


def test_private_path_validation_rejects_links_without_creating_output(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(Path, "is_symlink", lambda path: path.name == "linked")
    with pytest.raises(ValueError, match="symlinks"):
        private_run_path(tmp_path / "data/upgradepilot/linked/new-run")
    assert not (tmp_path / "data").exists()


@pytest.mark.asyncio
async def test_initial_metadata_is_durable_before_configuration_or_execution(harness, monkeypatch):
    original = MigrationJournal.configure
    checked = []

    def inspect(self, entries, budget):
        state = json.loads(self.safe_path("checkpoint.json").read_bytes())["state"]
        assert state["migration_protocol"] == "migration-arm-journal-v1"
        assert state["run_budget"] == {"max_attempts": 3, "max_provider_calls": 32}
        assert not state["attempts"]
        checked.append(True)
        return original(self, entries, budget)
    monkeypatch.setattr(MigrationJournal, "configure", inspect)
    await run(harness)
    assert checked == [True]


@pytest.mark.asyncio
async def test_last_arm_source_failure_cannot_mark_quality_metrics_valid(harness):
    harness["responses"].extend([{}, {}, {"source_unchanged": False}])
    with pytest.raises(ValueError, match="source changed"):
        await run(harness)
    report = summary(harness)
    assert report["complete"] and report["admissions_revoked"] and not report["quality_metrics_valid"]
    assert not report["arm_metrics"]["counterexample_repair"]["quality_metrics_valid"]


@pytest.mark.parametrize("reserved", ["schema", "scope", "run_id", "attempts"])
def test_checkpoint_initial_metadata_cannot_replace_core_bindings(tmp_path, reserved):
    from code_agent.benchmark_checkpoint import BenchmarkCheckpoint

    with BenchmarkCheckpoint(tmp_path / "journal", resume=False) as journal:
        with pytest.raises(ValueError, match="core bindings"):
            journal.bind("scope", "run", initial_metadata={reserved: "replacement"})
