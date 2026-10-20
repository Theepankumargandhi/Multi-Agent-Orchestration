"""Actual Docker study recovery plus labelled receipt-replay fault controls, never live providers."""
from __future__ import annotations

import argparse
import asyncio
import json
import shutil
from pathlib import Path
from unittest.mock import patch

from code_agent.upgrade_contracts import _unique_pairs
from code_agent.upgrade_inputs import CounterexamplePolicy
from code_agent.upgrade_models import MigrationSpec, fingerprint
from code_agent.upgrade_recovery import MigrationJournal, StudyRunBudget
from code_agent.upgrade_runner import run_migration
from code_agent.upgrade_study import StudyRequest, StudyTask, freeze_study, run_study
from evals.upgrade_counterexample_evaluation import CounterexampleSpecialist
from evals.upgradepilot_evaluation import FIXTURES, ROOT


def never_build_provider(*_args, **_kwargs):
    raise AssertionError("Recovery controls must not build a live provider")


def load_report(directory):
    report = json.loads((directory / "report.json").read_bytes(), object_pairs_hook=_unique_pairs)
    report["patch"] = (directory / "migration.diff").read_bytes().decode("utf-8") if (directory / "migration.diff").exists() else ""
    assert report["fingerprint"] == fingerprint({key: value for key, value in report.items() if key != "fingerprint"})
    return report


async def evaluate(control_root):
    control_root = control_root.resolve()
    private = (ROOT / "data/upgradepilot").resolve()
    if not control_root.is_relative_to(private) or control_root == private:
        raise ValueError("control roots must be new directories below ignored data/upgradepilot/")
    control_root.mkdir(parents=True, exist_ok=False)
    repository = control_root / "repositories/union"
    shutil.copytree(FIXTURES / "union", repository)
    (repository / "LICENSE").write_text("Authored local controls; no third-party permission assertion.\n", encoding="utf-8")
    spec = MigrationSpec.model_validate_json((repository / "migration.json").read_bytes())
    task = StudyTask(task_id=spec.task_id, repository="repositories/union", upstream_url="https://github.com/example/authored",
        upstream_commit="b" * 40, license_file="LICENSE", license_identifier="authored-control", owner_authorized=True, spec=spec)
    request = StudyRequest(study_id="authored-recovery", model="groq:offline-scripted-control", tasks=[task],
        policy=CounterexamplePolicy(discovery_cases=4, final_cases=4, max_shrink_evaluations=0))
    frozen = freeze_study(request, control_root)
    destination = control_root / "data/upgradepilot/study"
    budget = StudyRunBudget(max_attempts=4, max_provider_calls=48)
    calls, reports, checks = [], {}, []
    interrupted = False

    async def scripted(repository, selected, **kwargs):
        nonlocal interrupted
        arm = "codemod_only" if kwargs["model"] is None else "counterexample_repair" if kwargs["discovery_feedback"] else "existing_repair"
        calls.append(arm)
        if arm == "existing_repair" and not interrupted:
            interrupted = True
            raise InterruptedError("authored lost-return control; usage deliberately treated as uncertain")
        models = None if arm == "codemod_only" else tuple(CounterexampleSpecialist(role) for role in ("analysis", "actions", "review"))
        options = {key: value for key, value in kwargs.items() if key not in {"model", "authorize_provider"}}
        result = await run_migration(repository, selected, scripted_models=models, **options)
        reports[arm] = result
        return result

    with patch("code_agent.providers.build_provider", never_build_provider), patch("code_agent.upgrade_study.run_migration", scripted):
        try:
            await run_study(frozen, control_root, destination, authorize_provider=True, budget=budget)
            raise AssertionError("expected an interruption")
        except InterruptedError:
            pass
        try:
            await run_study(frozen, control_root, destination, authorize_provider=True, resume=True)
            raise AssertionError("implicit retry was allowed")
        except ValueError as exc:
            assert "retry-interrupted" in str(exc)
        checks.append({"name": "implicit_retry_blocked", "passed": calls == ["codemod_only", "existing_repair"],
            "evidence": "actual_Docker_first_arm_then_injected_interruption"})
        completed = await run_study(frozen, control_root, destination, authorize_provider=True, resume=True, retry_interrupted=True)
        resources = completed["resource_accounting"]
        checks.append({"name": "explicit_resume_with_retained_uncertainty", "passed": completed["complete"]
            and not completed["quality_metrics_valid"] and resources["admitted_attempts"] == 4
            and resources["unresolved_provider_call_reservations"] == 16
            and sum(report["model_calls"] for report in reports.values()) == 12
            and reports["codemod_only"]["status"] == "held"
            and all(reports[arm]["status"] == "ready_for_owner_review" for arm in ("existing_repair", "counterexample_repair"))
            and all(report["source_unchanged"] and not report["live_provider_calls"] for report in reports.values()),
            "evidence": "actual_Docker_codemod_and_two_scripted_specialist_repairs"})
        before = list(calls)
        await run_study(frozen, control_root, destination, authorize_provider=True, resume=True)
        checks.append({"name": "completed_holds_and_successes_not_reexecuted", "passed": calls == before,
            "evidence": "actual_checkpoint_resume"})
        corrupted = control_root / "data/upgradepilot/tampered-copy"
        shutil.copytree(destination, corrupted)
        receipt = corrupted / "attempts/1/report.json"
        receipt.write_bytes(receipt.read_bytes() + b" ")
        try:
            await run_study(frozen, control_root, corrupted, authorize_provider=True, resume=True, retry_interrupted=True)
            raise AssertionError("corrupted historical report was accepted")
        except ValueError as exc:
            assert "artifact integrity" in str(exc)
        checks.append({"name": "historical_receipt_tampering_blocked", "passed": calls == before,
            "evidence": "intentional_corruption_of_private_control_copy"})

    # Reuse real controller receipts as labelled doubles to isolate two storage fault windows.
    replay_calls = []
    async def replay(_repository, _selected, **kwargs):
        arm = "codemod_only" if kwargs["model"] is None else "counterexample_repair" if kwargs["discovery_feedback"] else "existing_repair"
        replay_calls.append(arm)
        return reports[arm]
    original_finish = MigrationJournal.finish_report
    def before_commit(self, key, report):
        raise InterruptedError("authored crash after saved receipt, before journal commit")
    adopted_dir = control_root / "data/upgradepilot/orphan-control"
    with patch("code_agent.providers.build_provider", never_build_provider), patch("code_agent.upgrade_study.run_migration", replay):
        with patch.object(MigrationJournal, "finish_report", before_commit):
            try:
                await run_study(frozen, control_root, adopted_dir, authorize_provider=True)
                raise AssertionError("expected storage interruption")
            except InterruptedError:
                pass
        assert MigrationJournal.finish_report is original_finish
        adopted = await run_study(frozen, control_root, adopted_dir, authorize_provider=True, resume=True)
        checks.append({"name": "saved_receipt_adopted_without_replaying_arm", "passed": adopted["complete"]
            and "adopted_saved_receipt" in adopted["recovery_dispositions"].values()
            and replay_calls.count("codemod_only") == 1 and adopted["resource_accounting"]["admitted_attempts"] == 3,
            "evidence": "real_receipt_replay_with_injected_pre_commit_failure_not_new_AI_execution"})

    restricted = control_root / "data/upgradepilot/budget-control"
    blocked_calls = []
    async def uncertain(_repository, _selected, **kwargs):
        blocked_calls.append(kwargs["model"])
        if kwargs["model"]:
            raise InterruptedError("injected unknown provider usage")
        return reports["codemod_only"]
    with patch("code_agent.providers.build_provider", never_build_provider), patch("code_agent.upgrade_study.run_migration", uncertain):
        try:
            await run_study(frozen, control_root, restricted, authorize_provider=True,
                budget=StudyRunBudget(max_attempts=4, max_provider_calls=16))
            raise AssertionError("expected interrupted arm")
        except InterruptedError:
            pass
        try:
            await run_study(frozen, control_root, restricted, authorize_provider=True, resume=True, retry_interrupted=True)
            raise AssertionError("unknown usage was incorrectly released")
        except ValueError as exc:
            assert "budget exhausted" in str(exc)
        checks.append({"name": "unknown_usage_cannot_fund_retry", "passed": len(blocked_calls) == 2,
            "evidence": "real_receipt_replay_with_simulated_uncertain_usage_not_new_AI_execution"})
    retained = [load_report(path.parent) for path in (destination / "attempts").glob("*/report.json")]
    assert len(retained) == 3 and all(report["source_unchanged"] for report in retained)
    return {"evidence_kind": "authored_recovery_actual_Docker_plus_labelled_receipt_replay_fault_controls",
        "total": len(checks), "passed": sum(bool(row["passed"]) for row in checks), "controls": checks,
        "control_root": str(control_root), "live_provider_calls": 0, "scripted_specialist_calls": 12,
        "provider_reservation_remaining_unknown": 16, "real_repository_quality_measured": False,
        "primary_study": completed, "interpretation": "Single-host recovery/accounting controls, not exactly-once API execution, billing accuracy or model quality."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--control-root", type=Path, required=True)
    args = parser.parse_args()
    result = asyncio.run(evaluate(args.control_root))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8", newline="") as stream:
        stream.write(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key not in {"controls", "primary_study"}}, indent=2))
    if result["passed"] != result["total"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
