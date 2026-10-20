"""Migration contracts and policy tests. Never import fixture applications on the host."""
import asyncio
import json
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from code_agent.models import SandboxPolicy
from code_agent.upgrade_contracts import (
    compare_contracts,
    parse_probe,
    result_hash,
    validate_source_contracts,
)
from code_agent.upgrade_models import (
    SOURCE_VERSION,
    TARGET_VERSION,
    AcceptedDifference,
    ContractCase,
    MigrationSpec,
    safe_relative,
)
from code_agent.upgrade_plan import plan_migration
from code_agent.upgrade_runner import BudgetModel, _source_pin
from code_agent.workspace import EphemeralWorkspace, WorkspacePolicyError
from evals.upgradepilot_evaluation import FIXTURES, corpus_fingerprint


def spec():
    return MigrationSpec.model_validate_json((FIXTURES / "union/migration.json").read_text())


def batch(recipe, values, version):
    return {"protocol": "upgradepilot-probe-v1", "pydantic_version": version,
        "outcomes": [{"case_id": case.case_id, "result": value, "sha256": result_hash(value)}
                     for case, value in zip(recipe.contracts, values, strict=True)]}


def test_frozen_corpus_and_pinned_nonexecuting_impact_plan():
    lock = json.loads((Path(__file__).parent.parent / "evals/experiments/upgradepilot.lock.json").read_text())
    assert lock["corpus_fingerprint"] == corpus_fingerprint()
    for name, expected in (("union", "union-coercion"), ("validator", "validators"), ("constraints", "field-options")):
        root = FIXTURES / name
        recipe = MigrationSpec.model_validate_json((root / "migration.json").read_text())
        plan = plan_migration(root, recipe)
        assert plan.status == "planned" and expected in {row.rule_id for row in plan.findings}
        assert all("5c842dfc9c245fb37aa1f5ec5b55c1aed10bd7e6" in row.source_url for row in plan.findings)


@pytest.mark.parametrize("path", ["../escape", "/absolute", "C:/windows", "app/../tests", "app\\bad", "-x", "app//models"])
def test_source_paths_are_portable_and_confined(path):
    with pytest.raises(ValueError):
        safe_relative(path)


def test_spec_forbids_shell_commands_duplicate_cases_and_implicit_exception_contracts():
    with pytest.raises(ValueError):
        spec().model_copy().model_validate({**spec().model_dump(), "test_command": ["-c", "print('unsafe')"]})
    with pytest.raises(ValueError):
        MigrationSpec.model_validate({**spec().model_dump(), "contracts": [spec().contracts[0].model_dump()] * 2})
    with pytest.raises(ValueError):
        ContractCase(case_id="bad", entrypoint="app:run", expected_kind="error")
    with pytest.raises(ValueError):
        ContractCase(case_id="bad", entrypoint="app:run", args=[float("nan")])


def test_contract_parser_rejects_missing_duplicate_tampered_and_nonfinite_results():
    recipe = spec()
    values = [{"kind": "value", "value": 1}] * 3 + [{"kind": "error", "type": "ValidationError"}]
    original = batch(recipe, values, SOURCE_VERSION)
    assert parse_probe(json.dumps(original), recipe, SOURCE_VERSION) == original
    assert validate_source_contracts(original, recipe)
    for bad in ({**original, "pydantic_version": TARGET_VERSION}, {**original, "outcomes": original["outcomes"][:-1]},
                {**original, "outcomes": list(reversed(original["outcomes"]))}, {**original, "extra": True}):
        with pytest.raises(ValueError):
            parse_probe(json.dumps(bad), recipe, SOURCE_VERSION)
    tampered = json.loads(json.dumps(original))
    tampered["outcomes"][0]["result"]["value"] = 2
    with pytest.raises(ValueError):
        parse_probe(json.dumps(tampered), recipe, SOURCE_VERSION)
    with pytest.raises(ValueError):
        parse_probe('{"protocol":"one","protocol":"two"}', recipe, SOURCE_VERSION)
    with pytest.raises(ValueError):
        parse_probe('{"value":NaN}', recipe, SOURCE_VERSION)


def test_type_sensitive_contracts_and_exact_owner_accepted_difference():
    recipe = spec()
    values = [{"kind": "value", "value": 1}] * 3 + [{"kind": "error", "type": "ValidationError"}]
    old = batch(recipe, values, SOURCE_VERSION)
    changed = [values[0], {"kind": "value", "value": "1"}, values[2], values[3]]
    new = batch(recipe, changed, TARGET_VERSION)
    assert result_hash({"kind": "value", "value": 1}) != result_hash({"kind": "value", "value": True})
    assert result_hash({"kind": "value", "value": 1}) != result_hash({"kind": "value", "value": 1.0})
    assert compare_contracts(old, new, recipe)["regressions"] == 1
    approval = AcceptedDifference(case_id="numeric-string", source_result_sha256=old["outcomes"][1]["sha256"],
        target_result_sha256=new["outcomes"][1]["sha256"], rule_id="union-coercion", rationale="Operator explicitly accepts this documented type change.")
    approved = recipe.model_copy(update={"accepted_differences": [approval]})
    assert compare_contracts(old, new, approved)["accepted_changes"] == 1
    new["outcomes"][1]["sha256"] = result_hash({"kind": "value", "value": 999})
    assert not compare_contracts(old, new, approved)["passed"]


def test_only_controller_can_change_dependency_pin_and_owner_files_stay_protected(tmp_path):
    (tmp_path / "requirements.txt").write_text("pydantic==1.10.24\r\n")
    policy = SandboxPolicy(controller_managed_paths=["requirements.txt"], protected_paths=["owner_checks.py"])
    workspace = EphemeralWorkspace(tmp_path, policy)
    workspace.prepare()
    try:
        with pytest.raises(WorkspacePolicyError):
            workspace.check_agent_edit_path("requirements.txt")
        with pytest.raises(WorkspacePolicyError):
            workspace.check_agent_edit_path("owner_checks.py")
        workspace.write_file("requirements.txt", _source_pin(workspace.read_file("requirements.txt")))
        assert "2.12.4" in workspace.read_file("requirements.txt")
    finally:
        workspace.cleanup()
    with pytest.raises(ValueError):
        _source_pin("pydantic>=1\n")
    with pytest.raises(ValueError):
        _source_pin("pydantic==1.10.24\n-c override.txt\n")


@pytest.mark.asyncio
async def test_single_call_budget_reserves_review_and_does_not_retry_unknown_spend():
    from code_agent.provider_reliability import ProviderCircuitOpen
    class Fails:
        async def ainvoke(self, *_args, **_kwargs):
            raise TimeoutError("provider output must not be retained")
    ledger = {"calls": 0, "reserved_tokens": 0, "max_calls": 2, "max_reserved_tokens": 100000, "deadline": time.monotonic() + 30}
    model = BudgetModel(Fails(), ledger, "action")
    with pytest.raises(ProviderCircuitOpen):
        await model.ainvoke([SimpleNamespace(content="a")])
    assert ledger["calls"] == 1
    with pytest.raises(ProviderCircuitOpen):
        await model.ainvoke([SimpleNamespace(content="b")])
    assert ledger["calls"] == 1
    with pytest.raises(ProviderCircuitOpen):
        await BudgetModel(Fails(), ledger, "review").ainvoke([SimpleNamespace(content="review")])
    assert ledger["calls"] == 1


@pytest.mark.asyncio
async def test_review_call_reserve_is_not_spent_by_the_implementer():
    from code_agent.provider_reliability import ProviderCircuitOpen
    class Good:
        async def ainvoke(self, *_args, **_kwargs):
            return "ok"
    ledger = {"calls": 0, "reserved_tokens": 0, "max_calls": 2, "max_reserved_tokens": 100000, "deadline": time.monotonic() + 30}
    assert await BudgetModel(Good(), ledger, "action").ainvoke([SimpleNamespace(content="a")]) == "ok"
    with pytest.raises(ProviderCircuitOpen):
        await BudgetModel(Good(), ledger, "action").ainvoke([SimpleNamespace(content="b")])
    assert await BudgetModel(Good(), ledger, "review").ainvoke([SimpleNamespace(content="review")]) == "ok"
    assert ledger["calls"] == 2


def test_impact_analysis_rejects_missing_scope_without_importing_code(tmp_path):
    with pytest.raises(ValueError):
        plan_migration(tmp_path, spec())
    (tmp_path / "app").mkdir()
    (tmp_path / "app/models.py").write_text("raise RuntimeError('must never execute on host')\nfrom pydantic.v1 import BaseModel\n")
    plan = plan_migration(tmp_path, spec())
    assert plan.status == "held" and plan.findings[0].rule_id == "v1-namespace"


def test_parser_rejects_nonobject_rows_and_harness_errors():
    recipe = spec()
    for data in ([], {"protocol": "upgradepilot-probe-v1", "pydantic_version": SOURCE_VERSION, "outcomes": [None]}):
        with pytest.raises(ValueError):
            parse_probe(json.dumps(data), recipe, SOURCE_VERSION)
    values = [{"kind": "harness_error", "type": "TimeoutError"}] * len(recipe.contracts)
    with pytest.raises(ValueError):
        parse_probe(json.dumps(batch(recipe, values, SOURCE_VERSION)), recipe, SOURCE_VERSION)


def test_cli_reserves_private_output_before_work_and_never_overwrites(tmp_path, monkeypatch):
    from code_agent.upgrade_cli import reserve_run, save_run
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ValueError):
        reserve_run(tmp_path / "public-report")
    directory = reserve_run(tmp_path / "data/upgradepilot/private-run")
    with pytest.raises(FileExistsError):
        reserve_run(directory)
    save_run({"status": "held", "patch": ""}, directory)
    with pytest.raises(FileExistsError):
        save_run({"status": "ready_for_owner_review", "patch": "replace"}, directory)
    assert json.loads((directory / "report.json").read_text())["status"] == "held"


def test_saved_patch_bytes_match_the_verified_fingerprint_on_windows(tmp_path, monkeypatch):
    import hashlib

    from code_agent.upgrade_cli import reserve_run, save_run
    monkeypatch.chdir(tmp_path)
    directory = reserve_run(tmp_path / "data/upgradepilot/byte-exact")
    patch = "--- a/requirements.txt\n+++ b/requirements.txt\n@@ -1 +1 @@\n-pydantic==1.10.24\r\n+pydantic==2.12.4\r\n"
    expected = hashlib.sha256(patch.encode()).hexdigest()
    save_run({"status": "ready_for_owner_review", "patch": patch, "patch_sha256": expected}, directory)
    saved = (directory / "migration.diff").read_bytes()
    assert saved == patch.encode() and hashlib.sha256(saved).hexdigest() == expected


@pytest.mark.asyncio
async def test_resource_creation_settles_before_cancellation_cleanup():
    from code_agent.upgrade_runner import _owned_operation
    owned = []
    def creation():
        time.sleep(.03)
        owned.append(True)
    task = asyncio.create_task(_owned_operation(creation))
    await asyncio.sleep(.01)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert owned == [True]


@pytest.mark.parametrize("path", [".env", "data/private.py", ".git/hooks/hook.py"])
def test_private_or_runtime_paths_are_not_migration_sources(path):
    with pytest.raises(ValueError):
        safe_relative(path)
