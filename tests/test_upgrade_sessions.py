"""Stateful contract policy and controller tests. Never import fixture applications."""
import copy
import json
from types import SimpleNamespace

import pytest

from code_agent.upgrade_contracts import result_hash
from code_agent.upgrade_models import SOURCE_VERSION, TARGET_VERSION, fingerprint
from code_agent.upgrade_session_runtime import SessionVerifier, compare_sessions, parse_session
from code_agent.upgrade_sessions import (
    SessionPolicy,
    SessionScenario,
    session_commitment,
    session_key,
    validate_sessions,
)
from evals.upgrade_session_evaluation import FIXTURE, LOCK, corpus_fingerprint, policy, recipe


def payload(scenario, values, version=SOURCE_VERSION):
    return {"protocol": "upgrade-session-v1", "pydantic_version": version, "scenario_id": scenario.scenario_id,
        "steps": [{"index": index, "method": step.method, "result": value, "sha256": result_hash(value)}
            for index, (step, value) in enumerate(zip(scenario.steps, values, strict=True))]}


def test_frozen_session_corpus_and_policy_are_bound_and_confined():
    locked = json.loads(LOCK.read_bytes())
    assert corpus_fingerprint() == locked["corpus_fingerprint"]
    assert session_commitment(policy()) == locked["session_commitment"]
    validate_sessions(FIXTURE, recipe(), policy())
    assert not {session_key(row) for row in policy().discovery} & {session_key(row) for row in policy().final}


@pytest.mark.parametrize("method", ["_private", "__dict__", "nested.call", "run;echo", ""])
def test_no_private_attributes_nested_lookup_or_shell_methods(method):
    with pytest.raises(ValueError):
        SessionScenario(scenario_id="bad", factory="app.models:create_basket", steps=[{"method": method}])


def test_bounded_finite_json_and_independent_sequences():
    profile = policy().model_dump()
    profile["final"] = [{**profile["discovery"][0], "scenario_id": "different-name-same-input"}]
    with pytest.raises(ValueError, match="unique"):
        SessionPolicy.model_validate(profile)
    for steps in ([], [{"method": "snapshot"}] * 13, [{"method": "add", "args": [float("nan")]}],
            [{"method": "add", "args": ["x" * 13000]}]):
        with pytest.raises(ValueError):
            SessionScenario(scenario_id="bad", factory="app.models:create_basket", steps=steps)
    changed = policy().model_dump()
    changed["final"][0]["steps"][-2]["args"] = ["38"]
    assert session_commitment(SessionPolicy.model_validate(changed)) != session_commitment(policy())


def test_scope_checked_without_importing_repository_code(tmp_path):
    (tmp_path / "app").mkdir()
    (tmp_path / "app/models.py").write_text("raise RuntimeError('never import on host')\n")
    validate_sessions(tmp_path, recipe(), policy())
    bad = policy().model_dump()
    bad["discovery"][0]["factory"] = "outside.module:factory"
    with pytest.raises(ValueError, match="source scope"):
        validate_sessions(tmp_path, recipe(), SessionPolicy.model_validate(bad))
    (tmp_path / "upgradepilot-session.json").write_text("{}")
    with pytest.raises(ValueError, match="reserved"):
        validate_sessions(tmp_path, recipe(), policy())


def test_trace_membership_order_dependency_and_typed_results_are_exact():
    scenario = policy().discovery[0]
    values = [{"kind": "value", "value": index} for index in range(len(scenario.steps))]
    original = payload(scenario, values)
    old = parse_session(json.dumps(original), scenario, SOURCE_VERSION)
    changed = copy.deepcopy(original)
    changed["pydantic_version"] = TARGET_VERSION
    changed["steps"][3]["result"]["value"] = "3"
    changed["steps"][3]["sha256"] = result_hash(changed["steps"][3]["result"])
    new = parse_session(json.dumps(changed), scenario, TARGET_VERSION)
    diagnosis = compare_sessions(scenario, old, new)
    assert diagnosis["first_divergence"]["index"] == 3 and diagnosis["different_steps"] == 1
    assert not diagnosis["passed"]
    assert compare_sessions(scenario, old, old)["passed"]
    for fields in ({"protocol": "wrong"}, {"pydantic_version": TARGET_VERSION}, {"scenario_id": "wrong"},
            {"steps": original["steps"][:-1]}, {"steps": list(reversed(original["steps"]))}, {"steps": [None] * len(values)}):
        with pytest.raises(ValueError):
            parse_session(json.dumps({**original, **fields}), scenario, SOURCE_VERSION)
    for field, value in (("index", False), ("sha256", "0" * 64), ("method", "wrong")):
        corrupt = copy.deepcopy(original)
        corrupt["steps"][0][field] = value
        with pytest.raises(ValueError):
            parse_session(json.dumps(corrupt), scenario, SOURCE_VERSION)
    for invalid in ('{"x":1,"x":2}', '{"x":NaN}', "x" * 65001):
        with pytest.raises(ValueError):
            parse_session(invalid, scenario, SOURCE_VERSION)


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["timeout", "exit", "unstable", "mutation", "command_budget"])
async def test_runtime_unavailable_observations_never_pass(monkeypatch, failure):
    import code_agent.upgrade_session_runtime as runtime

    scenario = policy().discovery[0]
    verifier = object.__new__(SessionVerifier)
    verifier.spec = recipe().model_copy(update={"command_timeout_seconds": 5 if failure == "command_budget" else 60})
    verifier.commands = 0
    calls = []
    monkeypatch.setattr(runtime, "manifest", lambda _: "after" if failure == "mutation" and calls else "same")
    monkeypatch.setattr(runtime, "digest", lambda value: value)

    def run(_command):
        calls.append(True)
        values = [{"kind": "value", "value": len(calls) if failure == "unstable" else 0}] * len(scenario.steps)
        return SimpleNamespace(exit_code=int(failure == "exit"), timed_out=failure == "timeout", stdout=json.dumps(payload(scenario, values)))
    sandbox = SimpleNamespace(workspace=SimpleNamespace(write_file=lambda *_args: None), run=run)
    with pytest.raises(ValueError):
        await verifier.observe(sandbox, scenario, SOURCE_VERSION)


@pytest.mark.asyncio
async def test_harness_attestation_rejects_stale_images():
    verifier = object.__new__(SessionVerifier)
    verifier.commands = 0
    verifier.source = verifier.target = SimpleNamespace(run=lambda _: SimpleNamespace(exit_code=0, timed_out=False, stdout="wrong"))
    with pytest.raises(ValueError, match="identity"):
        await verifier.attest()


@pytest.mark.asyncio
async def test_reducer_preserves_order_and_failure_category_and_excludes_final_sequence():
    scenario = policy().discovery[0]
    verifier = object.__new__(SessionVerifier)

    async def compare(candidate):
        methods = [step.method for step in candidate.steps]
        divergent = "add" in methods and "snapshot" in methods and methods.index("add") < len(methods) - 1
        first = {"method": "snapshot", "source": {"kind": "value", "value": 12}, "target": {"kind": "value", "value": "12"}} if divergent else None
        return {"passed": not divergent, "first_divergence": first}
    verifier.compare_one = compare
    original = await compare(scenario)
    result = await verifier.reduce_trace(scenario, original, 8, set())
    assert 1 < result["reduced_steps"] < result["original_steps"]
    assert result["shrink_evaluations"] <= 8 and "not_global" in result["minimality"]
    no_budget = await verifier.reduce_trace(scenario, original, 0, set())
    assert no_budget["reduced_steps"] == len(scenario.steps) and not no_budget["shrink_evaluations"]


@pytest.mark.asyncio
async def test_revoked_session_commitment_admits_no_container_or_model(monkeypatch):
    from code_agent.upgrade_runner import run_migration

    monkeypatch.setattr("code_agent.upgrade_runner.resolve_image", lambda _: "sha256:" + "a" * 64)
    def forbidden(*_args, **_kwargs):
        raise AssertionError("no container admission")
    monkeypatch.setattr("code_agent.upgrade_runner.DockerSandbox.start", forbidden)
    report = await run_migration(FIXTURE, recipe(), session_policy=policy(), expected_session_commitment="0" * 64)
    assert report["status"] == "held" and report["reason"] == "hold_at:preflight" and report["source_unchanged"]
    assert not report["model_calls"] and not report["patch"]


def test_parser_rejects_forged_harness_errors_even_with_valid_hashes():
    scenario = policy().discovery[0]
    values = [{"kind": "harness_error", "type": "TimeoutError"}] * len(scenario.steps)
    with pytest.raises(ValueError, match="unavailable"):
        parse_session(json.dumps(payload(scenario, values)), scenario, SOURCE_VERSION)


@pytest.mark.asyncio
async def test_study_binds_sessions_and_forwards_matched_arm_feedback(tmp_path, monkeypatch):
    from code_agent.upgrade_study import freeze_study, run_study
    from tests.test_upgrade_study import migration_report, setup_study

    request, _, _ = setup_study(tmp_path, monkeypatch)
    task = request.tasks[0].model_copy(update={"session_policy": policy()})
    request = request.model_copy(update={"tasks": [task]})
    frozen = freeze_study(request, tmp_path)
    assert frozen["snapshots"][task.task_id]["session_commitment"] == session_commitment(policy())
    received = []

    async def fake(*_args, **kwargs):
        received.append(kwargs)
        return migration_report(_args[1])
    monkeypatch.setattr("code_agent.upgrade_study.run_migration", fake)
    result = await run_study(frozen, tmp_path, tmp_path / "data/upgradepilot/sessions", authorize_provider=True)
    assert result["complete"] and [row["session_feedback"] for row in received] == [False, False, True]
    assert all(row["expected_session_commitment"] == session_commitment(policy()) for row in received)
    changed = copy.deepcopy(frozen)
    changed["snapshots"][task.task_id]["session_commitment"] = "0" * 64
    changed["fingerprint"] = fingerprint({key: value for key, value in changed.items() if key != "fingerprint"})
    with pytest.raises(ValueError, match="session campaign changed"):
        await run_study(changed, tmp_path, tmp_path / "data/upgradepilot/revoked", authorize_provider=True)
