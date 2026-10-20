"""Independent discovery/final inputs and bounded reduction; no application host imports."""
import asyncio
from pathlib import Path

import pytest

from code_agent.upgrade_differential import DifferentialVerifier, _batches
from code_agent.upgrade_inputs import (
    CounterexamplePolicy,
    complexity,
    generate_campaign,
    input_key,
    shrink_candidates,
    variants,
)
from code_agent.upgrade_models import ContractCase, MigrationSpec


def recipe():
    return MigrationSpec.model_validate_json(Path("tests/fixtures/upgradepilot/union/migration.json").read_bytes())


def test_campaign_is_frozen_reproducible_distinct_and_seed_bound():
    spec, policy = recipe(), CounterexamplePolicy()
    first, second = generate_campaign(spec, policy), generate_campaign(spec, policy)
    discovery = {input_key(row) for row in first.discovery}
    final = {input_key(row) for row in first.final}
    original = {input_key(row) for row in spec.contracts}
    assert first == second and len(discovery) == 24 and len(final) == 16
    assert not discovery & final and not (discovery | final) & original
    assert generate_campaign(spec, policy.model_copy(update={"seed": 22})).commitment != first.commitment
    assert first.discovery_fingerprint != first.final_fingerprint


def test_single_owner_example_stratifies_numeric_strings_and_fractional_values():
    spec = recipe()
    campaign = generate_campaign(spec.model_copy(update={"contracts": [spec.contracts[0]]}), CounterexamplePolicy())
    for cases in (campaign.discovery, campaign.final):
        assert any(case.args and isinstance(case.args[0], dict) and isinstance(case.args[0].get("value"), str)
            and case.args[0]["value"].strip().isdigit() for case in cases)


def test_nested_and_alias_keys_are_mutated_without_loading_model_code():
    case = ContractCase(case_id="seed", entrypoint="app.models:normalize", args=[{"externalAlias": [{"nested": "7"}]}])
    generated = list(variants(case))
    assert any(row.args == [{"externalAlias": [{"nested": None}]}] for row in generated)
    assert any(row.args == [{"externalAlias": []}] for row in generated)
    assert any(row.args == [{"externalAlias": [{}]}] for row in generated)
    assert any(row.args == [{"externalAlias": [{"nested": "7"}], "upgradepilot_extra": "unexpected"}] for row in generated)


def test_policy_and_generation_fail_closed_when_insufficient_or_oversized():
    with pytest.raises(ValueError):
        CounterexamplePolicy(discovery_cases=50, final_cases=50)
    with pytest.raises(ValueError):
        CounterexamplePolicy(max_shrink_evaluations=-1)
    spec = recipe().model_copy(update={"contracts": [ContractCase(case_id="none", entrypoint="app.models:normalize")]})
    with pytest.raises(ValueError, match="insufficient"):
        generate_campaign(spec, CounterexamplePolicy())


def test_shrink_candidates_are_strictly_smaller_unique_and_bounded():
    seed = ContractCase(case_id="failure", entrypoint="app.models:normalize", args=[{"value": "999", "noise": [1, 2]}])
    candidates = shrink_candidates(seed, 10)
    assert 0 < len(candidates) <= 10
    assert len({row.case_id for row in candidates}) == len(candidates)
    assert all(complexity(row) < complexity(seed) for row in candidates)
    assert shrink_candidates(seed, 0) == []


def test_reduction_prioritizes_preserving_the_invocation_and_numeric_string_shape():
    seed = ContractCase(case_id="failure", entrypoint="app.models:normalize", args=[{"value": "999"}])
    first = shrink_candidates(seed, 1)[0]
    assert isinstance(first.args[0], dict) and first.args[0]["value"].strip().lstrip("-").isdigit()


def test_batches_respect_command_time_and_json_budgets():
    spec = recipe().model_copy(update={"command_timeout_seconds": 5})
    rows = [ContractCase(case_id=str(index), entrypoint="app.models:normalize", args=["x" * 7000]) for index in range(8)]
    batches = list(_batches(rows, spec))
    assert len(batches) == 4 and all(len(batch) == 2 for batch in batches)


@pytest.mark.asyncio
async def test_shrinking_never_executes_a_frozen_final_input_and_respects_budget():
    seed = ContractCase(case_id="failure", entrypoint="app.models:normalize", args=[{"value": "999", "noise": [1]}])
    forbidden = shrink_candidates(seed, 1)[0]
    calls = []

    class FakeVerifier:
        async def compare(self, candidates):
            calls.extend(candidates)
            return [{"case": case, "regression": True, "source": {"kind": "value", "value": 0},
                "target": {"kind": "value", "value": "0"}, "source_sha256": "a" * 64, "target_sha256": "b" * 64}
                for case in candidates]

    row = {"case": seed, "regression": True, "source": {"kind": "value"}, "target": {"kind": "value"}}
    result = await DifferentialVerifier.reduce(FakeVerifier(), row, 9, {input_key(forbidden)})
    assert result["shrink_evaluations"] <= 9 and all(input_key(case) != input_key(forbidden) for case in calls)
    assert result["reduced_input_bytes"] < result["original_input_bytes"]


@pytest.mark.asyncio
async def test_verifier_cleans_both_resources_after_partial_start_failure():
    from types import SimpleNamespace

    events = []

    def fail():
        raise RuntimeError("start unavailable")

    verifier = object.__new__(DifferentialVerifier)
    verifier.source = SimpleNamespace(start=lambda: events.append("source-start"), close=lambda: events.append("source-close"))
    verifier.target = SimpleNamespace(start=fail, close=lambda: events.append("target-close"))
    with pytest.raises(RuntimeError):
        await verifier.__aenter__()
    assert events == ["source-start", "target-close", "source-close"]


@pytest.mark.asyncio
async def test_settled_execution_does_not_race_cleanup_on_cancellation():
    import time

    from code_agent.upgrade_differential import settled_call
    events = []

    def operation():
        time.sleep(.02)
        events.append("settled")

    task = asyncio.create_task(settled_call(operation))
    await asyncio.sleep(.005)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert events == ["settled"]


def test_docker_output_is_decoded_as_utf8_not_the_windows_locale(monkeypatch, tmp_path):
    import locale
    import subprocess
    from io import BytesIO
    from types import SimpleNamespace

    from code_agent.models import SandboxPolicy
    from code_agent.sandbox import DockerSandbox

    policy = SandboxPolicy(command_timeout_seconds=17, max_output_chars=1000)
    calls = []
    stdout = BytesIO(b"x" * 4100 + b"\xc3\xa9")
    stderr = BytesIO(b"\xe2\x82\xac\xff")

    def wait(*, timeout):
        assert timeout == policy.command_timeout_seconds
        return 5

    def fake_popen(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(stdout=stdout, stderr=stderr, wait=wait, poll=lambda: 5)

    # Exercise the real bounded reader; only the Docker process is replaced.
    monkeypatch.setattr(locale, "getencoding", lambda: "cp1252")
    monkeypatch.setattr(locale, "getpreferredencoding", lambda _do_setlocale=True: "cp1252")
    monkeypatch.setattr("code_agent.bounded_process.subprocess.Popen", fake_popen)
    sandbox = DockerSandbox(tmp_path, policy)
    sandbox.container_id = "authored-no-container-created"
    command = ["python", "-c", "pass"]
    result = sandbox.run(command)

    assert calls == [(
        ["docker", "exec", "--user", "65532:65532", sandbox.container_id, *command],
        {"stdin": subprocess.DEVNULL, "stdout": subprocess.PIPE, "stderr": subprocess.PIPE, "bufsize": 0},
    )]
    assert result.command == command and result.exit_code == 5 and not result.timed_out
    assert result.stdout == "x" * 999 + "é"
    assert result.stderr == "\u20ac\ufffd"
    assert stdout.closed and stderr.closed


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["unstable", "harness_error", "mutation"])
async def test_generated_execution_rejects_unavailable_unstable_or_mutating_evidence(monkeypatch, failure):
    import json
    from types import SimpleNamespace

    from code_agent.upgrade_contracts import result_hash
    from code_agent.upgrade_models import SOURCE_VERSION

    spec = recipe()
    calls = []
    manifests = iter(["before", "changed", "changed"]) if failure == "mutation" else iter(["same"] * 5)
    monkeypatch.setattr("code_agent.upgrade_differential.manifest", lambda _: next(manifests))
    monkeypatch.setattr("code_agent.upgrade_differential.digest", lambda value: value)

    def run(_command):
        calls.append(True)
        result = {"kind": "harness_error", "type": "TimeoutError"} if failure == "harness_error" else {"kind": "value", "value": len(calls)}
        stdout = json.dumps({"protocol": "upgradepilot-probe-v1", "pydantic_version": SOURCE_VERSION,
            "outcomes": [{"case_id": spec.contracts[0].case_id, "result": result, "sha256": result_hash(result)}]})
        return SimpleNamespace(exit_code=0, timed_out=False, stdout=stdout)

    verifier = object.__new__(DifferentialVerifier)
    verifier.spec, verifier.commands = spec, 0
    sandbox = SimpleNamespace(workspace=SimpleNamespace(write_file=lambda *_: None), run=run)
    with pytest.raises(ValueError):
        await verifier._observe(sandbox, [spec.contracts[0]], SOURCE_VERSION)


@pytest.mark.asyncio
async def test_empty_evidence_cannot_pass():
    from code_agent.upgrade_differential import verify_generated
    with pytest.raises(ValueError, match="empty"):
        await verify_generated(None, None, recipe(), None, [], policy=CounterexamplePolicy())
