"""Generated workflow laws and calibrated oracle semantics; no host application imports."""
import copy
import json
from types import SimpleNamespace

import pytest

from code_agent.upgrade_contracts import result_hash
from code_agent.upgrade_models import SOURCE_VERSION, fingerprint
from code_agent.upgrade_relations import (
    RelationPolicy,
    RelationSeed,
    assess_relation,
    generate_relations,
    relation_inventory,
    relation_reserve,
    validate_relations,
)
from code_agent.upgrade_sessions import session_key
from evals.upgrade_relation_evaluation import FIXTURE, LOCK, corpus_fingerprint, policy, recipe


def compiled(kind):
    campaign = generate_relations(recipe(), policy())
    return next(case for case in (*campaign.discovery, *campaign.final) if case.relation == kind)


def trace(case, *, before=0, middle=1, after=1, error="ValueError"):
    rows = [{"result": {"kind": "value", "value": None}} for _ in case.scenarios[0].steps]
    for index, value in ((case.before, before), (case.middle, middle), (case.after, after)):
        rows[index]["result"] = {"kind": "value", "value": value}
    if case.error_index is not None:
        rows[case.error_index]["result"] = {"kind": "error", "type": error}
    for row in rows:
        row["sha256"] = result_hash(row["result"])
    return rows


def test_frozen_campaign_is_reproducible_disjoint_and_covers_each_declared_law():
    locked = json.loads(LOCK.read_bytes())
    assert corpus_fingerprint() == locked["corpus_fingerprint"]
    campaign = generate_relations(recipe(), policy())
    assert campaign == generate_relations(recipe(), policy())
    assert campaign.commitment == locked["campaign_commitment"]
    assert len(campaign.discovery) == len(campaign.final) == 4
    assert relation_inventory(campaign.discovery) == relation_inventory(campaign.final)
    assert len(relation_inventory(campaign.discovery)) == 4
    assert not {session_key(s) for row in campaign.discovery for s in row.scenarios} & {
        session_key(s) for row in campaign.final for s in row.scenarios}
    assert relation_reserve(campaign.final) > 0
    validate_relations(FIXTURE, recipe(), policy())


@pytest.mark.parametrize("change", ["empty", "bad_index", "observer", "missing_error", "irrelevant_error", "duplicate", "parameterized_observer"])
def test_owner_policy_rejects_ambiguous_laws_and_unbounded_or_invalid_inputs(change):
    # Start from a positive-operation law, then violate one policy constraint.
    data = next(row.model_dump() for row in policy().seeds if row.relation == "idempotent_retry")
    if change == "empty":
        data["operation_indices"] = []
    elif change == "bad_index":
        data["operation_indices"] = [-1]
    elif change == "observer":
        data["operation_indices"] = [len(data["scenario"]["steps"]) - 1]
    elif change == "missing_error":
        data["relation"] = "rejected_call_rollback"
    elif change == "irrelevant_error":
        data["expected_error"] = "ValueError"
    elif change == "duplicate":
        data["operation_indices"] = [0, 0]
    else:
        data["scenario"]["steps"][-1]["args"] = ["not an observation contract"]
    with pytest.raises(ValueError):
        RelationSeed.model_validate(data)


def test_no_effect_and_null_observations_cannot_count_as_idempotency_evidence():
    case = compiled("idempotent_retry")
    assert assess_relation(case, [trace(case)])["passed"]
    no_effect = assess_relation(case, [trace(case, middle=0, after=0)])
    assert not no_effect["passed"] and "operation_effect_missing" in no_effect["violations"]
    missing = assess_relation(case, [trace(case, before=None)])
    assert not missing["passed"] and "uninformative_observation" in missing["violations"]
    repeated = assess_relation(case, [trace(case, after=2)])
    assert repeated["violations"] == ["retry_not_idempotent"]
    typed = assess_relation(case, [trace(case, middle=1, after=True)])
    assert not typed["passed"]


def test_read_noninterference_detects_changed_observer_state():
    case = compiled("read_noninterference")
    assert assess_relation(case, [trace(case)])["passed"]
    assert "read_changed_state" in assess_relation(case, [trace(case, after=2)])["violations"]


def test_rollback_checks_the_expected_rejection_and_observed_state_independently():
    case = compiled("rejected_call_rollback")
    assert assess_relation(case, [trace(case, before=0, middle=0, after=0)])["passed"]
    assert "rejected_call_changed_state" in assess_relation(case, [trace(case)])["violations"]
    wrong = assess_relation(case, [trace(case, before=0, after=0, error="TypeError")])
    assert "expected_rejection_not_observed" in wrong["violations"]


def test_commutativity_requires_both_effects_and_equal_final_observations():
    case = compiled("commutative_pair")
    left, right = trace(case, middle=1, after=3), trace(case, middle=2, after=3)
    assert assess_relation(case, [left, right])["passed"]
    assert "operation_order_changed_result" in assess_relation(case, [left, trace(case, middle=2, after=4)])["violations"]
    assert "second_operation_effect_missing" in assess_relation(case, [left, trace(case, middle=3, after=3)])["violations"]
    with pytest.raises(ValueError, match="incomplete"):
        assess_relation(case, [left])
    with pytest.raises(ValueError, match="incomplete"):
        assess_relation(case, [left[:-1], right])


def test_generation_holds_insufficient_pools_long_expansions_and_out_of_scope_factories(tmp_path):
    data = policy().model_dump()
    data["discovery_cases"] = data["final_cases"] = 8
    with pytest.raises(ValueError, match="insufficient"):
        generate_relations(recipe(), RelationPolicy.model_validate(data))
    data = policy().model_dump()
    seed = next(row for row in data["seeds"] if row["relation"] == "idempotent_retry")
    seed["scenario"]["steps"] = [seed["scenario"]["steps"][0]] * 10 + [seed["scenario"]["steps"][-1]]
    seed["operation_indices"] = [9]
    with pytest.raises(ValueError):
        generate_relations(recipe(), RelationPolicy.model_validate(data))
    (tmp_path / "app").mkdir()
    (tmp_path / "app/models.py").write_text("raise RuntimeError('never import')\n")
    validate_relations(tmp_path, recipe(), policy())
    data = policy().model_dump()
    data["seeds"][0]["scenario"]["factory"] = "outside.module:factory"
    with pytest.raises(ValueError):
        validate_relations(tmp_path, recipe(), RelationPolicy.model_validate(data))


def test_campaign_binds_laws_inputs_budgets_and_compiler_identity():
    original = generate_relations(recipe(), policy())
    data = policy().model_dump()
    data["seed"] += 1
    assert generate_relations(recipe(), RelationPolicy.model_validate(data)).commitment != original.commitment
    data = policy().model_dump()
    data["seeds"][0]["scenario"]["steps"][0]["args"] = [13]
    assert generate_relations(recipe(), RelationPolicy.model_validate(data)).commitment != original.commitment


@pytest.mark.asyncio
async def test_revoked_relation_commitment_holds_before_container_or_model(monkeypatch):
    from code_agent.upgrade_runner import run_migration

    monkeypatch.setattr("code_agent.upgrade_runner.resolve_image", lambda _: "sha256:" + "a" * 64)
    def forbidden(*_args, **_kwargs):
        raise AssertionError("no container or model admission")
    monkeypatch.setattr("code_agent.upgrade_runner.DockerSandbox.start", forbidden)
    report = await run_migration(FIXTURE, recipe(), relation_policy=policy(), expected_relation_commitment="0" * 64)
    assert report["reason"] == "hold_at:preflight" and report["source_unchanged"]
    assert report["model_calls"] == 0 and not report["patch"]


@pytest.mark.asyncio
async def test_runtime_uses_one_copy_discards_unavailable_evidence_and_closes_owned_resource(monkeypatch):
    import code_agent.upgrade_relation_runtime as runtime

    case = compiled("idempotent_retry")
    events = []
    class Sandbox:
        def __init__(self, *_args):
            pass

        def start(self):
            events.append("start")

        def close(self):
            events.append("close")
    class Reader:
        def __init__(self, _spec):
            self.commands = 2

        async def attest_one(self, _sandbox):
            pass

        async def observe(self, *_args):
            raise ValueError("timeout or unstable evidence")
    monkeypatch.setattr(runtime, "DockerSandbox", Sandbox)
    monkeypatch.setattr(runtime, "SessionReader", Reader)
    with pytest.raises(ValueError):
        await runtime.verify_relations(FIXTURE, recipe(), object(), [case], SOURCE_VERSION)
    assert events == ["start", "close"]
    with pytest.raises(ValueError, match="empty"):
        await runtime.verify_relations(FIXTURE, recipe(), object(), [], SOURCE_VERSION)


@pytest.mark.asyncio
async def test_source_oracle_violation_blocks_repair_even_if_original_tests_pass(monkeypatch):
    # Run the actual control flow without containers; no source code is executed.
    import code_agent.upgrade_runner as runner
    from code_agent.upgrade_runner import run_migration
    from code_agent.workspace import EphemeralWorkspace

    class FakeSandbox:
        def __init__(self, repository, sandbox_policy):
            self.repository = repository
            self.workspace = EphemeralWorkspace(repository, sandbox_policy)

        def start(self):
            self.workspace.prepare()

        def close(self):
            self.workspace.cleanup()
    async def attest(*_args):
        pass
    async def run(*_args):
        return SimpleNamespace(exit_code=0)
    async def probes(*_args):
        return {}
    async def invalid(*_args, **_kwargs):
        return {"passed": False, "total": 8, "violations": 2}
    monkeypatch.setattr(runner, "resolve_image", lambda _: "sha256:" + "a" * 64)
    monkeypatch.setattr(runner, "DockerSandbox", FakeSandbox)
    monkeypatch.setattr(runner, "_attest", attest)
    monkeypatch.setattr(runner, "_run", run)
    monkeypatch.setattr(runner, "_probes", probes)
    monkeypatch.setattr(runner, "validate_source_contracts", lambda *_args: True)
    monkeypatch.setattr(runner, "verify_relations", invalid)
    report = await run_migration(FIXTURE, recipe(), relation_policy=policy(), scripted_models=(object(), object(), object()))
    assert report["reason"] == "hold_at:metamorphic_source_calibration"
    assert report["model_calls"] == 0 and report["baseline"] is None and report["source_unchanged"]


@pytest.mark.asyncio
async def test_frozen_study_forwards_laws_and_revokes_changed_campaigns(tmp_path, monkeypatch):
    from code_agent.upgrade_study import freeze_study, run_study
    from tests.test_upgrade_study import migration_report, setup_study

    request, _, _ = setup_study(tmp_path, monkeypatch)
    task = request.tasks[0].model_copy(update={"relation_policy": policy()})
    request = request.model_copy(update={"tasks": [task]})
    frozen = freeze_study(request, tmp_path)
    commitment = frozen["snapshots"][task.task_id]["relation_commitment"]
    received = []
    async def fake(*_args, **kwargs):
        received.append(kwargs)
        return migration_report(_args[1])
    monkeypatch.setattr("code_agent.upgrade_study.run_migration", fake)
    result = await run_study(frozen, tmp_path, tmp_path / "data/upgradepilot/laws", authorize_provider=True)
    assert result["complete"] and [row["relation_feedback"] for row in received] == [False, False, True]
    assert all(row["expected_relation_commitment"] == commitment for row in received)
    changed = copy.deepcopy(frozen)
    changed["snapshots"][task.task_id]["relation_commitment"] = "0" * 64
    changed["fingerprint"] = fingerprint({key: value for key, value in changed.items() if key != "fingerprint"})
    with pytest.raises(ValueError, match="relation campaign changed"):
        await run_study(changed, tmp_path, tmp_path / "data/upgradepilot/revoked", authorize_provider=True)
