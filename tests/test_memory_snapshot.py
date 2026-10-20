import json
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from code_agent.evaluation import run_benchmark
from code_agent.evaluation_models import CodeBenchmarkCase
from code_agent.memory_snapshot import (
    FrozenRepairMemory,
    SourceLineage,
    audit_snapshot,
    freeze_memory,
    issue_identity,
    prepare_memory,
    validate_snapshot,
)
from code_agent.models import CodeTask, SandboxPolicy
from code_agent.repair_memory import RepairMemoryStore, digest, scope_hash
from evals.repair_memory_evaluation import _scripted_evidence


def make_snapshot(directory):
    store = RepairMemoryStore(Path(directory) / "lessons.db")
    identifier = store.approve(_scripted_evidence(), project="project", namespace="alice", repository="repo",
        problem="Incorrect tax rounding precision", lesson="Preserve decimal precision and verify monetary boundaries.",
        applicable_paths=["tax.py"], reviewer="owner", approve=True, now=1000)
    source = SourceLineage(case_id="training-case", family_id="training-family", dataset_sha256="a" * 64,
        issue="A monetary tax adjustment loses precision on fractional currency values.", completed_at=999)
    snapshot = freeze_memory(store, project="project", namespace="alice", repositories=["repo"],
        lineage={identifier: source}, frozen_at=1001)
    return store, snapshot, identifier


def case(**changes):
    values = dict(id="heldout-case", repository="repo", issue="Fix tax rounding for negative refundable amounts without changing precision.",
                  test_command=["python", "-m", "pytest", "-q"], metadata={"family_id": "refund-signs"})
    values.update(changes)
    return CodeBenchmarkCase(**values)


def test_freeze_audit_and_immutable_read_without_live_database(tmp_path):
    store, snapshot, identifier = make_snapshot(tmp_path)
    cases = [case()]
    frozen = FrozenRepairMemory.prepare(snapshot, cases, now=1002)
    store.withdraw(identifier, scope_hash("project", "alice", "repo"), delete=True)
    snapshot.repositories["repo"].entries.clear()
    with pytest.raises(FrozenInstanceError):
        frozen.payload = "changed"

    class Workspace:
        def list_files(self, limit):
            return ["tax.py"]

    context, receipt = frozen.retrieve(CodeTask(repository="repo", issue=cases[0].issue,
        benchmark_case_id=cases[0].id, policy=SandboxPolicy(repair_memory_enabled=True)), Workspace(), stage="repair", iteration=1)
    assert "Preserve decimal precision" in context
    assert identifier in receipt.lesson_scores
    assert receipt.snapshot_sha256 == frozen.fingerprint
    assert receipt.status == "selected"
    assert frozen.dataset_sha256 == digest([item.model_dump(mode="json") for item in cases])
    assert "fractional currency values" not in frozen.payload  # source issue retained only as hashes


@pytest.mark.parametrize("change,code", [
    ({"id": "training-case"}, "source_case_overlap"),
    ({"metadata": {"family_id": "training-family"}}, "source_family_overlap"),
    ({"issue": "A monetary tax adjustment loses precision on fractional currency values."}, "source_issue_exact_match"),
    ({"issue": "A MONETARY tax adjustment loses precision on fractional currency values!!!"}, "source_issue_exact_match"),
    ({"issue": "A monetary tax adjustment loses precision on fractional currency values and additional boundaries."}, "source_issue_near_duplicate"),
    ({"issue": "Incorrect tax rounding precision"}, "lesson_problem_overlap"),
    ({"issue": "Preserve decimal precision and verify monetary boundaries."}, "lesson_text_overlap"),
])
def test_leakage_controls_fail_closed(tmp_path, change, code):
    _, snapshot, _ = make_snapshot(tmp_path)
    report = audit_snapshot(snapshot, [case(**change)], now=1002)
    assert report["passed"] is False
    assert code in {finding["code"] for finding in report["findings"]}
    assert report["semantic_independence_proven"] is False
    assert "currency values" not in json.dumps(report)


def test_dataset_overlap_and_future_freeze_rejected(tmp_path):
    _, snapshot, _ = make_snapshot(tmp_path)
    cases = [case()]
    snapshot.repositories["repo"].entries[0].source_dataset_sha256 = digest([item.model_dump(mode="json") for item in cases])
    snapshot.seal()
    codes = {item["code"] for item in audit_snapshot(snapshot, cases, now=1000)["findings"]}
    assert codes >= {"source_dataset_overlap", "future_snapshot"}


def test_missing_family_scope_or_lineage_are_not_guessed(tmp_path):
    store, snapshot, _ = make_snapshot(tmp_path)
    for selected in ([case(metadata={})], [case(repository="other")]):
        with pytest.raises(ValueError):
            audit_snapshot(snapshot, selected)
    with pytest.raises(ValueError, match="lineage"):
        freeze_memory(store, project="project", namespace="alice", repositories=["repo"], lineage={}, frozen_at=1001)
    with pytest.raises(ValueError, match="unique"):
        freeze_memory(store, project="project", namespace="alice", repositories=["repo", "repo"], lineage={})


def test_snapshot_integrity_temporal_binding_and_duplicate_ids(tmp_path):
    _, snapshot, _ = make_snapshot(tmp_path)
    snapshot.fingerprint = "0" * 64
    with pytest.raises(ValueError, match="binding"):
        validate_snapshot(snapshot)
    snapshot.seal()
    snapshot.repositories["repo"].entries[0].source_completed_at = 1005
    snapshot.seal()
    with pytest.raises(ValueError, match="lineage"):
        validate_snapshot(snapshot)
    snapshot.repositories["repo"].entries[0].source_completed_at = 999
    snapshot.repositories["repo"].entries.append(snapshot.repositories["repo"].entries[0].model_copy(deep=True))
    snapshot.seal()
    with pytest.raises(ValueError, match="lineage"):
        validate_snapshot(snapshot)


def test_file_binding_and_audit_selection_drift_are_rejected(tmp_path):
    _, snapshot, _ = make_snapshot(tmp_path)
    path = tmp_path / "snapshot.json"
    path.write_text(snapshot.model_dump_json())
    frozen = FrozenRepairMemory.load(path, snapshot.fingerprint, [case()])
    with pytest.raises(ValueError, match="expected fingerprint"):
        FrozenRepairMemory.load(path, "0" * 64, [case()])
    with pytest.raises(ValueError, match="drifted"):
        prepare_memory([case(id="another-case")], SandboxPolicy(repair_memory_enabled=True), frozen, "verified_pr")
    with pytest.raises(ValueError, match="supported"):
        prepare_memory([case()], SandboxPolicy(repair_memory_enabled=True), frozen, "adaptive_supervisor")
    with pytest.raises(ValueError, match="required"):
        prepare_memory([case()], SandboxPolicy(repair_memory_enabled=True), None, "single_agent")


@pytest.mark.asyncio
async def test_invalid_snapshot_rejected_before_checkpoint_or_model(tmp_path):
    _, snapshot, _ = make_snapshot(tmp_path)
    frozen = FrozenRepairMemory.prepare(snapshot, [case()])
    checkpoint = tmp_path / "checkpoint"
    with pytest.raises(ValueError, match="drifted"):
        await run_benchmark([case(id="another-case")], checkpoint_dir=checkpoint,
                            policy=SandboxPolicy(repair_memory_enabled=True), memory_snapshot=frozen)
    assert not checkpoint.exists()


@pytest.mark.asyncio
@pytest.mark.parametrize("valid_receipts", [True, False])
async def test_benchmark_pins_snapshot_and_holds_missing_exposure_evidence(tmp_path, monkeypatch, valid_receipts):
    _, snapshot, _ = make_snapshot(tmp_path)
    frozen = FrozenRepairMemory.prepare(snapshot, [case()])
    (tmp_path / "repo").mkdir()

    class Workspace:
        def list_files(self, limit):
            return ["tax.py"]

    class Sandbox:
        workspace = Workspace()

        def __init__(self, *_args):
            pass

        def start(self):
            pass

        def close(self):
            pass

    class Agent:
        async def solve(self, task, sandbox):
            _, receipt = frozen.retrieve(task, sandbox.workspace, stage="implementation", iteration=0)
            result = _scripted_evidence()
            result.repair_memory = [receipt] if valid_receipts else []
            return result

    def build(_model, **kwargs):
        assert kwargs["repair_memory_snapshot"].fingerprint == frozen.fingerprint
        return Agent()

    monkeypatch.setattr("code_agent.evaluation.DockerSandbox", Sandbox)
    monkeypatch.setattr("code_agent.evaluation.build_verified_pr_agent", build)
    report = await run_benchmark([case()], repository_root=tmp_path, default_model="scripted",
        policy=SandboxPolicy(repair_memory_enabled=True), memory_snapshot=frozen, workflow="verified_pr")
    assert report.quality_metrics_valid == valid_receipts
    assert report.sandbox_policy["repair_memory_snapshot_sha256"] == frozen.fingerprint
    assert (report.outcomes[0].failure_category == "memory_evidence_invalid") != valid_receipts


def test_issue_identity_normalizes_case_and_punctuation():
    assert issue_identity("Precision, remains VALID!")[0] == issue_identity("precision remains valid")[0]


@pytest.mark.parametrize("update", [
    {"snapshot_sha256": "b" * 64}, {"scope_sha256": "b" * 64}, {"query_sha256": ""},
    {"lesson_scores": {"b" * 64: 0.5}}, {"lesson_scores": {}}, {"context_sha256": ""},
    {"approximate_tokens": 751}, {"lesson_scores": {"b" * 64: float("nan")}},
    {"invalid_records": 1}, {"status": "no_match"},
])
def test_receipt_drift_cannot_publish_quality_metrics(tmp_path, update):
    _, snapshot, _ = make_snapshot(tmp_path)
    frozen = FrozenRepairMemory.prepare(snapshot, [case()])
    selected = CodeTask(repository="repo", issue=case().issue, benchmark_case_id=case().id)

    class Workspace:
        def list_files(self, limit):
            return ["tax.py"]

    _, receipt = frozen.retrieve(selected, Workspace(), stage="implementation", iteration=0)
    assert frozen.valid_receipts(selected, [receipt])
    assert not frozen.valid_receipts(selected, [receipt.model_copy(update=update)])


@pytest.mark.asyncio
async def test_paired_matrix_pins_same_cases_and_only_enables_memory_for_candidate(tmp_path, monkeypatch):
    from code_agent.benchmark_release import create_dataset_lock, execute_matrix

    _, snapshot, _ = make_snapshot(tmp_path)
    cases = [case()]
    (tmp_path / "cases.jsonl").write_text(cases[0].model_dump_json() + "\n")
    (tmp_path / "snapshot.json").write_text(snapshot.model_dump_json())
    lock = create_dataset_lock(cases, dataset_name="private-heldout", dataset_path="cases.jsonl")
    (tmp_path / "lock.json").write_text(lock.model_dump_json())
    matrix = {"name": "memory-on-off", "dataset": "cases.jsonl", "dataset_name": "private-heldout",
        "dataset_lock": "lock.json", "repository_root": ".", "output_dir": "outputs",
        "variants": [{"name": "off", "model": "scripted"}, {"name": "on", "model": "scripted",
            "repair_memory_snapshot": "snapshot.json", "repair_memory_snapshot_sha256": snapshot.fingerprint}]}
    path = tmp_path / "matrix.json"
    path.write_text(json.dumps(matrix))
    calls = []

    class Report:
        execution_status = "completed"
        run_id = "scripted"

    async def run(selected, **kwargs):
        calls.append((selected, kwargs))
        return Report()

    def release(*args, **kwargs):
        assert len(calls) == 2
        raise RuntimeError("stop after scripted dispatch")

    monkeypatch.setattr("code_agent.benchmark_release.run_benchmark", run)
    monkeypatch.setattr("code_agent.benchmark_release.build_release", release)
    with pytest.raises(RuntimeError, match="scripted dispatch"):
        await execute_matrix(path)
    assert calls[0][0] == calls[1][0] == cases
    assert calls[0][1]["default_model"] == calls[1][1]["default_model"] == "scripted"
    assert calls[0][1]["workflow"] == calls[1][1]["workflow"] == "verified_pr"
    assert calls[0][1]["policy"].repair_memory_enabled is False
    assert calls[1][1]["policy"].repair_memory_enabled is True
    assert "memory_snapshot" not in calls[0][1]
    assert calls[1][1]["memory_snapshot"].fingerprint == snapshot.fingerprint

    matrix["variants"][1]["repair_memory_snapshot_sha256"] = "b" * 64
    path.write_text(json.dumps(matrix))
    calls.clear()
    with pytest.raises(ValueError, match="expected fingerprint"):
        await execute_matrix(path)
    assert calls == []  # audit every arm before spending on even the baseline


@pytest.mark.asyncio
async def test_verified_stages_use_frozen_backend_and_not_environment_store(tmp_path, monkeypatch):
    from test_verified_pr import FakeSandbox, SequenceModel, analysis, approval, repository

    from code_agent.models import CodeAction
    from code_agent.verified_pr import VerifiedPRAgent

    _, snapshot, _ = make_snapshot(tmp_path)
    frozen = FrozenRepairMemory.prepare(snapshot, [case()])
    root = repository(tmp_path)
    (root / "tax.py").write_text("PRECISION = 2\n")
    policy = SandboxPolicy(repair_memory_enabled=True, max_repair_rounds=0)
    sandbox = FakeSandbox(root, policy)

    def live_store(*args, **kwargs):
        raise AssertionError("frozen experiments must never read mutable memory")

    monkeypatch.setattr("code_agent.agent.retrieve_for_task", live_store)
    actions = SequenceModel([
        CodeAction(kind="write", path="app.py", content="VALUE = 2\n"), CodeAction(kind="test"),
        CodeAction(kind="write", path="tests/test_app.py", content="from app import VALUE\ndef test_value():\n    assert VALUE == 2\n"),
        CodeAction(kind="test"),
    ])
    selected = CodeTask(repository="repo", issue=case().issue, benchmark_case_id=case().id, policy=policy)
    agent = VerifiedPRAgent(SequenceModel([analysis()]), actions, SequenceModel([approval()]), repair_memory_snapshot=frozen)
    try:
        result = await agent.solve(selected, sandbox)
    finally:
        sandbox.close()
    assert result.status == "completed"
    assert len(result.repair_memory) == 4
    assert {receipt.stage for receipt in result.repair_memory} == {"implementation", "test_author"}
    assert all(receipt.status == "selected" for receipt in result.repair_memory)
    assert frozen.valid_receipts(selected, result.repair_memory)


def test_empty_scope_receipts_and_unaudited_tasks(tmp_path):
    _, snapshot, _ = make_snapshot(tmp_path)
    frozen = FrozenRepairMemory.prepare(snapshot, [case()])
    selected = CodeTask(repository="repo", issue=case().issue, benchmark_case_id=case().id)

    class Workspace:
        def list_files(self, limit):
            return []

    _, receipt = frozen.retrieve(selected, Workspace(), stage="implementation", iteration=0)
    assert receipt.status == "no_match"
    assert frozen.valid_receipts(selected, [receipt])
    assert not frozen.valid_receipts(selected, [receipt.model_copy(update={"approximate_tokens": 1})])
    other = selected.model_copy(update={"benchmark_case_id": "unaudited"})
    assert not frozen.valid_receipts(other, [receipt])
    with pytest.raises(ValueError, match="audited"):
        frozen.retrieve(other, Workspace(), stage="implementation", iteration=0)
    with pytest.raises(ValueError, match="supported"):
        prepare_memory([case()], SandboxPolicy(), frozen, "verified_pr")
    with pytest.raises(ValueError, match="audited"):
        prepare_memory([case()], SandboxPolicy(repair_memory_enabled=True), snapshot, "verified_pr")


def test_invalid_lineage_corrupt_payload_and_bounded_findings(tmp_path):
    store, snapshot, identifier = make_snapshot(tmp_path)
    source = SourceLineage(case_id="extra", family_id="independent-family", dataset_sha256="a" * 64,
        issue="An independent original issue description.", completed_at=999)
    with pytest.raises(ValueError, match="blank"):
        SourceLineage.model_validate({**source.model_dump(), "family_id": " "})
    with pytest.raises(ValueError, match="exactly"):
        freeze_memory(store, project="project", namespace="alice", repositories=["repo"],
                      lineage={identifier: source, "b" * 64: source}, frozen_at=1001)
    with pytest.raises(ValueError, match="unique"):
        audit_snapshot(snapshot, [case(), case()])
    origin = snapshot.repositories["repo"].entries[0]
    selected = [case(id=f"heldout-{index}", issue="A monetary tax adjustment loses precision on fractional currency values.",
                     metadata={"family_id": origin.source_family_id}) for index in range(400)]
    report = audit_snapshot(snapshot, selected)
    assert report["findings_capped"] and not report["passed"]
    path = tmp_path / "invalid.json"
    path.write_text('{"PRIVATE_ISSUE": "must not appear in errors"}')
    with pytest.raises(ValueError, match="could not be read or validated") as error:
        FrozenRepairMemory.load(path, "a" * 64, [case()])
    assert "PRIVATE_ISSUE" not in str(error.value)
