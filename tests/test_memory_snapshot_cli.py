import json
import sys

import pytest

from code_agent.memory_snapshot import SourceLineage
from code_agent.memory_snapshot_cli import main
from code_agent.repair_memory import RepairMemoryStore
from evals.repair_memory_evaluation import _scripted_evidence


def test_freeze_cli_never_overwrites_private_snapshot(tmp_path, monkeypatch, capsys):
    import time

    store = RepairMemoryStore(tmp_path / "memory.db")
    identifier = store.approve(_scripted_evidence(), project="project", namespace="alice", repository="repo",
        problem="Incorrect tax rounding precision", lesson="Preserve decimal precision and validate boundary amounts.",
        applicable_paths=["tax.py"], reviewer="owner", approve=True)
    lineage = tmp_path / "lineage.json"
    lineage.write_text(json.dumps({identifier: SourceLineage(case_id="training", family_id="training-family",
        dataset_sha256="a" * 64, issue="Precision is lost for fractional tax currency values.", completed_at=time.time() - 60).model_dump()}))
    output = tmp_path / "snapshot.json"
    monkeypatch.setattr(sys, "argv", ["memory-snapshot", "freeze", "--store", str(store.path), "--project", "project",
        "--namespace", "alice", "--repositories", "repo", "--lineage", str(lineage), "--output", str(output)])
    main()
    original = output.read_bytes()
    assert len(json.loads(capsys.readouterr().out)["snapshot_sha256"]) == 64
    with pytest.raises(SystemExit, match="1"):
        main()
    assert output.read_bytes() == original


def test_cli_does_not_print_private_invalid_lineage(tmp_path, monkeypatch, capsys):
    lineage = tmp_path / "lineage.json"
    lineage.write_text(json.dumps({"lesson": {"issue": "PRIVATE_SECRET_ISSUE_BODY"}}))
    monkeypatch.setattr(sys, "argv", ["memory-snapshot", "freeze", "--project", "project", "--namespace", "alice",
        "--repositories", "repo", "--lineage", str(lineage), "--output", str(tmp_path / "snapshot.json")])
    with pytest.raises(SystemExit, match="1"):
        main()
    assert "PRIVATE_SECRET_ISSUE_BODY" not in capsys.readouterr().err


def test_audit_cli_writes_opaque_report_and_rejects_overlap(tmp_path, monkeypatch, capsys):
    from test_memory_snapshot import case, make_snapshot

    _, snapshot, _ = make_snapshot(tmp_path)
    path = tmp_path / "snapshot.json"
    path.write_text(snapshot.model_dump_json())
    dataset = tmp_path / "cases.jsonl"
    dataset.write_text(case(id="training-case").model_dump_json() + "\n")
    output = tmp_path / "audit.json"
    monkeypatch.setattr(sys, "argv", ["memory-snapshot", "audit", str(path), str(dataset), "--output", str(output)])
    with pytest.raises(SystemExit, match="1"):
        main()
    report = json.loads(capsys.readouterr().out)
    assert not report["passed"]
    assert json.loads(output.read_text()) == report
    assert "currency values" not in json.dumps(report)
