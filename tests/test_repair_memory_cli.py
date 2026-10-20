import json
import sys

import pytest

from code_agent.repair_memory_cli import main


def test_cli_full_approval_revoke_delete_lifecycle(tmp_path, monkeypatch, capsys):
    from evals.repair_memory_evaluation import _scripted_evidence

    path = tmp_path / "trajectory.json"
    path.write_text(json.dumps({"case": {"repository": "repo"}, "result": _scripted_evidence().model_dump(mode="json")}))
    base = ["repair-memory", "--store", str(tmp_path / "store.db"),
            "--project", "project", "--namespace", "alice", "--repository", "repo"]
    monkeypatch.setattr(sys, "argv", base + ["approve", "--trajectory", str(path),
        "--problem", "Fix tax rounding precision", "--lesson", "Preserve decimal precision and verify boundary amounts",
        "--paths", "tax.py", "--reviewer", "owner", "--approve-lesson"])
    main()
    identifier = json.loads(capsys.readouterr().out)["lesson_id"]
    for command in ("revoke", "delete"):
        monkeypatch.setattr(sys, "argv", base + [command, "--lesson-id", identifier])
        main()
        assert json.loads(capsys.readouterr().out)["changed"] is True


def test_cli_requires_correct_repository_before_result_validation(tmp_path, monkeypatch):
    path = tmp_path / "trajectory.json"
    path.write_text(json.dumps({"case": {"repository": "another-repo"}, "result": {}}))
    monkeypatch.setattr(sys, "argv", ["repair-memory", "--store", str(tmp_path / "store.db"),
        "--project", "project", "--namespace", "alice", "--repository", "repo", "approve",
        "--trajectory", str(path), "--problem", "Fix rounding precision", "--lesson", "Preserve decimal precision",
        "--paths", "tax.py", "--reviewer", "owner", "--approve-lesson"])
    with pytest.raises(SystemExit, match="2"):
        main()
    assert not (tmp_path / "store.db").exists()


def test_cli_job_tenant_must_match_scope(tmp_path, monkeypatch):
    path = tmp_path / "trajectory.json"
    path.write_text(json.dumps({"request": {"repository": "repo"}, "user_id": "bob", "result": {}}))
    monkeypatch.setattr(sys, "argv", ["repair-memory", "--project", "project", "--namespace", "alice", "--repository", "repo", "approve",
        "--trajectory", str(path), "--problem", "Fix rounding precision", "--lesson", "Preserve decimal precision",
        "--paths", "tax.py", "--reviewer", "owner", "--approve-lesson"])
    with pytest.raises(SystemExit, match="2"):
        main()


def test_cli_invalid_evidence_does_not_print_raw_payload(tmp_path, monkeypatch, capsys):
    path = tmp_path / "trajectory.json"
    path.write_text(json.dumps({"case": {"repository": "repo"}, "result": {"status": "PRIVATE_SECRET_BODY"}}))
    monkeypatch.setattr(sys, "argv", ["repair-memory", "--project", "project", "--namespace", "alice", "--repository", "repo", "approve",
        "--trajectory", str(path), "--problem", "Fix rounding precision", "--lesson", "Preserve decimal precision",
        "--paths", "tax.py", "--reviewer", "owner", "--approve-lesson"])
    with pytest.raises(SystemExit, match="1"):
        main()
    output = capsys.readouterr()
    assert "PRIVATE_SECRET_BODY" not in output.err + output.out
