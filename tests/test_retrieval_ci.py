from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from code_agent.context_evaluation import ContextEvalCase, DirectoryWorkspace
from code_agent.intelligence import CodeIntelligenceIndex
from code_agent.retrieval_backends import RetrievalConfig
from code_agent.retrieval_learning import (
    main as retrieval_main,
)
from code_agent.retrieval_learning import (
    mine_hard_negatives,
    train_pairwise_fusion,
    validate_fusion,
)
from scripts.ci.report_retrieval_candidate import candidate_summary, main


def evidence(tmp_path: Path, *, held: bool = False):
    root = tmp_path / "source"
    (root / "code").mkdir(parents=True)
    for name, content in {
        "worker.py": "def claim_worker_lease(): return True\n",
        "queue.py": "def enqueue_work(): return True\n",
        "noise.py": "def password_secret_list(): return []\n",
    }.items():
        (root / "code" / name).write_text(content, encoding="utf-8")
    cases = [ContextEvalCase(id="lease", query="claim worker lease", relevant_paths=["code/worker.py"])]
    index = CodeIntelligenceIndex.build(DirectoryWorkspace(root), config=RetrievalConfig())
    artifact = train_pairwise_fusion(
        mine_hard_negatives(index, cases, negatives_per_positive=2),
        negatives_per_positive=2, epochs=20,
    )
    if held:
        # Authored bad-candidate control, not a trained quality claim.
        artifact.weights = {name: -abs(value) for name, value in artifact.weights.items()}
        artifact.seal()
    report = validate_fusion(index, cases, artifact, top_k=3, max_tokens=4096)
    assert report.promotion_approved is not held
    artifact_path, report_path = tmp_path / "artifact.json", tmp_path / "report.json"
    artifact_path.write_text(artifact.model_dump_json(), encoding="utf-8")
    report_path.write_text(report.model_dump_json(), encoding="utf-8")
    dataset = tmp_path / "cases.jsonl"
    dataset.write_text(cases[0].model_dump_json() + "\n", encoding="utf-8")
    return SimpleNamespace(root=root, artifact=artifact_path, report=report_path, dataset=dataset)


@pytest.mark.parametrize("held", [False, True])
def test_ci_reports_pass_or_hold_without_activating_model(tmp_path: Path, monkeypatch, held: bool):
    fixture = evidence(tmp_path, held=held)
    summary_file = tmp_path / "summary.md"
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(summary_file))
    before = fixture.artifact.read_bytes()
    assert main([str(fixture.report), "--artifact", str(fixture.artifact)]) == 0
    text = summary_file.read_text(encoding="utf-8")
    assert ("HOLD: not approved for serving" if held else "PASS: eligible for owner review") in text
    assert "does not approve or activate" in text
    assert fixture.artifact.read_bytes() == before


@pytest.mark.parametrize("tamper", ["decision", "delta", "lineage", "cases"])
def test_ci_does_not_swallow_invalid_experimental_evidence(tmp_path: Path, tamper: str):
    fixture = evidence(tmp_path, held=True)
    report = json.loads(fixture.report.read_text(encoding="utf-8"))
    if tamper == "decision":
        report["promotion_approved"] = True
    elif tamper == "delta":
        report["ndcg_delta"] = 1
    elif tamper == "lineage":
        report["index_fingerprint"] = "unrelated-source"
    else:
        report["learned"]["outcomes"][0]["case_id"] = "different-case"
    fixture.report.write_text(json.dumps(report), encoding="utf-8")
    assert main([str(fixture.report), "--artifact", str(fixture.artifact)]) == 2


def test_ci_rejects_tampered_artifact(tmp_path: Path):
    fixture = evidence(tmp_path)
    artifact = json.loads(fixture.artifact.read_text(encoding="utf-8"))
    artifact["weights"]["lexical"] += 1
    fixture.artifact.write_text(json.dumps(artifact), encoding="utf-8")
    assert main([str(fixture.report), "--artifact", str(fixture.artifact)]) == 2


@pytest.mark.parametrize("missing", [False, True])
def test_ci_missing_or_malformed_report_is_a_code_failure(tmp_path: Path, missing: bool):
    fixture = evidence(tmp_path)
    if missing:
        fixture.report.unlink()
    else:
        fixture.report.write_text("not JSON", encoding="utf-8")
    assert main([str(fixture.report), "--artifact", str(fixture.artifact)]) == 2


def test_strict_cli_still_rejects_held_candidate(tmp_path: Path, monkeypatch):
    fixture = evidence(tmp_path, held=True)
    output = tmp_path / "strict-decision.json"
    monkeypatch.setattr(sys, "argv", [
        "retrieval-learning", "validate-fusion", str(fixture.dataset),
        "--repository-root", str(fixture.root), "--artifact", str(fixture.artifact),
        "--top-k", "3", "--require-promotion", "--output", str(output),
    ])
    with pytest.raises(SystemExit) as caught:
        retrieval_main()
    assert caught.value.code == 1
    assert json.loads(output.read_text(encoding="utf-8"))["promotion_approved"] is False
    assert candidate_summary(output, fixture.artifact)[0] is False


def test_workflows_separate_code_ci_from_strict_promotion():
    root = Path(__file__).resolve().parents[1]
    code_ci = (root / ".github/workflows/ci.yml").read_text(encoding="utf-8")
    promotion = (root / ".github/workflows/retrieval-promotion.yml").read_text(encoding="utf-8")
    evaluation = next(line for line in code_ci.splitlines() if "retrieval_learning validate-fusion" in line)
    assert "--require-promotion" not in evaluation
    assert "--check evals/experiments/code_context_fusion.json" in code_ci
    assert "report_retrieval_candidate.py" in code_ci
    assert "experimental-retrieval-decision" in code_ci
    assert "continue-on-error" not in code_ci
    assert "--require-promotion" in promotion
    assert "workflow_dispatch" in promotion
