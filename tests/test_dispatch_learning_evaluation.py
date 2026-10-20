import json

from code_agent.dispatch_learning import fingerprint
from code_agent.dispatch_learning_cli import fresh_output, load_records, main
from evals.dispatch_learning_evaluation import authored_records, run_controls
from evals.dispatch_learning_evaluation import main as controls_main


def test_authored_controls_are_deterministic_held_and_non_serving():
    artifact, report, passed = run_controls()
    assert passed and run_controls() == (artifact, report, passed)
    assert report["decision"] == "held" and not report["serving_activation"]
    assert "synthetic_or_mixed_review_labels" in report["blocking_reasons"]
    assert report["fingerprint"] == fingerprint({k: v for k, v in report.items() if k != "fingerprint"})
    assert len(report["risk_coverage_curve"]) == 6
    assert report["metrics"]["test"]["coverage"] == 0.75
    assert report["risk_coverage_curve"][-1]["accepted_review_agreement"] is None


def test_controls_cli_never_overwrites_and_strict_gate_rejects_synthetic(tmp_path):
    output = tmp_path / "controls"
    assert controls_main(["--output-dir", str(output)]) == 0
    before = (output / "evaluation.json").read_bytes()
    assert controls_main(["--output-dir", str(output)]) == 2
    assert (output / "evaluation.json").read_bytes() == before
    assert controls_main(["--output-dir", str(tmp_path / "strict"), "--require-gate"]) == 1
    assert controls_main(["--output-dir", "code_agent/protected-controls"]) == 2
    control = json.loads((output / "controls.json").read_text())
    assert control["provider_calls"] == 0 and control["controls_passed"]


def test_review_is_separate_explicit_and_snapshot_remains_immutable(tmp_path):
    record = authored_records()[0]
    source = tmp_path / "snapshot.json"
    source.write_text(json.dumps([record.snapshot.model_dump(mode="json")]), encoding="utf-8")
    before = source.read_bytes()
    output = tmp_path / "review.json"
    args = ["review", "--snapshot", str(source), "--record-id", record.snapshot.record_id,
        "--preferred-role", "defensive", "--reviewer-id", "reviewer-1", "--independent-review",
        "--label-source", "human", "--reviewed-at", "2026-09-02T00:00:00Z", "--split", "train", "--output", str(output)]
    assert main(args) == 0 and source.read_bytes() == before
    reviewed = load_records(output)
    assert reviewed[0].label_source == "human" and reviewed[0].snapshot == record.snapshot
    assert main(args) == 2
    assert main(args[:-1] + [str(source)]) == 2 and source.read_bytes() == before
    source.write_text('[null]', encoding="utf-8")
    assert main(args[:-1] + [str(tmp_path / "bad-review.json")]) == 2


def test_train_cli_writes_new_bundle_without_serving_or_mutating_inputs(tmp_path):
    source = tmp_path / "reviews.jsonl"
    source.write_text("\n".join(row.model_dump_json() for row in authored_records()), encoding="utf-8")
    before = source.read_bytes()
    output = tmp_path / "trained"
    args = ["train", "--dataset", str(source), "--output-dir", str(output), "--require-gate"]
    assert main(args) == 1 and source.read_bytes() == before
    report = json.loads((output / "evaluation.json").read_text())
    assert report["decision"] == "held" and not report["serving_activation"]
    assert "not a probability" in (output / "review.md").read_text()
    assert main(args) == 2
    try:
        fresh_output(tmp_path / "reviews" / "nested", [tmp_path / "reviews"])
    except ValueError:
        pass
    else:
        raise AssertionError("output must not be within an input dataset")


def test_invalid_training_does_not_create_bundle(tmp_path):
    source = tmp_path / "bad.json"
    source.write_text(json.dumps([row.model_dump(mode="json") for row in authored_records()[:6]]), encoding="utf-8")
    output = tmp_path / "must-not-exist"
    assert main(["train", "--dataset", str(source), "--output-dir", str(output)]) == 2
    assert not output.exists()
