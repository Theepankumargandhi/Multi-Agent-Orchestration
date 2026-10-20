from __future__ import annotations

import json

import pytest

from code_agent.dispatch_learning import fit_advisor
from code_agent.dispatch_supervision import ReviewQueuePolicy
from code_agent.dispatch_supervision_cli import main, read_json, read_many
from evals.dispatch_supervision_evaluation import authored_pool, run_controls
from evals.dispatch_supervision_evaluation import main as control_main
from tests.test_dispatch_supervision import reference


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def queue_inputs(tmp_path):
    artifact = fit_advisor(reference())
    artifact_path = write(tmp_path / "artifact.json", artifact.model_dump(mode="json"))
    pool = write(tmp_path / "pool.json", [row.model_dump(mode="json") for row in authored_pool()])
    policy = write(tmp_path / "policy.json", ReviewQueuePolicy(review_budget_slots=6, adjudication_reserve_slots=2).model_dump())
    return artifact, artifact_path, pool, policy


def test_authored_controls_remain_deterministic_and_held():
    report = run_controls()
    assert report == run_controls() and report["controls_passed"]
    assert report["decision"] == "held" and report["provider_calls"] == 0 and not report["serving_activation"]
    assert report["consensus"]["training_only"] and report["shift"]["status"] == "shifted"


def test_control_cli_is_immutable_and_never_promotes(tmp_path):
    output = tmp_path / "controls"
    args = ["--output-dir", str(output)]
    assert control_main(args) == 0
    before = (output / "controls.json").read_bytes()
    assert control_main(args) == 2 and (output / "controls.json").read_bytes() == before
    assert control_main(["--output-dir", str(tmp_path / "strict"), "--require-gate"]) == 1
    assert control_main(["--output-dir", "code_agent/protected-control"]) == 2


def test_queue_annotate_consensus_pipeline_preserves_inputs_and_blinding(tmp_path):
    artifact, artifact_path, pool, policy = queue_inputs(tmp_path)
    before = pool.read_bytes()
    output = tmp_path / "queue"
    args = ["queue", "--pool", str(pool), "--artifact", str(artifact_path), "--artifact-fingerprint", artifact.fingerprint,
            "--policy", str(policy), "--created-at", "2026-10-02T00:00:00Z", "--output-dir", str(output)]
    assert main(args) == 0 and pool.read_bytes() == before
    assert main(args) == 2
    packet = output / "reviewer-packet.json"
    packet_before = packet.read_bytes()
    snapshots = read_json(packet)["snapshots"]
    assert len(snapshots) == 2
    assert "priority" not in packet.read_text() and "proposed_role" not in packet.read_text()
    for row in snapshots:
        for index in range(2):
            review_path = tmp_path / "reviews" / f"{row['record_id']}-{index}.json"
            annotate = ["annotate", "--packet", str(packet), "--record-id", row["record_id"], "--reviewer-id", f"reviewer-{index}",
                "--preferred-role", "defensive", "--label-source", "human", "--independent-review",
                "--reviewed-at", "2026-10-02T12:00:00Z", "--output", str(review_path)]
            assert main(annotate) == 0 and main(annotate) == 2
    assert packet.read_bytes() == packet_before
    consolidated = tmp_path / "consensus"
    assert main(["consensus", "--batch", str(output / "private-batch.json"), "--reviews", str(tmp_path / "reviews"),
                 "--output-dir", str(consolidated)]) == 0
    records = read_json(consolidated / "training-reviews.json")
    assert len(records) == 2 and all(row["split"] == "train" and row["label_source"] == "human" for row in records)
    audit = read_json(consolidated / "consensus-audit.json")
    assert audit["unresolved_records"] == 0 and not audit["serving_activation"]
    assert main(["consensus", "--batch", str(output / "private-batch.json"), "--reviews", str(tmp_path / "reviews"),
                 "--output-dir", str(consolidated)]) == 2


def test_cli_conflict_requires_separate_adjudicator_and_preserves_synthetic(tmp_path):
    report = run_controls()
    batch = report["batch"]
    batch_path = write(tmp_path / "batch.json", batch)
    from code_agent.dispatch_supervision import ReviewBatch
    from evals.dispatch_supervision_evaluation import authored_reviews
    typed = ReviewBatch.model_validate(batch)
    reviews = authored_reviews(typed)
    reviews_path = write(tmp_path / "reviews.json", [row.model_dump(mode="json") for row in reviews])
    packet = write(tmp_path / "packet.json", typed.reviewer_packet())
    row = typed.snapshots[0]
    adjudication = tmp_path / "adjudication.json"
    args = ["adjudicate", "--packet", str(packet), "--record-id", row.record_id, "--reviewer-id", "third-reviewer",
        "--preferred-role", "defensive", "--label-source", "human", "--independent-review", "--reason", "execution_evidence",
        "--reviewed-at", "2026-10-02T12:00:00Z", "--output", str(adjudication)]
    assert main(args) == 0
    output = tmp_path / "resolved"
    assert main(["consensus", "--batch", str(batch_path), "--reviews", str(reviews_path), "--adjudications", str(adjudication),
                 "--output-dir", str(output)]) == 0
    audit = read_json(output / "consensus-audit.json")
    assert audit["unresolved_records"] == 0 and all(row["label_source"] == "synthetic" for row in audit["records"])
    args[args.index("third-reviewer")] = reviews[0].reviewer_id
    args[-1] = str(tmp_path / "same-reviewer.json")
    assert main(args) == 0  # Binding is checked when the private batch and all labels meet.
    bad_output = tmp_path / "bad-resolved"
    assert main(["consensus", "--batch", str(batch_path), "--reviews", str(reviews_path), "--adjudications", args[-1],
                 "--output-dir", str(bad_output)]) == 2 and not bad_output.exists()


@pytest.mark.parametrize("mutation", ["wrong_pin", "protected_output", "input_overwrite", "future", "invalid_policy"])
def test_queue_cli_rejects_bad_inputs_without_creating_output(tmp_path, mutation):
    artifact, artifact_path, pool, policy = queue_inputs(tmp_path)
    output = tmp_path / "must-not-exist"
    pin, created = artifact.fingerprint, "2026-10-02T00:00:00Z"
    if mutation == "wrong_pin":
        pin = "b" * 64
    elif mutation == "protected_output":
        output = "code_agent/protected-output"
    elif mutation == "input_overwrite":
        output = pool
    elif mutation == "future":
        created = "2026-09-01T00:00:00Z"
    else:
        write(policy, {"review_budget_slots": 1})
    assert main(["queue", "--pool", str(pool), "--artifact", str(artifact_path), "--artifact-fingerprint", pin,
                 "--policy", str(policy), "--created-at", created, "--output-dir", str(output)]) == 2
    assert not (tmp_path / "must-not-exist").exists()


def test_shift_cli_reports_without_serving_and_refuses_overwrite(tmp_path):
    artifact, artifact_path, pool, _ = queue_inputs(tmp_path)
    source = write(tmp_path / "reference.json", [row.model_dump(mode="json") for row in reference()])
    output = tmp_path / "shift.json"
    args = ["shift", "--reference", str(source), "--pool", str(pool), "--artifact", str(artifact_path),
            "--artifact-fingerprint", artifact.fingerprint, "--output", str(output)]
    assert main(args) == 0
    report = read_json(output)
    assert report["status"] == "stable" and not report["serving_activation"]
    before = output.read_bytes()
    assert main(args) == 2 and output.read_bytes() == before


def test_read_bounds_and_annotation_role_and_timestamp_checks(tmp_path):
    oversized = tmp_path / "oversized.json"
    oversized.write_text(" " * 2_000_001, encoding="utf-8")
    with pytest.raises(ValueError):
        read_json(oversized)
    with pytest.raises(ValueError):
        read_many(tmp_path / "empty", object)
    report = run_controls()
    from code_agent.dispatch_supervision import ReviewBatch
    packet = write(tmp_path / "packet.json", ReviewBatch.model_validate(report["batch"]).reviewer_packet())
    row = read_json(packet)["snapshots"][0]
    args = ["annotate", "--packet", str(packet), "--record-id", row["record_id"], "--reviewer-id", "reviewer-1",
        "--preferred-role", "defensive", "--independent-review", "--label-source", "human",
        "--reviewed-at", "2026-09-01T00:00:00Z", "--output", str(tmp_path / "bad-label.json")]
    assert main(args) == 2
    args[args.index(row["record_id"])] = "b" * 64
    assert main(args) == 2
