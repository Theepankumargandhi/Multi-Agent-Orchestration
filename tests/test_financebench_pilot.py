"""Pilot integrity and leakage controls on authored metadata; no external requests."""

import hashlib
import json
from pathlib import Path

import pytest

from agentforge_sdk import DocumentPage, PlannerOutput, PlannerUsage
from agentforge_sdk.groq_reasoning import GroqRequestError
from agentforge_sdk.models import fingerprint
from evals import financebench_pilot as pilot


def case(identifier="case_1", name="ATLAS_2026_10K"):
    return {"financebench_id": identifier, "doc_name": name, "question": "Atlas latency?", "answer": "GOLD PRIVATE ANSWER",
            "evidence": [{"doc_name": name, "evidence_page_num": 0, "evidence_text": "GOLD PRIVATE EVIDENCE"}]}


def cohort(tmp_path, cases=None):
    cases = cases or [case()]
    directory = tmp_path / "cohort"
    directory.mkdir()
    pdfs = directory / "pdfs"
    pdfs.mkdir()
    documents = {}
    for row in cases:
        raw = b"%PDF-authored-admission-control"
        (pdfs / (row["doc_name"] + ".pdf")).write_bytes(raw)
        documents[row["doc_name"]] = {"status": "downloaded", "sha256": hashlib.sha256(raw).hexdigest()}
    gold = json.dumps(cases).encode()
    (directory / "gold.json").write_bytes(gold)
    manifest = {"case_ids": [r["financebench_id"] for r in cases], "documents": documents,
                "gold_sha256": hashlib.sha256(gold).hexdigest()}
    manifest["fingerprint"] = fingerprint(manifest)
    (directory / "cohort.json").write_text(json.dumps(manifest), encoding="utf-8")
    return directory


@pytest.fixture
def local_output(monkeypatch):
    def create(path):
        path.mkdir(exist_ok=False)
        return path
    monkeypatch.setattr(pilot, "private_directory", create)
    monkeypatch.setattr(pilot, "load_pdf", lambda *a, **k: (DocumentPage(document_id="atlas", source="atlas.pdf", page=1,
        text="Atlas latency is 180 ms."),))


def test_cohort_selection_is_frozen_order_invariant_and_bounded():
    rows = [case(f"case_{n}") for n in range(25)]
    def encode(values):
        return "\n".join(json.dumps(r) for r in values).encode()
    assert pilot.freeze_cases(encode(rows)) == pilot.freeze_cases(encode(list(reversed(rows))))
    assert len(pilot.freeze_cases(encode(rows))) == 20
    with pytest.raises(ValueError):
        pilot.freeze_cases(encode([rows[0], rows[0]]))


@pytest.mark.parametrize("field,value", [("doc_name", "../escape"), ("financebench_id", "../escape"),
                                       ("question", ""), ("evidence", [{"doc_name": "x", "evidence_page_num": -1}])])
def test_untrusted_case_paths_and_page_numbers_fail_before_writes(field, value):
    row = {**case(), field: value}
    with pytest.raises(ValueError):
        pilot.validate_case(row)


def test_actual_and_readme_evidence_document_keys_are_supported_but_not_ambiguous():
    row = case()
    pilot.validate_case(row)
    row["evidence"][0]["evidence_doc_name"] = row["evidence"][0].pop("doc_name")
    pilot.validate_case(row)
    row["evidence"][0]["doc_name"] = "conflicting-name"
    with pytest.raises(ValueError):
        pilot.validate_case(row)


def test_changed_cohort_gold_rejected(tmp_path):
    directory = cohort(tmp_path)
    (directory / "gold.json").write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="gold"):
        pilot.load_cohort(directory)


def test_no_live_mode_without_explicit_free_plan_confirmation(tmp_path):
    with pytest.raises(ValueError, match="Free-plan"):
        pilot.run(tmp_path, tmp_path / "output", live=True)


def test_gold_never_enters_live_baseline_or_planner_context(tmp_path, monkeypatch, local_output):
    directory, captured = cohort(tmp_path), []
    monkeypatch.setattr(pilot, "_key", lambda *a: "authored-test-key-" + "x" * 30)

    def complete(self, system, payload):
        with self._lock:
            self._requests += 1
        captured.append(payload)
        if "plan_schema" in payload:
            card = next(c for c in payload["candidates"] if c["kind"] == "quote")
            answer = {"action": "quote", "candidate_ids": [card["candidate_id"]]}
        else:
            answer = {"answer": "180 ms", "candidate_ids": [payload["candidates"][0]["candidate_id"]]}
        return PlannerOutput(plan_json=json.dumps(answer), usage=PlannerUsage(input_tokens=20, output_tokens=10, provider_requests=1))

    monkeypatch.setattr(pilot.GroqJSONClient, "complete", complete)
    result = pilot.run(directory, tmp_path / "output", live=True, confirm_free_plan=True, model="authored-test-model")
    assert result["paired_cases"] == 1 and result["provider_requests_attempted"] == 2
    assert result["answer_accuracy"] is None and not result["human_review_completed"]
    assert "GOLD PRIVATE" not in json.dumps(captured)
    assert captured[0]["candidates"] == captured[1]["candidates"]


def test_provider_error_stops_all_remaining_cases_without_automatic_retry(tmp_path, monkeypatch, local_output):
    directory = cohort(tmp_path, [case("case_1"), case("case_2")])
    monkeypatch.setattr(pilot, "_key", lambda *a: "authored-test-key-" + "x" * 30)
    calls = []

    def fail(self, *args):
        self._requests += 1
        calls.append(1)
        raise GroqRequestError("rate_limited")

    monkeypatch.setattr(pilot.GroqJSONClient, "complete", fail)
    result = pilot.run(directory, tmp_path / "output", live=True, confirm_free_plan=True, model="authored-test-model")
    assert result["provider_requests_attempted"] == 1 and result["provider_stop_code"] == "rate_limited"
    assert len(calls) == 1 and result["paired_cases"] == 0


def test_offline_admission_is_not_labelled_a_live_quality_result(tmp_path, local_output):
    result = pilot.run(cohort(tmp_path), tmp_path / "output")
    assert result["mode"] == "offline_admission_only" and result["provider_requests_attempted"] == 0
    assert result["answer_accuracy"] is None


def test_review_labels_bind_to_predictions_and_never_silently_mark_abstention_correct(tmp_path, local_output):
    output = tmp_path / "output"
    pilot.run(cohort(tmp_path), output)
    labels = json.loads((output / "review-template.json").read_text())
    baseline = {**labels[0], "reviewer_id": "authored-reviewer", "answer_correct": True,
                "answer_supported": True, "citations_correct": True, "rationale": "Authored intake control, not real human annotation."}
    reviews = tmp_path / "reviews.json"
    reviews.write_text(json.dumps([baseline]), encoding="utf-8")
    score = pilot.score_reviews(output, reviews, tmp_path / "score")
    assert score["metrics"]["baseline"]["reviewed_cases"] == 1
    for key in ("variant", "case_id"):
        reviews.write_text(json.dumps([{**baseline, key: []}]), encoding="utf-8")
        with pytest.raises(ValueError, match="outside the profile"):
            pilot.score_reviews(output, reviews, tmp_path / "malformed-score")
    baseline["prediction_sha256"] = "f" * 64
    reviews.write_text(json.dumps([baseline]), encoding="utf-8")
    with pytest.raises(ValueError, match="changed prediction"):
        pilot.score_reviews(output, reviews, tmp_path / "bad-score")


def test_private_output_rejects_workspace_root_and_outside_targets(tmp_path):
    with pytest.raises(ValueError):
        pilot.private_directory(Path.cwd())
    with pytest.raises(ValueError):
        pilot.private_directory(tmp_path)
