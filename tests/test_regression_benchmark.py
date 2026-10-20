from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from code_agent.regression_benchmark import (
    IndependentReview,
    RegressionCase,
    RegressionCorpus,
    case_inputs,
    freeze_corpus,
    review_kit,
    run_benchmark,
    summarize,
)
from code_agent.upgrade_models import fingerprint

ROOT = Path(__file__).parent.parent
IMAGE = "sha256:" + "a" * 64


def corpus():
    return RegressionCorpus.model_validate_json((ROOT / "evals/datasets/patch_regression_controls.json").read_bytes())


def test_freeze_controls_and_real_case_gate():
    frozen = freeze_corpus(corpus(), ROOT, image_resolver=lambda _: IMAGE)
    assert frozen["independently_reviewed_real_cases"] == 0
    assert not frozen["official_benchmark_result"]
    assert len(frozen["bindings"]) == 4
    with pytest.raises(ValueError, match="not enough"):
        freeze_corpus(corpus(), ROOT, minimum_real_cases=10, image_resolver=lambda _: IMAGE)


@pytest.mark.parametrize("change", [
    {"evidence_kind": "historical_real_patch"}, {"repository": "../outside"},
    {"patch": ".env"}, {"review": {"submitter_id": "same", "reviewer_id": "same"}},
])
def test_corpus_cannot_launder_controls_or_unsafe_paths(change):
    with pytest.raises(ValueError):
        RegressionCase.model_validate({**corpus().cases[0].model_dump(), **change})


def test_family_split_leakage_is_rejected():
    payload = corpus().model_dump()
    payload["cases"][1]["split"] = "development"
    with pytest.raises(ValueError, match="families"):
        RegressionCorpus.model_validate(payload)


def test_real_review_needs_distinct_identities():
    with pytest.raises(ValueError, match="different reviewer"):
        IndependentReview(submitter_id="Owner", reviewer_id=" owner ", evidence_url="https://example.com/review", artifacts_sha256="a" * 64, conclusion="regression")


def test_json_only_domains_integrate_with_review_intake_without_bypassing_review(tmp_path):
    from code_agent.patch_inputs import PatchVerifySpec

    case = corpus().cases[0]
    _, original, patch, _ = case_inputs(ROOT, case)
    shutil.copytree(ROOT / case.repository, tmp_path / "snapshot")
    payload = original.model_dump()
    payload["integer_domains"] = []
    payload["json_domains"] = [{"domain_id": "structured", "entrypoint": original.integer_domains[0].entrypoint,
                                "input_schema": {"type": "integer", "minimum": 0, "maximum": 5}}]
    spec = PatchVerifySpec.model_validate(payload)
    (tmp_path / "spec.json").write_text(spec.model_dump_json())
    (tmp_path / "candidate.diff").write_text(patch)
    intake = RegressionCase.model_validate({**case.model_dump(), "repository": "snapshot", "spec": "spec.json",
                                           "patch": "candidate.diff", "evidence_kind": "historical_real_patch",
                                           "patch_origin": "upstream_commit", "upstream_url": "https://github.com/example/unit",
                                           "upstream_commit": "a" * 40})
    assert case_inputs(tmp_path, intake)[1].json_domains
    assert review_kit(intake, tmp_path, "authored-unit")["status"] == "awaiting_independent_review_not_admitted"
    with pytest.raises(ValueError, match="without independent"):
        freeze_corpus(RegressionCorpus(corpus_id="unit", cases=[intake]), tmp_path, image_resolver=lambda _: IMAGE)


def test_review_intake_does_not_invent_independent_review():
    values = corpus().cases[0].model_dump()
    values.update(evidence_kind="historical_real_patch", patch_origin="historical_reversal",
                  upstream_url="https://github.com/example/example", upstream_commit="a" * 40)
    # These are synthetic metadata for this unit control, not a checked-in real case.
    intake = RegressionCase.model_validate(values)
    kit = review_kit(intake, ROOT, "submitter")
    assert kit["status"] == "awaiting_independent_review_not_admitted"
    assert kit["review_template"]["reviewer_id"] == "" and kit["review_template"]["evidence_url"] == ""
    with pytest.raises(ValueError, match="without independent"):
        freeze_corpus(RegressionCorpus(corpus_id="pending", cases=[intake]), ROOT, image_resolver=lambda _: IMAGE)


def test_review_binds_exact_source_patch_spec_and_license():
    case = corpus().cases[0]
    _, _, _, artifacts = case_inputs(ROOT, case)
    review = {"submitter_id": "author", "reviewer_id": "different-reviewer", "evidence_url": "https://example.com/review",
              "artifacts_sha256": fingerprint(artifacts), "conclusion": "regression"}
    real = RegressionCase.model_validate({**case.model_dump(), "evidence_kind": "historical_real_patch",
        "patch_origin": "historical_reversal", "upstream_url": "https://github.com/example/example",
        "upstream_commit": "b" * 40, "review": review})
    assert case_inputs(ROOT, real)[3] == artifacts
    tampered = real.model_dump()
    tampered["review"]["artifacts_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="exact artifacts"):
        case_inputs(ROOT, RegressionCase.model_validate(tampered))


@pytest.mark.asyncio
async def test_stale_source_or_implementation_blocks_before_verifier(tmp_path):
    shutil.copytree(ROOT / "tests/fixtures/patch_verify", tmp_path / "tests/fixtures/patch_verify")
    frozen = freeze_corpus(corpus(), tmp_path, image_resolver=lambda _: IMAGE)
    (tmp_path / "tests/fixtures/patch_verify/app.py").write_text("raise RuntimeError('changed')\n")

    async def forbidden(*args, **kwargs):
        raise AssertionError("must not run stale benchmark")

    with pytest.raises(ValueError, match="changed since freeze"):
        await run_benchmark(frozen, tmp_path, verifier=forbidden, image_resolver=lambda _: IMAGE)
    frozen["corpus"]["corpus_id"] = "tampered"
    with pytest.raises(ValueError, match="integrity"):
        await run_benchmark(frozen, tmp_path, verifier=forbidden, image_resolver=lambda _: IMAGE)


def test_missing_results_do_not_disappear_and_zero_real_metrics_are_null():
    rows = [{"mode": "generated", "evidence_kind": "authored_control", "expected": expected,
             "decision": decision, "duration_ms": 5, "provider_cost_usd": 0, "tests_green_eligible": decision != "inconclusive"}
            for expected, decision in (("regression", "alarm"), ("regression", "inconclusive"), ("correct", "no_alarm"), ("correct", "inconclusive"))]
    result = summarize(rows)
    controls = result["authored_control"]["generated"]
    assert controls["declared_cases"] == 4 and controls["regression_detection_rate"] == 0.5
    assert controls["inconclusive_cases"] == 2 and controls["false_alarm_rate"] == 0
    assert result["historical_real_patch"]["generated"]["regression_detection_rate"] is None
    assert controls["infrastructure_cost_usd"] is None


@pytest.mark.asyncio
async def test_benchmark_records_three_matched_strategies_and_retains_hold():
    data = corpus()
    data = RegressionCorpus.model_validate({**data.model_dump(), "cases": [data.cases[0].model_dump()]})
    frozen = freeze_corpus(data, ROOT, image_resolver=lambda _: IMAGE)
    calls = []

    async def scripted(repository, patch, spec, *, mode):
        calls.append((spec.task_id, mode, spec.image))
        result = {"status": "inconclusive", "reason": "execution_unavailable", "trials": [], "source_unchanged": True,
                  "duration_ms": 5, "provider_cost_usd": 0}
        result["fingerprint"] = fingerprint(result)
        return result

    report = await run_benchmark(frozen, ROOT, verifier=scripted, image_resolver=lambda _: IMAGE)
    assert len(calls) == 3 and all(row[2] == IMAGE for row in calls)
    assert len(report["rows"]) == 3 and all(row["decision"] == "inconclusive" for row in report["rows"])
    assert report["independently_reviewed_real_cases"] == 0
    assert "owner" not in json.dumps(report["metrics"])
