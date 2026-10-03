from pathlib import Path

import pytest

from code_agent.context_evaluation import DirectoryWorkspace
from code_agent.intelligence import CodeIntelligenceIndex
from code_agent.retrieval_backends import RetrievalConfig
from evals.retrieval_robustness import DEFAULT_PLAN, RobustnessPlan, main, markdown, perturb, run
from evals.retrieval_span_evaluation import DEFAULT_DATASET, DEFAULT_ROOT, load_cases, verify_report


@pytest.fixture(scope="module")
def source():
    return CodeIntelligenceIndex.build(DirectoryWorkspace(DEFAULT_ROOT), config=RetrievalConfig(prefer_tree_sitter=False))


def plan():
    values = RobustnessPlan.model_validate_json(DEFAULT_PLAN.read_text()).model_dump()
    values["comparison"]["bootstrap_samples"] = 100
    return RobustnessPlan.model_validate(values)


@pytest.mark.parametrize("name", ["whitespace", "polite_wrapper", "irrelevant_terms", "instruction_noise"])
def test_transforms_preserve_targets_and_reset_review_provenance(name):
    case = load_cases(DEFAULT_DATASET)[0]
    case.review_status, case.reviewer, case.reviewed_at = "human_reviewed", "owner", "2026-10-02T00:00:00Z"
    transformed = perturb(case, name)
    assert transformed.query != case.query
    assert transformed.id == case.id and transformed.family_id == case.family_id
    assert transformed.required_spans == case.required_spans
    assert transformed.review_status == "synthetic_seed" and not transformed.reviewer
    assert case.review_status == "human_reviewed"


def test_deterministic_source_bound_family_paired_report(source):
    cases = load_cases(DEFAULT_DATASET)
    first, second = run(source, cases, plan()), run(source, cases, plan())
    assert first == second and verify_report(first)
    assert first["families"] == 6 and first["cases"] == 12
    assert first["decision"] == "held" and not first["production_activation"]
    for point in first["comparisons"]:
        for arm in point["arms"].values():
            assert arm["worst_case_coverage"]["span_recall"] <= arm["clean"]["span_recall"]
            assert all(len(stress["paired_clean_to_stress_95_ci"]) == 2 for stress in arm["stress"].values())
    assert not any(case.query in str(first) for case in cases)
    assert "Worst-case" in markdown(first)
    first["numerical_gate_passed"] = not first["numerical_gate_passed"]
    assert not verify_report(first)


def test_bad_variants_and_oversized_queries_fail(source):
    values = {**plan().model_dump(), "perturbations": ["whitespace", "whitespace"]}
    with pytest.raises(ValueError, match="unique"):
        run(source, load_cases(DEFAULT_DATASET), RobustnessPlan.model_validate(values))
    with pytest.raises(ValueError):
        RobustnessPlan(perturbations=["invented"])
    with pytest.raises(ValueError, match="unknown"):
        perturb(load_cases(DEFAULT_DATASET)[0], "invented")
    case = load_cases(DEFAULT_DATASET)[0].model_copy(update={"query": "long question " * 300})
    with pytest.raises(ValueError):
        perturb(case, "polite_wrapper")


def test_stale_label_is_not_hidden(source):
    cases = load_cases(DEFAULT_DATASET)
    cases[0].required_spans[0].sha256 = "0" * 64
    with pytest.raises(ValueError, match="stale"):
        run(source, cases, plan())


def test_cli_protects_inputs_source_and_reports(tmp_path):
    output = tmp_path / "review.json"
    assert main(["--output", str(output)]) == 0
    assert output.with_suffix(".md").exists()
    original = output.read_bytes()
    assert main(["--output", str(output)]) == 2 and output.read_bytes() == original
    assert main(["--output", str(DEFAULT_DATASET)]) == 2
    assert main(["--output", str(DEFAULT_ROOT / "src/report.json")]) == 2
    assert main(["--output", str(tmp_path / "report.md")]) == 2
    assert main(["--output", str(tmp_path / "strict.json"), "--require-gate"]) == 1


def test_tolerance_is_protocol_bound(source):
    base = plan()
    changed = RobustnessPlan.model_validate({**base.model_dump(), "maximum_span_drop": 0.0})
    cases = load_cases(DEFAULT_DATASET)
    assert run(source, cases, base)["protocol_fingerprint"] != run(source, cases, changed)["protocol_fingerprint"]


def test_ci_has_no_activation_authority():
    workflow = (Path(__file__).resolve().parents[1] / ".github/workflows/ci.yml").read_text()
    command = next(line for line in workflow.splitlines() if "python -m evals.retrieval_robustness" in line)
    assert "--require-gate" not in command
    assert "retrieval-robustness-comparison" in workflow


def test_degraded_stress_control_is_caught(source, monkeypatch):
    original = source.select
    empty = CodeIntelligenceIndex.build(DirectoryWorkspace(DEFAULT_ROOT), config=RetrievalConfig(prefer_tree_sitter=False))
    empty.files = {}

    def degraded(query, **kwargs):
        if "Quoted untrusted text" in query and kwargs["packing_policy"] == "balanced_v1":
            return empty.select(query, **kwargs)
        return original(query, **kwargs)

    monkeypatch.setattr(source, "select", degraded)
    report = run(source, load_cases(DEFAULT_DATASET), plan())
    assert not report["numerical_gate_passed"] and not report["production_activation"]
    assert any("drop exceeds" in reason for reason in report["reasons"])
    assert any("worst-case" in reason for reason in report["reasons"])
    assert report["comparisons"][1]["arms"]["candidate"]["worst_case_coverage"]["span_recall"] == 0
