from __future__ import annotations

import json
from pathlib import Path

import pytest

from code_agent.context_evaluation import DirectoryWorkspace
from code_agent.intelligence import CodeIntelligenceIndex
from code_agent.retrieval_backends import RetrievalConfig
from evals.retrieval_span_evaluation import (
    DEFAULT_DATASET,
    DEFAULT_ROOT,
    SpanPlan,
    evaluate,
    load_cases,
    receipt_fingerprint,
    visible_lines,
)
from tests.test_code_intelligence import MemoryWorkspace, sample_workspace


def index(files=None, **config):
    workspace = MemoryWorkspace(files) if files is not None else DirectoryWorkspace(DEFAULT_ROOT)
    return CodeIntelligenceIndex.build(workspace, config=RetrievalConfig(prefer_tree_sitter=False, **config))


def test_packing_ablation_changes_only_the_packer_and_recovers_fixture_exit_lines():
    plan = SpanPlan.model_validate_json(Path("evals/experiments/context_packing_plan.json").read_text())
    report = evaluate(index(), load_cases(DEFAULT_DATASET), plan)
    assert plan.baseline == plan.candidate
    assert report["decision"] == "held" and not report["production_activation"]
    for point in report["comparisons"]:
        assert point["baseline"]["span_recall"] == pytest.approx(5 / 6)
        assert point["candidate"]["span_recall"] == 1
        assert point["candidate"]["file_recall"] == point["baseline"]["file_recall"] == 1
        assert point["paired_family_bootstrap_95_ci"][0] >= 0


@pytest.mark.parametrize("budget", [64, 96, 128, 256, 512, 1024])
def test_balanced_packs_are_bounded_deterministic_and_source_complete(budget):
    source = index()
    for case in load_cases(DEFAULT_DATASET):
        first = source.select(case.query, top_k=4, max_tokens=budget, packing_policy="balanced_v1")
        second = source.select(case.query, top_k=4, max_tokens=budget, packing_policy="balanced_v1")
        assert first == second
        assert first.receipt.estimated_tokens <= budget
        assert first.receipt.context_chars == len(first.prompt_context)
        assert first.receipt.fingerprint == receipt_fingerprint(first.receipt)
        assert first.receipt.packing_diagnostics["partial_source_lines"] == 0
        observed = visible_lines(source, first)
        assert first.receipt.packing_diagnostics["retained_source_lines"] == len(observed)


def test_duplicate_code_in_distinct_files_keeps_distinct_provenance():
    code = "def authorize_release(owner):\n    return owner == 'approved-owner'\n"
    source = index({"app/a.py": code, "app/b.py": code})
    pack = source.select("authorize_release approved owner", top_k=2, max_tokens=512, packing_policy="balanced_v1")
    observed = visible_lines(source, pack)
    assert ("app/a.py", 2) in observed and ("app/b.py", 2) in observed
    assert pack.receipt.duplicate_lines_removed == 0
    assert pack.receipt.source_line_ranges == {"app/b.py": [[1, 2]], "app/a.py": [[1, 2]]}


def test_huge_line_is_omitted_instead_of_emitting_a_partial_prefix():
    source = index({"app/long.py": "def inspect_blob():\n    blob = '" + "x" * 5000 + "'\n    return True\n"})
    pack = source.select("inspect_blob", top_k=1, max_tokens=128, packing_policy="balanced_v1")
    assert pack.receipt.estimated_tokens <= 128
    assert "blob =" not in pack.prompt_context
    visible_lines(source, pack)


def test_default_and_explicit_legacy_rendering_match_exactly():
    source = index()
    for case in load_cases(DEFAULT_DATASET):
        implicit = source.select(case.query, max_tokens=512)
        explicit = source.select(case.query, max_tokens=512, packing_policy="legacy")
        assert implicit == explicit
        assert implicit.receipt.packing_policy == "legacy"
        assert not implicit.receipt.source_line_ranges


def test_policy_is_opt_in_and_invalid_values_fail_closed(monkeypatch):
    monkeypatch.delenv("CODE_CONTEXT_PACKING_POLICY", raising=False)
    assert RetrievalConfig.from_environment().packing_policy == "legacy"
    monkeypatch.setenv("CODE_CONTEXT_PACKING_POLICY", "balanced_v1")
    assert RetrievalConfig.from_environment().packing_policy == "balanced_v1"
    with pytest.raises(ValueError, match="unknown"):
        RetrievalConfig(packing_policy="untrusted")
    with pytest.raises(ValueError, match="packing_policy"):
        index().select("render report", packing_policy="untrusted")
    configured = index(packing_policy="balanced_v1").select("render report")
    assert configured.receipt.packing_policy == "balanced_v1"


def test_policy_and_line_audit_are_integrity_bound():
    source = index()
    pack = source.select("render_evaluation_report", max_tokens=512, packing_policy="balanced_v1")
    assert pack.receipt.fingerprint == receipt_fingerprint(pack.receipt)
    pack.receipt.source_line_ranges["src/report.py"] = [[1, 35]]
    with pytest.raises(ValueError, match="provenance receipt"):
        visible_lines(source, pack)
    # Resealing incorrect metadata still fails against actual source sections.
    pack.receipt.fingerprint = receipt_fingerprint(pack.receipt)
    with pytest.raises(ValueError, match="audit contradicts"):
        visible_lines(source, pack)


def test_empty_balanced_pack_has_a_valid_bounded_receipt():
    source = index({})
    pack = source.select("missing source", max_tokens=64, packing_policy="balanced_v1")
    assert pack.receipt.fingerprint == receipt_fingerprint(pack.receipt)
    assert pack.receipt.context_chars == len(pack.prompt_context)
    assert not visible_lines(source, pack)


def test_multi_language_snippets_remain_source_bound():
    source = CodeIntelligenceIndex.build(sample_workspace(), config=RetrievalConfig(prefer_tree_sitter=False))
    for query in ["authenticate_user", "handleLogin", "validateToken", "ClaimLease", "retry_job"]:
        pack = source.select(query, max_tokens=256, top_k=3, packing_policy="balanced_v1")
        assert visible_lines(source, pack)
        assert pack.receipt.estimated_tokens <= 256


def test_wrong_packing_policy_is_rejected_by_the_evaluator(monkeypatch):
    source = index()
    original = source.select

    def force_legacy(query, **kwargs):
        return original(query, **{**kwargs, "packing_policy": "legacy"})

    monkeypatch.setattr(source, "select", force_legacy)
    plan = SpanPlan(candidate_packing="balanced_v1", bootstrap_samples=100)
    with pytest.raises(ValueError, match="different packing policy"):
        evaluate(source, load_cases(DEFAULT_DATASET), plan)


def test_plan_is_serializable_before_running_and_separates_protocols():
    baseline = SpanPlan(bootstrap_samples=100)
    candidate = baseline.model_copy(update={"candidate_packing": "balanced_v1"})
    left = evaluate(index(), load_cases(DEFAULT_DATASET), baseline)
    right = evaluate(index(), load_cases(DEFAULT_DATASET), candidate)
    assert left["protocol_fingerprint"] != right["protocol_fingerprint"]
    assert json.loads(candidate.model_dump_json())["candidate_packing"] == "balanced_v1"
