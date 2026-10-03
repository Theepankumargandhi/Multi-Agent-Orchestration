from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import pytest

from code_agent.context_evaluation import DirectoryWorkspace
from code_agent.intelligence import CodeIntelligenceIndex, ContextPack
from code_agent.retrieval_backends import RetrievalConfig
from evals.retrieval_span_evaluation import (
    DEFAULT_DATASET,
    DEFAULT_ROOT,
    EvidenceSpan,
    SpanCase,
    SpanPlan,
    evaluate,
    family_aggregate,
    load_cases,
    main,
    paired_family_interval,
    receipt_fingerprint,
    score_pack,
    span_digest,
    validate_cases,
    validate_targets,
    verify_report,
    visible_lines,
)


@pytest.fixture
def index():
    return CodeIntelligenceIndex.build(DirectoryWorkspace(DEFAULT_ROOT), config=RetrievalConfig(prefer_tree_sitter=False))


def test_smoke_comparison_is_reproducible_private_and_synthetic_held(index):
    cases, plan = load_cases(DEFAULT_DATASET), SpanPlan(bootstrap_samples=100)
    first, second = evaluate(index, cases, plan), evaluate(index, cases, plan)
    assert first == second
    assert first["cases"] == 12 and first["families"] == 6
    assert first["decision"] == "held" and not first["production_activation"]
    assert any("synthetic" in reason for reason in first["reasons"])
    assert verify_report(first)
    serialized = json.dumps(first)
    assert all(case.query not in serialized for case in cases)
    assert "def release_lease" not in serialized
    assert [point["token_budget"] for point in first["comparisons"]] == [256, 512, 1024]
    assert any(row["file_recall"] > row["span_recall"]
               for point in first["comparisons"] for row in point["outcomes"]["candidate"])
    first["comparisons"][0]["candidate"]["span_recall"] = -1
    assert not verify_report(first)


def test_file_hit_is_not_complete_evidence(index):
    case = next(case for case in load_cases(DEFAULT_DATASET) if case.id == "report-original")
    pack = index.select(case.query, top_k=4, max_tokens=256, strategy="hybrid_rerank")
    outcome = score_pack(index, case, pack)
    assert outcome["file_recall"] == 1
    assert outcome["span_recall"] == 0
    assert outcome["failure"] in {"compression_loss", "partial_evidence"}


def test_visible_lines_rejects_tampered_receipts_and_source_lines(index):
    pack = index.select("release_lease owner expires_at", top_k=1, max_tokens=1024)
    assert ("src/leases.py", 2) in visible_lines(index, pack)
    pack.receipt.selected_files[0].snippet_sha256 = "wrong"
    with pytest.raises(ValueError, match="provenance receipt"):
        visible_lines(index, pack)


def test_truncated_source_line_does_not_receive_evidence_credit(index):
    pack = index.select("release_lease owner expires_at", top_k=1, max_tokens=1024)
    header, snippet = pack.prompt_context.split("\n\n")[1].split("\n", 1)
    snippet = "\n".join(snippet.splitlines()[:1])[:-2]
    context = pack.prompt_context.split("\n\n")[0] + "\n\n" + header + "\n" + snippet
    receipt = pack.receipt.model_copy(deep=True)
    receipt.context_chars = len(context)
    receipt.estimated_tokens = max(1, math.ceil(len(context) / 4))
    receipt.selected_files[0].snippet_sha256 = hashlib.sha256(snippet.encode()).hexdigest()
    receipt.fingerprint = receipt_fingerprint(receipt)
    assert not visible_lines(index, ContextPack(receipt=receipt, prompt_context=context))


@pytest.mark.parametrize("change", ["digest", "range", "missing"])
def test_stale_or_missing_span_labels_fail_closed(index, change):
    case = load_cases(DEFAULT_DATASET)[0].model_copy(deep=True)
    if change == "digest":
        case.required_spans[0].sha256 = "0" * 64
    elif change == "range":
        case.required_spans[0].end_line = 999
    else:
        case.required_spans[0].path = "src/absent.py"
    with pytest.raises(ValueError):
        validate_targets(index, [case])


@pytest.mark.parametrize("path", ["../outside.py", "/absolute.py", "C:/secret.py", "src\\file.py", "src/./file.py"])
def test_evidence_paths_cannot_escape_the_corpus(path):
    with pytest.raises(ValueError):
        EvidenceSpan(path=path, start_line=1, end_line=2, sha256="0" * 64)


def test_family_macro_and_bootstrap_do_not_count_paraphrases_as_independent_tasks():
    rows = [{"case_id": f"same-{i}", "family_id": "same", **dict.fromkeys(
        ["file_recall", "line_recall", "span_recall", "complete_evidence", "context_precision", "estimated_tokens"], 0.0,
    )} for i in range(100)]
    other = {"case_id": "other", "family_id": "other", **dict.fromkeys(
        ["file_recall", "line_recall", "span_recall", "complete_evidence", "context_precision", "estimated_tokens"], 1.0,
    )}
    assert family_aggregate(rows + [other])["span_recall"] == 0.5
    candidate = [{**row, "span_recall": 1.0} for row in rows] + [other]
    assert paired_family_interval(rows + [other], candidate, seed="fixed", samples=200) == [0.0, 1.0]
    with pytest.raises(ValueError, match="identical ordered"):
        paired_family_interval(rows, list(reversed(candidate)), seed="fixed", samples=100)


def test_label_families_cannot_be_inflated_or_overlapped():
    case = load_cases(DEFAULT_DATASET)[0]
    other = case.model_copy(deep=True)
    other.id, other.family_id, other.query = "other", "pretend-independent", "A distinct question with the same target"
    with pytest.raises(ValueError, match="same evidence target"):
        validate_cases([case, other])
    with pytest.raises(ValueError, match="overlap"):
        SpanCase(**{**case.model_dump(), "required_spans": case.required_spans * 2})
    with pytest.raises(ValueError, match="duplicate normalized queries"):
        validate_cases([case, case.model_copy(update={"id": "duplicate"})])


def test_review_provenance_and_fixed_primary_budget_are_required():
    case = load_cases(DEFAULT_DATASET)[0]
    with pytest.raises(ValueError, match="reviewer"):
        SpanCase(**{**case.model_dump(), "review_status": "human_reviewed"})
    with pytest.raises(ValueError, match="timestamp"):
        SpanCase(**{**case.model_dump(), "review_status": "human_reviewed", "reviewer": "owner"})
    with pytest.raises(ValueError, match="fixed primary"):
        SpanPlan(token_budgets=[256, 1024], primary_budget=512)


def test_complete_reviewed_same_strategy_can_pass_but_small_families_hold(index):
    cases = [SpanCase(**{**case.model_dump(), "review_status": "human_reviewed", "reviewer": "test-reviewer",
                        "reviewed_at": "2026-10-02T00:00:00Z"})
             for case in load_cases(DEFAULT_DATASET) if case.family_id != "report"]
    plan = SpanPlan(baseline="lexical", candidate="lexical", token_budgets=[4096], primary_budget=4096,
                    minimum_families=5, bootstrap_samples=100)
    result = evaluate(index, cases, plan)
    assert result["decision"] == "ready_for_owner_review"
    assert result["quality_gate_passed"] and not result["production_activation"]
    small = evaluate(index, cases[:2], plan)
    assert small["decision"] == "held"
    assert any("too few" in reason for reason in small["reasons"])


def test_cli_emits_cards_rejects_overwrites_and_keeps_strict_hold(tmp_path: Path):
    output = tmp_path / "report.json"
    assert main(["--output", str(output)]) == 0
    assert output.with_suffix(".md").exists()
    before = output.read_bytes()
    assert main(["--output", str(output)]) == 2
    assert output.read_bytes() == before
    assert main(["--output", str(tmp_path / "strict.json"), "--require-gate"]) == 1
    assert main(["--output", str(DEFAULT_DATASET)]) == 2
    assert main(["--output", str(DEFAULT_ROOT / "src/new_report.json")]) == 2
    workflow = (Path(__file__).resolve().parents[1] / ".github/workflows/ci.yml").read_text(encoding="utf-8")
    command = next(line for line in workflow.splitlines() if "python -m evals.retrieval_span_evaluation" in line)
    assert "--plan evals/experiments/retrieval_span_plan.json" in command
    assert "--require-gate" not in command
    assert "evidence-span-retrieval-comparison" in workflow


def test_span_digest_normalized_source_and_empty_report_integrity():
    assert span_digest(["a", "b"], 1, 2) == hashlib.sha256(b"a\nb").hexdigest()
    assert not verify_report({})


def test_overlapping_targets_do_not_create_new_families():
    case = load_cases(DEFAULT_DATASET)[0]
    other = case.model_copy(deep=True)
    other.id, other.family_id, other.query = "new", "new-family", "A question targeting almost the same source"
    other.required_spans[0].end_line += 1
    with pytest.raises(ValueError, match="overlapping spans"):
        validate_cases([case, other])


def test_paired_family_bootstrap_rejects_unbounded_sampling():
    with pytest.raises(ValueError, match="bounded"):
        paired_family_interval([], [], seed="fixed", samples=0)


def test_source_contradiction_and_receipt_shape_are_rejected(index):
    pack = index.select("release_lease owner expires_at", top_k=1, max_tokens=1024)
    damaged = pack.receipt.model_copy(deep=True)
    damaged.context_chars -= 1
    with pytest.raises(ValueError, match="length"):
        visible_lines(index, ContextPack(receipt=damaged, prompt_context=pack.prompt_context))
    prefix, section = pack.prompt_context.split("\n\n")
    header, _ = section.split("\n", 1)
    snippet = "    2: definitely_not_the_source"
    context = prefix + "\n\n" + header + "\n" + snippet
    damaged.context_chars = len(context)
    damaged.estimated_tokens = max(1, math.ceil(len(context) / 4))
    damaged.selected_files[0].snippet_sha256 = hashlib.sha256(snippet.encode()).hexdigest()
    damaged.fingerprint = receipt_fingerprint(damaged)
    with pytest.raises(ValueError, match="contradicts its source"):
        visible_lines(index, ContextPack(receipt=damaged, prompt_context=context))


def test_degraded_candidate_is_held_by_quality_and_family_interval(index, monkeypatch):
    cases = [SpanCase(**{**case.model_dump(), "review_status": "human_reviewed", "reviewer": "test-reviewer",
                        "reviewed_at": "2026-10-02T00:00:00Z"}) for case in load_cases(DEFAULT_DATASET)]
    original = index.select

    def degraded(query, **kwargs):
        pack = original(query, **kwargs)
        if kwargs["strategy"] == "hybrid":
            # Authored bad-context control: retain a valid receipt, emit no source sections.
            pack.prompt_context = pack.prompt_context.split("\n\n")[0]
            pack.receipt.selected_files = []
            pack.receipt.context_chars = len(pack.prompt_context)
            pack.receipt.estimated_tokens = math.ceil(len(pack.prompt_context) / 4)
            pack.receipt.fingerprint = receipt_fingerprint(pack.receipt)
        return pack

    monkeypatch.setattr(index, "select", degraded)
    result = evaluate(index, cases, SpanPlan(candidate="hybrid", bootstrap_samples=100))
    assert result["decision"] == "held" and not result["quality_gate_passed"]
    assert any("paired family interval" in reason for reason in result["reasons"])
    assert any("fixed floor" in reason for reason in result["reasons"])
    assert any("slice regressed" in reason for reason in result["reasons"])


def test_receipt_is_bound_to_query_and_strategy(index, monkeypatch):
    case = load_cases(DEFAULT_DATASET)[0]
    other = index.select("cache_entry_valid expires_at", top_k=1, max_tokens=1024)
    with pytest.raises(ValueError, match="different query"):
        score_pack(index, case, other)
    original = index.select

    def wrong_strategy(query, **kwargs):
        return original(query, **{**kwargs, "strategy": "lexical"})

    monkeypatch.setattr(index, "select", wrong_strategy)
    with pytest.raises(ValueError, match="different retrieval strategy"):
        evaluate(index, load_cases(DEFAULT_DATASET), SpanPlan(bootstrap_samples=100))


def test_real_oversized_pack_triggers_budget_hold(index, monkeypatch):
    original = index.select

    def oversized(query, **kwargs):
        return original(query, **{**kwargs, "max_tokens": 4096})

    monkeypatch.setattr(index, "select", oversized)
    result = evaluate(index, load_cases(DEFAULT_DATASET), SpanPlan(token_budgets=[64], primary_budget=64,
                                                                 bootstrap_samples=100))
    assert not result["quality_gate_passed"]
    assert any("exceeded" in reason for reason in result["reasons"])
