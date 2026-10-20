"""Ablation views keep the evidence policy and never repair the original set."""

import pytest
from test_evidence_reconciliation import SCOPE, query, source

from agentforge_sdk import EvidenceSet, EvidenceSource, ReconciliationConfig


def test_omission_view_preserves_minimum_artifact_policy_and_baseline():
    collection = EvidenceSet([source("a"), source("b")])
    baseline = collection.lookup(query())
    assert collection.source_ids == ("a", "b")
    view = collection.without_sources(["a"])
    check = view.lookup(query())
    assert view.source_ids == ("b",) and check.status == "abstained"
    assert "insufficient_distinct_artifacts" in check.reasons
    assert check.value is None and collection.lookup(query()) == baseline
    assert not collection.verify_fact(check) and not view.verify_fact(baseline)


def test_three_agreeing_sources_remain_supported_but_not_proven_true():
    collection = EvidenceSet([source("a"), source("b"), source("c")])
    for omitted in collection.source_ids:
        view = collection.without_sources([omitted])
        fact = view.lookup(query())
        assert fact.status == "agreed" and fact.value == "125" and view.verify_fact(fact)
        assert omitted not in {o.source_id for o in fact.observations}
    assert collection.source_ids == ("a", "b", "c")


def test_excluding_a_dissenting_report_masks_conflict_only_in_the_counterfactual():
    collection = EvidenceSet([source("a"), source("b"), source("c", after="150")])
    baseline = collection.lookup(query())
    assert baseline.status == "conflict" and baseline.value is None
    masked = collection.without_sources(["c"]).lookup(query())
    assert masked.status == "agreed" and masked.value == "125"
    assert collection.lookup(query()) == baseline and not collection.verify_fact(masked)


def test_duplicate_byte_aliases_do_not_turn_omission_counts_into_independent_sources():
    reader = source("a").reader
    collection = EvidenceSet(
        [EvidenceSource("a", SCOPE, reader), EvidenceSource("alias", SCOPE, reader), source("b")]
    )
    assert collection.without_sources(["alias"]).lookup(query()).status == "agreed"
    remaining = collection.without_sources(["b"]).lookup(query())
    assert remaining.distinct_artifacts == 1 and remaining.status == "abstained"


@pytest.mark.parametrize(
    "excluded", [[], "a", ["unknown"], ["a", "a"], ["a", "b"], [1], [["a"]], ["a"] * 17, {"a"}]
)
def test_invalid_or_all_source_omissions_are_rejected(excluded):
    with pytest.raises(ValueError):
        EvidenceSet([source("a"), source("b")]).without_sources(excluded)


def test_omission_order_does_not_change_the_retained_context():
    collection = EvidenceSet([source(name) for name in ("d", "b", "a", "c")])
    assert (
        collection.without_sources(["b", "d"]).fingerprint
        == collection.without_sources(["d", "b"]).fingerprint
    )
    assert collection.without_sources(["a"]).fingerprint != collection.without_sources(["b"]).fingerprint


@pytest.mark.parametrize("minimum", [1, 3])
@pytest.mark.parametrize("coverage", [True, False])
def test_custom_policy_and_extraction_bounds_survive_omission(minimum, coverage):
    policy = ReconciliationConfig(
        min_distinct_artifacts=minimum,
        require_complete_coverage=coverage,
        max_total_regions=200,
        max_total_text_chars=1000,
    )
    sources = [source(name) for name in ("a", "b", "c", "d")]
    original = EvidenceSet(sources, config=policy)
    view = original.without_sources(("d",))
    expected = EvidenceSet(sources[:3], config=policy)
    assert view.fingerprint == expected.fingerprint and view.lookup(query()) == expected.lookup(query())
    assert original.source_ids == ("a", "b", "c", "d")


def test_views_keep_retained_readers_live_review_authority(tmp_path):
    pytest.importorskip("pdfplumber")
    from agentforge_sdk import ReviewAuthority, ReviewedRAG, apply_review, prepare_review
    from evals.evidence_review_fixtures import create_review_fixture

    path, original, correction = create_review_fixture(tmp_path)
    authority = ReviewAuthority({"reviewer": b"a" * 32}, clock=lambda: 1000)
    draft = prepare_review(original, [correction], rationale="Authored source omission authority control")
    approval = authority.decide(draft, reviewer_id="reviewer", decision="approve")
    reviewed = ReviewedRAG(
        apply_review(original, draft, approval, authority=authority), authority=authority, source_path=path
    )
    collection = EvidenceSet([source("omit"), source("native"), EvidenceSource("reviewed", SCOPE, reviewed)])
    view = collection.without_sources(["omit"])
    fact = view.lookup(query())
    assert fact.status == "agreed"
    authority.revoke(approval)
    assert not view.verify_fact(fact)
    blocked = view.lookup(query())
    assert blocked.status == "abstained" and any(o.status == "unauthorized" for o in blocked.observations)
