"""Exact structured-evidence controls; agreement is not source truth or semantic inference."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from decimal import localcontext

import pytest
from test_document_changes import document

from agentforge_sdk import (
    EvidenceSet,
    EvidenceSource,
    FactCalculation,
    FactQuery,
    FactReconciliation,
    MultimodalDocument,
    MultimodalRAG,
    ReconciliationConfig,
    Region,
)
from agentforge_sdk.models import fingerprint
from agentforge_sdk.spatial import region_fingerprint

SCOPE = "atlas:requests:2026-Q1:units"


def source(name, *, scope=SCOPE, **options):
    return EvidenceSource(name, scope, MultimodalRAG(document(version=name, **options)))


def query(row="Requests", column="After", scope=SCOPE):
    return FactQuery(row_label=row, column_label=column, scope=scope)


def test_agreement_cites_headers_and_value_from_each_source():
    collection = EvidenceSet([source("finance"), source("ops", after="125.0", cover=True, reverse=True)])
    fact = collection.lookup(query())
    assert fact.status == "agreed" and fact.value == "125" and fact.unit == "scalar"
    assert fact.distinct_artifacts == 2 and collection.verify_fact(fact)
    assert [o.source_id for o in fact.observations] == ["finance", "ops"]
    assert [[c.quote for c in o.citations] for o in fact.observations] == [["Requests", "After", "125"], ["Requests", "After", "125.0"]]
    assert all(c.page == 2 for c in fact.observations[1].citations)
    assert FactReconciliation.model_validate_json(fact.model_dump_json()) == fact
    assert "125" not in repr(fact) and "Requests" not in repr(query())


def test_agreement_is_order_invariant_not_an_order_based_winner():
    sources = [source("z"), source("a")]
    first, second = EvidenceSet(sources), EvidenceSet(list(reversed(sources)))
    assert first.fingerprint == second.fingerprint and first.lookup(query()) == second.lookup(query())


def test_majority_does_not_hide_one_conflicting_report():
    collection = EvidenceSet([source("a"), source("b"), source("c"), source("d", after="150")])
    fact = collection.lookup(query())
    assert fact.status == "conflict" and fact.value is None and fact.unit is None
    assert "numeric_values_disagree" in fact.reasons
    assert [(g.value, g.source_ids) for g in fact.groups] == [("125", ("a", "b", "c")), ("150", ("d",))]
    calc = collection.calculate("sum", [fact])
    assert calc.status == "abstained" and calc.reason == "unresolved_fact"
    assert collection.verify_calculation(calc)


def test_one_file_under_multiple_aliases_cannot_satisfy_artifact_count():
    reader = source("a").reader
    collection = EvidenceSet([EvidenceSource("a", SCOPE, reader), EvidenceSource("b", SCOPE, reader)])
    fact = collection.lookup(query())
    assert fact.status == "abstained" and fact.distinct_artifacts == 1
    assert "insufficient_distinct_artifacts" in fact.reasons
    assert EvidenceSet([source("single")], config=ReconciliationConfig(min_distinct_artifacts=1)).lookup(query()).status == "agreed"


def test_scopes_do_not_infer_reporting_period_or_entity():
    collection = EvidenceSet([source("a"), source("b"), source("other-quarter", after="900", scope="atlas:requests:2026-Q2:units")])
    fact = collection.lookup(query())
    assert fact.status == "agreed" and fact.observations[-1].status == "out_of_scope"
    assert not fact.observations[-1].citations
    unknown = collection.lookup(query(scope="unknown"))
    assert unknown.status == "abstained" and "scope_not_found" in unknown.reasons


@pytest.mark.parametrize("value", ["1,000", "1e3", "125 requests", "2026-01-01", "USD 125%", "NaN", "1.2345678901234"])
def test_unsupported_numeric_syntax_is_not_guessed(value):
    collection = EvidenceSet([source("a"), source("b", after=value)])
    fact = collection.lookup(query())
    assert fact.status == "abstained" and "unsupported_number" in fact.reasons
    assert fact.observations[1].status == "unsupported_number" and len(fact.observations[1].citations) == 3


def test_unit_disagreement_is_incomparable_not_currency_conversion():
    collection = EvidenceSet([source("a"), source("b", currency="EUR")])
    fact = collection.lookup(query(row="Revenue"))
    assert fact.status == "abstained" and "incompatible_units" in fact.reasons and fact.value is None


@pytest.mark.parametrize("left,right,value,unit", [("USD 125.00", "$125", "125", "USD"),
                                                   ("-0", "+0", "0", "scalar"),
                                                   ("125.0", "+125", "125", "scalar"),
                                                   ("125%", "125.0%", "125", "%")])
def test_supported_equivalent_number_forms_compare_exactly(left, right, value, unit):
    fact = EvidenceSet([source("a", after=left), source("b", after=right)]).lookup(query())
    assert fact.status == "agreed" and (fact.value, fact.unit) == (value, unit)


@pytest.mark.parametrize("duplicate,held", [(True, False), (False, True)])
def test_ambiguity_or_held_extraction_cannot_be_excluded_by_relaxed_coverage(duplicate, held):
    collection = EvidenceSet([source("a"), source("b"), source("bad", duplicate=duplicate, held=held)],
                             config=ReconciliationConfig(require_complete_coverage=False))
    fact = collection.lookup(query())
    assert fact.status == "abstained" and fact.observations[-1].status == ("ambiguous" if duplicate else "held")


def test_missing_fact_requires_explicit_partial_coverage_policy():
    sources = [source("a"), source("b"), source("missing", duplicate=True)]
    strict = EvidenceSet(sources).lookup(query(row="Revenue"))
    relaxed = EvidenceSet(sources, config=ReconciliationConfig(require_complete_coverage=False)).lookup(query(row="Revenue"))
    assert strict.status == "abstained" and "incomplete_coverage" in strict.reasons
    assert relaxed.status == "agreed" and relaxed.observations[-1].status == "missing"


def mutate(reader, change):
    payload = reader.document.model_dump()
    change(payload)
    for page in payload["pages"]:
        for r in page["regions"]:
            r["region_id"] = region_fingerprint(payload["file_sha256"], page["page"], Region.model_validate(r))
    payload["fingerprint"] = fingerprint({k: v for k, v in payload.items() if k != "fingerprint"})
    return MultimodalRAG(MultimodalDocument.model_validate(payload))


@pytest.mark.parametrize("mode", ["header", "table", "sparse"])
def test_duplicate_headers_multiple_matching_tables_and_sparse_grids_are_ambiguous(mode):
    reader = source("b").reader

    def change(payload):
        cells = payload["pages"][0]["regions"]
        if mode == "header":
            next(r for r in cells if r["kind"] == "table_cell" and r["row"] == 0 and r["column"] == 1)["text"] = "After"
        elif mode == "table":
            copies = [{**r, "table_id": "second"} for r in cells if r["kind"] == "table_cell"]
            payload["pages"][0]["regions"] = (*cells, *copies)
        else:
            payload["pages"][0]["regions"] = tuple(r for r in cells if not (r["row"] == 0 and r["column"] == 0))

    collection = EvidenceSet([source("a"), EvidenceSource("b", SCOPE, mutate(reader, change))])
    fact = collection.lookup(query())
    assert fact.observations[1].status == "ambiguous" and fact.status == "abstained"


@pytest.mark.parametrize("operation,value,unit", [("sum", "225", "scalar"), ("difference", "-25", "scalar"),
                                               ("ratio", "0.8", "scalar"), ("percent_change", "25", "%")])
def test_reconciled_calculations_share_exact_decimal_rules(operation, value, unit):
    collection = EvidenceSet([source("a"), source("b")])
    facts = [collection.lookup(query(column=c)) for c in ("Before", "After")]
    calc = collection.calculate(operation, facts)
    assert calc.status == "calculated" and (calc.value, calc.unit) == (value, unit)
    assert collection.verify_calculation(FactCalculation.model_validate_json(calc.model_dump_json()))
    with localcontext() as ctx:
        ctx.prec = 2
        assert collection.calculate(operation, facts) == calc


def test_zero_denominator_and_mixed_units_abstain_with_verifiable_facts():
    collection = EvidenceSet([source("a", after="0"), source("b", after="0")])
    before, after = [collection.lookup(query(column=c)) for c in ("Before", "After")]
    zero = collection.calculate("ratio", [before, after])
    assert zero.reason == "zero_denominator" and collection.verify_calculation(zero)
    revenue = collection.lookup(query(row="Revenue"))
    mixed = collection.calculate("sum", [before, revenue])
    assert mixed.reason == "incompatible_units" and collection.verify_calculation(mixed)


def test_foreign_or_tampered_facts_are_not_used_or_echoed_in_calculations():
    collection = EvidenceSet([source("a"), source("b")])
    fact = collection.lookup(query())
    bad = fact.model_copy(update={"value": "999"})
    bad = bad.model_copy(update={"fingerprint": fingerprint(bad.model_dump(mode="json", exclude={"fingerprint"}))})
    assert not collection.verify_fact(bad)
    calc = collection.calculate("sum", [bad])
    assert calc.reason == "unverifiable_fact" and not calc.facts and calc.value is None
    assert not collection.verify_calculation(calc) and collection.verify_calculation(calc, facts=[bad])
    other = EvidenceSet([source("c"), source("d")])
    assert other.calculate("sum", [fact]).reason == "unverifiable_fact"


def test_calculation_or_coverage_tampering_is_rejected_even_if_rehashed():
    collection = EvidenceSet([source("a"), source("b")])
    fact = collection.lookup(query())
    calc = collection.calculate("sum", [fact])
    for result, verifier, updates in ((fact, collection.verify_fact, {"distinct_artifacts": 9}),
                                      (fact, collection.verify_fact, {"observations": fact.observations[:1]}),
                                      (calc, collection.verify_calculation, {"value": "999"})):
        fake = result.model_copy(update=updates)
        fake = fake.model_copy(update={"fingerprint": fingerprint(fake.model_dump(mode="json", exclude={"fingerprint"}))})
        assert not verifier(fake)


def test_lookup_never_uses_answer_selectors_or_embedders(monkeypatch):
    sources = [source("a"), source("b")]
    for s in sources:
        monkeypatch.setattr(s.reader, "answer", lambda *a, **k: pytest.fail("text backend called"))
        monkeypatch.setattr(s.reader, "calculate", lambda *a, **k: pytest.fail("cell tool was not needed for literal citations"))
    collection = EvidenceSet(sources)
    fact = collection.lookup(query())
    assert fact.status == "agreed" and collection.verify_fact(fact)


@pytest.mark.parametrize("values", [{"min_distinct_artifacts": True}, {"max_total_regions": 0},
                                   {"require_complete_coverage": "yes"}, {"max_total_text_chars": float("inf")}])
def test_policy_is_strict_and_bounded(values):
    with pytest.raises(ValueError):
        ReconciliationConfig(**values)


@pytest.mark.parametrize("label", ["", " Requests", "Requests\n", "Private\x00", "\ud800"])
def test_query_labels_require_canonical_valid_utf8(label):
    with pytest.raises(ValueError):
        query(row=label)


def test_invalid_collections_and_calculation_arity_are_rejected():
    s = source("a")
    for sources in ([], [s, s], [s] * 17, [s.reader]):
        with pytest.raises(ValueError):
            EvidenceSet(sources)
    with pytest.raises(ValueError):
        EvidenceSet([s], config=ReconciliationConfig(max_total_regions=1))
    with pytest.raises(ValueError):
        EvidenceSet([source("long", title="A" * 1200)], config=ReconciliationConfig(max_total_text_chars=1000))
    with pytest.raises(ValueError):
        EvidenceSource("../unsafe", SCOPE, s.reader)
    with pytest.raises(FrozenInstanceError):
        s.scope = "changed"
    collection = EvidenceSet([s], config=ReconciliationConfig(min_distinct_artifacts=1))
    fact = collection.lookup(query())
    for operation, facts in (("sum", []), ("ratio", [fact]), ("sum", [fact, fact]), ([], [fact])):
        with pytest.raises(ValueError):
            collection.calculate(operation, facts)


@pytest.mark.parametrize("invalidation", ["revoke", "expire", "source"])
def test_review_lineage_is_kept_and_current_authority_is_required(tmp_path, invalidation):
    pytest.importorskip("pdfplumber")
    from agentforge_sdk import ReviewAuthority, ReviewedRAG, ReviewError, apply_review, prepare_review
    from evals.evidence_review_fixtures import create_review_fixture

    path, original, correction = create_review_fixture(tmp_path)
    clock = [1000]
    authority = ReviewAuthority({"reviewer": b"a" * 32}, clock=lambda: clock[0])
    draft = prepare_review(original, [correction], rationale="Authored numeric reconciliation control")
    approval = authority.decide(draft, reviewer_id="reviewer", decision="approve")
    reader = ReviewedRAG(apply_review(original, draft, approval, authority=authority), authority=authority, source_path=path)
    collection = EvidenceSet([source("native"), EvidenceSource("reviewed", SCOPE, reader)])
    fact = collection.lookup(query())
    assert fact.status == "agreed" and collection.verify_fact(fact)
    citations = fact.observations[1].citations
    assert all(c.origin == "reviewer_supplied" and c.approval_fingerprint == approval.fingerprint for c in citations)
    assert all(reader.verify_citation(c) for c in citations)
    assert reader.citation(citations[-1].spatial.region_id) == citations[-1]
    calc = collection.calculate("sum", [fact])
    if invalidation == "revoke":
        authority.revoke(approval)
    elif invalidation == "expire":
        clock[0] += 3601
    else:
        path.write_bytes(b"changed")
    assert not collection.verify_fact(fact) and not collection.verify_calculation(calc)
    blocked = collection.lookup(query())
    assert blocked.status == "abstained" and blocked.observations[1].status == "unauthorized"
    assert blocked.observations[1].value is None and not blocked.observations[1].citations
    with pytest.raises(ReviewError, match="review"):
        reader.citation(citations[-1].spatial.region_id)


def test_later_source_revocation_discards_earlier_reviewed_observation(tmp_path, monkeypatch):
    pytest.importorskip("pdfplumber")
    from agentforge_sdk import ReviewAuthority, ReviewedRAG, apply_review, prepare_review
    from evals.evidence_review_fixtures import create_review_fixture

    path, original, correction = create_review_fixture(tmp_path)
    authority = ReviewAuthority({"reviewer": b"a" * 32}, clock=lambda: 1000)
    draft = prepare_review(original, [correction], rationale="Authored mid-lookup revocation control")
    approval = authority.decide(draft, reviewer_id="reviewer", decision="approve")
    reviewed = ReviewedRAG(apply_review(original, draft, approval, authority=authority), authority=authority, source_path=path)
    native = source("m-native")
    actual = native.reader.citation

    def revoke(*args, **kwargs):
        authority.revoke(approval)
        return actual(*args, **kwargs)

    collection = EvidenceSet([EvidenceSource("a-reviewed", SCOPE, reviewed), native, source("z-native")])
    monkeypatch.setattr(native.reader, "citation", revoke)
    fact = collection.lookup(query())
    assert fact.status == "abstained" and "unauthorized_source" in fact.reasons
    assert fact.distinct_artifacts == 2  # Two good artifacts cannot bypass the revoked one.
    assert fact.observations[0].status == "unauthorized"
    assert fact.observations[0].value is None and not fact.observations[0].citations
