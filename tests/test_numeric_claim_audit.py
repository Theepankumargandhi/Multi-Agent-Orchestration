"""Exact numeric assertions are not natural-language entailment or source truth."""

from decimal import localcontext

import pytest
from test_evidence_reconciliation import query, source

from agentforge_sdk import (
    EvidenceSet,
    NumericClaim,
    NumericClaimCheck,
    check_numeric_claim,
    verify_numeric_claim,
)
from agentforge_sdk.models import fingerprint


def claim(value="125", unit="scalar", label="Requests are 125"):
    return NumericClaim(claim_id="request-count", label=label, expected_value=value, expected_unit=unit)


@pytest.mark.parametrize(
    "value,status",
    [
        ("125", "supported"),
        ("125.000", "supported"),
        ("+125", "supported"),
        ("126", "refuted"),
        ("124.999999999999", "refuted"),
    ],
)
def test_exact_decimal_claims_over_agreed_facts(value, status):
    evidence = EvidenceSet([source("a"), source("b")])
    fact = evidence.lookup(query())
    check = check_numeric_claim(evidence, claim(value), fact)
    assert check.status == status and check.observed_value == "125"
    assert verify_numeric_claim(
        evidence, NumericClaimCheck.model_validate_json(check.model_dump_json()), fact
    )
    with localcontext() as ctx:
        ctx.prec = 2
        assert check_numeric_claim(evidence, claim(value), fact) == check


@pytest.mark.parametrize(
    "value", ["1,000", "1e3", "NaN", "Infinity", " 125", "125%", "$125", "01", "", 125, 125.0]
)
def test_expected_values_are_explicit_bounded_decimal_strings(value):
    with pytest.raises(ValueError):
        claim(value)


def test_unit_mismatch_is_unresolved_not_refutation_or_conversion():
    evidence = EvidenceSet([source("a"), source("b")])
    fact = evidence.lookup(query(row="Revenue"))
    check = check_numeric_claim(evidence, claim("150.75", "EUR"), fact)
    assert (check.status, check.reason) == ("unresolved", "unit_mismatch")
    assert check.observed_unit == "USD"


@pytest.mark.parametrize("options", [{"after": "150"}, {"held": True}, {"after": "1,000"}])
def test_conflict_or_unusable_evidence_never_supports_a_matching_observation(options):
    evidence = EvidenceSet([source("a"), source("b", **options)])
    check = check_numeric_claim(evidence, claim(), evidence.lookup(query()))
    assert (check.status, check.reason) == ("unresolved", "evidence_unresolved")
    assert check.observed_value is None and check.observed_unit is None


def test_derived_calculations_labels_and_tamper_verification():
    evidence = EvidenceSet([source("a"), source("b")])
    facts = [evidence.lookup(query(column=c)) for c in ("Before", "After")]
    calc = evidence.calculate("percent_change", facts)
    spec = claim("25", "%", "This label is not semantically checked")
    check = check_numeric_claim(evidence, spec, calc, facts=facts)
    assert check.status == "supported" and verify_numeric_claim(evidence, check, calc, facts=facts)
    changed = check.model_dump(mode="json")
    changed["status"] = "refuted"
    changed["fingerprint"] = fingerprint({k: v for k, v in changed.items() if k != "fingerprint"})
    assert not verify_numeric_claim(evidence, NumericClaimCheck(**changed), calc, facts=facts)
    altered = calc.model_copy(update={"value": "999"})
    blocked = check_numeric_claim(evidence, spec, altered, facts=facts)
    assert blocked.reason == "evidence_unverifiable" and blocked.observed_value is None
    other = EvidenceSet([source("a"), source("changed", after="999")])
    assert not verify_numeric_claim(other, check, calc, facts=facts)


def test_signed_zero_and_fractional_percent_are_not_scale_inference():
    evidence = EvidenceSet([source("a", after="-0"), source("b", after="+0")])
    assert check_numeric_claim(evidence, claim("0.000"), evidence.lookup(query())).status == "supported"
    evidence = EvidenceSet([source("a"), source("b")])
    calc = evidence.calculate(
        "percent_change", [evidence.lookup(query(column=c)) for c in ("Before", "After")]
    )
    assert check_numeric_claim(evidence, claim("0.25", "%"), calc).status == "refuted"


def test_plain_rehash_is_not_proof_of_source_or_authorization():
    evidence = EvidenceSet([source("a"), source("b")])
    fact = evidence.lookup(query())
    changed = fact.model_copy(update={"value": "999"})
    check = check_numeric_claim(evidence, claim("999"), changed)
    assert check.status == "unresolved" and check.reason == "evidence_unverifiable"
    assert "Requests are" not in repr(claim())
    with pytest.raises(ValueError):
        claim(label="PRIVATE\nTEXT")
