"""Exact numeric assertions over verified evidence; never semantic claim extraction."""

from collections.abc import Sequence
from decimal import Decimal
from typing import Literal

from pydantic import Field, field_validator, model_validator

from agentforge_sdk.models import Contract, fingerprint
from agentforge_sdk.reconciliation import EvidenceSet, FactCalculation, FactReconciliation, Unit


class NumericClaim(Contract):
    claim_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,79}$")
    label: str = Field(min_length=1, max_length=160, repr=False)
    # A separate unit avoids interpreting currency/scale embedded in prose.
    expected_value: str = Field(
        min_length=1, max_length=100, pattern=r"^[+-]?(?:0|[1-9][0-9]{0,59})(?:\.[0-9]{1,60})?$"
    )
    expected_unit: Unit

    @field_validator("label")
    @classmethod
    def literal_label(cls, value):
        if value != " ".join(value.split()) or any(ord(c) < 32 for c in value):
            raise ValueError("claim labels require canonical single-line text")
        try:
            value.encode("utf-8")
        except UnicodeError:
            raise ValueError("claim labels require valid UTF-8") from None
        return value


class NumericClaimCheck(Contract):
    protocol: Literal["agentforge-numeric-claim-check-v1"] = "agentforge-numeric-claim-check-v1"
    collection_fingerprint: str = Field(pattern=r"^[a-f0-9]{64}$")
    evidence_fingerprint: str = Field(pattern=r"^[a-f0-9]{64}$")
    claim: NumericClaim = Field(repr=False)
    status: Literal["supported", "refuted", "unresolved"]
    reason: Literal[
        "exact_value_match",
        "numeric_value_differs",
        "unit_mismatch",
        "evidence_unresolved",
        "evidence_unverifiable",
    ]
    observed_value: str | None = Field(default=None, max_length=100, repr=False)
    observed_unit: Unit | None = None
    fingerprint: str = Field(pattern=r"^[a-f0-9]{64}$")

    @model_validator(mode="after")
    def binding(self):
        if self.fingerprint != fingerprint(self.model_dump(mode="json", exclude={"fingerprint"})):
            raise ValueError("claim-check binding is invalid")
        return self


def check_numeric_claim(
    evidence: EvidenceSet,
    claim: NumericClaim,
    receipt: FactReconciliation | FactCalculation,
    *,
    facts: Sequence[FactReconciliation] | None = None,
) -> NumericClaimCheck:
    """Compare one caller-mapped assertion, withholding values from unusable evidence.

    The display label is not parsed or entailed. Exact Decimal equality accepts
    25 and 25.0, but never rounding tolerances, unit conversions or scope guesses.
    Supplied receipts must be verified against the current frozen evidence set.
    """
    claim = NumericClaim.model_validate(claim.model_dump())
    if not isinstance(receipt, (FactReconciliation, FactCalculation)):
        raise ValueError("claim audit requires a typed reconciliation receipt")
    payload = dict(
        protocol="agentforge-numeric-claim-check-v1",
        collection_fingerprint=evidence.fingerprint,
        evidence_fingerprint=fingerprint(receipt.model_dump(mode="json")),
        claim=claim.model_dump(mode="json"),
        status="unresolved",
        reason="evidence_unverifiable",
        observed_value=None,
        observed_unit=None,
    )
    valid = (
        evidence.verify_fact(receipt)
        if isinstance(receipt, FactReconciliation)
        else evidence.verify_calculation(receipt, facts=facts)
    )
    if valid:
        if receipt.status not in {"agreed", "calculated"}:
            payload["reason"] = "evidence_unresolved"
        else:
            payload.update(observed_value=receipt.value, observed_unit=receipt.unit)
            if receipt.unit != claim.expected_unit:
                # Incomparable is not numerical refutation: USD and EUR are not comparable values.
                payload["reason"] = "unit_mismatch"
            elif Decimal(receipt.value) == Decimal(claim.expected_value):
                payload.update(status="supported", reason="exact_value_match")
            else:
                payload.update(status="refuted", reason="numeric_value_differs")
    return NumericClaimCheck(**payload, fingerprint=fingerprint(payload))


def verify_numeric_claim(
    evidence: EvidenceSet,
    check: NumericClaimCheck,
    receipt: FactReconciliation | FactCalculation,
    *,
    facts: Sequence[FactReconciliation] | None = None,
) -> bool:
    """Recompute, including current source/review authority; plain hashes are not signatures."""
    try:
        check = NumericClaimCheck.model_validate(check.model_dump())
        return check == check_numeric_claim(evidence, check.claim, receipt, facts=facts)
    except (ValueError, TypeError, AttributeError, KeyError):
        return False
