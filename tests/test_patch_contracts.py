from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from code_agent.patch_contracts import (
    ContractReview,
    DomainSuggestions,
    accept_reviewed_domains,
    propose_domains,
)
from code_agent.patch_inputs import PatchVerifySpec
from code_agent.patch_verify import verify_patch

FIXTURE = Path(__file__).parent / "fixtures/patch_structured"


def inputs():
    spec = PatchVerifySpec.model_validate_json((FIXTURE / "spec.json").read_bytes())
    return (
        spec,
        (FIXTURE / "regression.diff").read_text(),
        {"app.py": (FIXTURE / "app.py").read_bytes().decode()},
    )


class Model:
    calls = 0

    def __init__(self, result):
        self.result = result

    def with_structured_output(self, schema):
        assert schema is DomainSuggestions
        return self

    async def ainvoke(self, messages):
        self.calls += 1
        assert "UNTRUSTED DATA" in messages[0].content
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


def suggestion(spec):
    return {
        "json_domains": [row.model_dump() for row in spec.json_domains],
        "review_notes": "Owner must confirm total() preservation; exclude customer_label's intended change.",
    }


@pytest.mark.asyncio
async def test_suggestion_is_opt_in_and_cannot_self_authorize():
    spec, patch, sources = inputs()
    model = Model(suggestion(spec))
    with pytest.raises(ValueError, match="authorization"):
        await propose_domains(model, patch=patch, sources=sources, model_label="scripted-control")
    assert model.calls == 0
    proposal = await propose_domains(
        model, patch=patch, sources=sources, model_label="scripted-control", authorize_model=True
    )
    assert model.calls == proposal.model_invocations == 1
    assert proposal.status == "needs_owner_review" and proposal.provider_cost_usd is None
    review = ContractReview(
        proposal_sha256=proposal.fingerprint,
        reviewer="owner",
        decision="accepted",
        preservation_confirmed=True,
        rationale="Preserve all declared valid cart totals.",
    )
    base = spec.model_copy(update={"json_domains": []})
    accepted = accept_reviewed_domains(base, proposal, review, patch=patch, sources=sources)
    assert accepted.contracts == base.contracts and accepted.json_domains == spec.json_domains
    assert accepted.contract_review.patch_sha256 == proposal.patch_sha256
    with pytest.raises(ValueError, match="separately authorize"):
        accept_reviewed_domains(
            base.model_copy(update={"preservation_authorized": False}),
            proposal,
            review,
            patch=patch,
            sources=sources,
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("fault", ["extra_authority", "outside_entrypoint", "raw_error"])
async def test_malformed_or_unsafe_model_output_is_held_without_raw_error_leak(fault):
    spec, patch, sources = inputs()
    result = suggestion(spec)
    if fault == "extra_authority":
        result["preservation_authorized"] = True
    elif fault == "outside_entrypoint":
        result["json_domains"][0]["entrypoint"] = "os:system"
    else:
        result = RuntimeError("PRIVATE-PROVIDER-RESPONSE")
    model = Model(result)
    with pytest.raises(ValueError, match="usage may be unknown") as error:
        await propose_domains(
            model, patch=patch, sources=sources, model_label="control", authorize_model=True
        )
    assert model.calls == 1 and "PRIVATE-PROVIDER" not in str(error.value)


@pytest.mark.asyncio
async def test_rejected_stale_and_tampered_proposals_cannot_be_adopted():
    spec, patch, sources = inputs()
    proposal = await propose_domains(
        Model(suggestion(spec)), patch=patch, sources=sources, model_label="control", authorize_model=True
    )
    review = ContractReview(
        proposal_sha256=proposal.fingerprint,
        reviewer="owner",
        decision="accepted",
        preservation_confirmed=True,
        rationale="Reviewed actual input preservation scope.",
    )
    base = spec.model_copy(update={"json_domains": []})
    for selected, decision, raw_patch, context in [
        (proposal, review.model_copy(update={"decision": "rejected"}), patch, sources),
        (proposal, review, patch + "\n", sources),
        (proposal, review, patch, {"app.py": sources["app.py"] + "\n# changed"}),
        (proposal.model_copy(update={"fingerprint": "0" * 64}), review, patch, sources),
    ]:
        with pytest.raises(ValueError):
            accept_reviewed_domains(base, selected, decision, patch=raw_patch, sources=context)


@pytest.mark.asyncio
async def test_execution_rechecks_reviewed_source_before_starting_any_container():
    spec, patch, sources = inputs()
    proposal = await propose_domains(
        Model(suggestion(spec)), patch=patch, sources=sources, model_label="control", authorize_model=True
    )
    review = ContractReview(
        proposal_sha256=proposal.fingerprint,
        reviewer="owner",
        decision="accepted",
        preservation_confirmed=True,
        rationale="Reviewed actual input preservation scope.",
    )
    approved = accept_reviewed_domains(
        spec.model_copy(update={"json_domains": []}), proposal, review, patch=patch, sources=sources
    )
    data = approved.model_dump()
    data["contract_review"]["source_files"]["app.py"] = "0" * 64

    def forbidden(*args):
        raise AssertionError("Docker must not start on stale approval")

    report = await verify_patch(
        FIXTURE,
        patch,
        PatchVerifySpec.model_validate(data),
        sandbox_factory=forbidden,
        image_resolver=forbidden,
    )
    assert report["status"] == "inconclusive" and not report["trials"]
    assert report["source_unchanged"]


@pytest.mark.asyncio
async def test_timeout_has_no_automatic_retry_or_free_usage_claim():
    spec, patch, sources = inputs()

    class Slow(Model):
        async def ainvoke(self, messages):
            self.calls += 1
            await asyncio.sleep(20)

    model = Slow(suggestion(spec))
    with pytest.raises(ValueError, match="usage may be unknown"):
        await propose_domains(
            model,
            patch=patch,
            sources=sources,
            model_label="control",
            authorize_model=True,
            timeout_seconds=1,
        )
    assert model.calls == 1


def test_acceptance_requires_separate_explicit_preservation_attestation():
    with pytest.raises(ValueError):
        ContractReview(
            proposal_sha256="0" * 64,
            reviewer="owner",
            decision="accepted",
            rationale="Do not silently approve anything.",
        )
