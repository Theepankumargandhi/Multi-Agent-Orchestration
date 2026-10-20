"""Portable replay distinguishes consistency from numeric support and authenticity."""

import json

import pytest
from test_evidence_reconciliation import query, source

from agentforge_sdk import (
    ClaimBundleError,
    ClaimReplayEntry,
    ClaimReplayRecipe,
    EvidenceSet,
    NumericClaim,
    build_claim_replay_bundle,
    check_numeric_claim,
    load_claim_replay_bundle,
    verify_claim_replay_bundle,
)
from agentforge_sdk.bundle_cli import main
from agentforge_sdk.models import fingerprint


def example(*, conflict=False):
    evidence = EvidenceSet([source("a"), source("b", after="150" if conflict else "125")])
    before, after = [evidence.lookup(query(column=c)) for c in ("Before", "After")]
    calculation = evidence.calculate("percent_change", [before, after])
    entries = []
    for name, value, unit, receipt, queries, operation in [
        ("count", "125", "scalar", after, (after.query,), None),
        ("wrong", "999", "scalar", after, (after.query,), None),
        ("growth", "25", "%", calculation, (before.query, after.query), "percent_change"),
    ]:
        claim = NumericClaim(claim_id=name, label="PRIVATE_LABEL", expected_value=value, expected_unit=unit)
        entries.append(
            ClaimReplayEntry(
                recipe=ClaimReplayRecipe(claim=claim, queries=queries, operation=operation),
                check=check_numeric_claim(evidence, claim, receipt, facts=[before, after]),
            )
        )
    return evidence, entries


def bundle(**kwargs):
    return build_claim_replay_bundle(*example(**kwargs))


def reseal(data):
    data["fingerprint"] = fingerprint({k: v for k, v in data.items() if k != "fingerprint"})
    return json.dumps(data)


def test_roundtrip_replays_every_check_without_claiming_source_truth():
    original = bundle()
    parsed = load_claim_replay_bundle(original.model_dump_json())
    result = verify_claim_replay_bundle(parsed, expected_fingerprint=original.fingerprint)
    assert result.replay_verified and result.fingerprint_pinned
    assert (result.total, result.supported, result.refuted, result.unresolved) == (3, 2, 1, 0)
    assert not result.all_numeric_assertions_supported
    assert (
        not result.source_files_authenticated
        and not result.extraction_reexecuted
        and not result.release_authorized
    )
    assert result.model_calls == 0 and parsed == original
    assert build_claim_replay_bundle(*example()) == original


def test_conflicting_snapshot_verifies_as_unresolved_not_a_pass():
    result = verify_claim_replay_bundle(bundle(conflict=True))
    assert result.replay_verified and result.unresolved == 3
    assert not result.all_numeric_assertions_supported and not result.fingerprint_pinned


@pytest.mark.parametrize(
    "change",
    ["outer", "value", "rehashed_check", "recipe", "order", "policy", "source", "extra", "duplicate_claim"],
)
def test_tampering_or_resealed_recipe_changes_cannot_replay_original_checks(change):
    data = bundle().model_dump(mode="json")
    if change == "outer":
        data["fingerprint"] = "0" * 64
    elif change in {"value", "rehashed_check"}:
        check = data["entries"][0]["check"]
        check["observed_value"] = "999"
        if change == "rehashed_check":
            check["fingerprint"] = fingerprint({k: v for k, v in check.items() if k != "fingerprint"})
    elif change == "recipe":
        data["entries"][0]["recipe"]["queries"][0]["column_label"] = "Before"
    elif change == "order":
        data["entries"][2]["recipe"]["queries"].reverse()
    elif change == "policy":
        data["config"]["min_distinct_artifacts"] = 1
    elif change == "source":
        data["sources"].pop()
    elif change == "extra":
        data["execute"] = "PRIVATE_COMMAND"
    else:
        data["entries"].append(data["entries"][0])
    raw = json.dumps(data) if change == "outer" else reseal(data)
    with pytest.raises(ClaimBundleError):
        verify_claim_replay_bundle(load_claim_replay_bundle(raw))


@pytest.mark.parametrize(
    "raw",
    [
        b"",
        b"{",
        b"[]",
        b"\xff",
        "x" * 3_000_001,
        '{"protocol":1,"protocol":2}',
        '{"nested":{"x":1,"x":2}}',
        "[" * 1500 + "]" * 1500,
    ],
    ids=[
        "empty",
        "truncated",
        "wrong_root",
        "invalid_utf8",
        "over_bytes",
        "duplicate_root",
        "duplicate_nested",
        "deep_nesting",
    ],
)
def test_untrusted_json_is_bounded_and_rejects_duplicate_keys_without_echo(raw):
    with pytest.raises(ClaimBundleError) as exc:
        load_claim_replay_bundle(raw)
    assert str(exc.value) == "claim replay bundle is invalid or exceeds its budget"


def test_separate_pin_rejects_an_entire_consistent_replacement_snapshot():
    original, replacement = bundle(), bundle(conflict=True)
    assert verify_claim_replay_bundle(replacement).replay_verified
    with pytest.raises(ClaimBundleError):
        verify_claim_replay_bundle(replacement, expected_fingerprint=original.fingerprint)


@pytest.mark.parametrize(
    "operation,columns",
    [(None, ("Before", "After")), ("difference", ("After",)), ("sum", ("After", "After"))],
)
def test_recipe_arity_and_duplicate_operands_are_rejected(operation, columns):
    claim = example()[1][0].recipe.claim
    with pytest.raises(ValueError):
        ClaimReplayRecipe(claim=claim, queries=tuple(query(column=c) for c in columns), operation=operation)


def test_cli_summary_and_exit_codes_are_distinct_and_do_not_leak_claims(tmp_path, capsys):
    data = bundle()
    path = tmp_path / "bundle.json"
    path.write_text(data.model_dump_json(), encoding="utf-8")
    assert main([str(path), "--expected-fingerprint", data.fingerprint]) == 0
    output = capsys.readouterr().out
    assert json.loads(output)["refuted"] == 1 and "PRIVATE_LABEL" not in output
    assert main([str(path), "--require-supported"]) == 2
    capsys.readouterr()
    assert main([str(path), "--expected-fingerprint", "0" * 64]) == 1
    assert json.loads(capsys.readouterr().out)["replay_verified"] is False
    assert main([str(tmp_path)]) == 1
    assert "PRIVATE_LABEL" not in capsys.readouterr().out


def test_builder_refuses_live_review_authority_without_laundering_it(tmp_path):
    from agentforge_sdk import EvidenceSource, ReviewAuthority, ReviewedRAG, apply_review, prepare_review
    from evals.evidence_review_fixtures import create_review_fixture

    path, original, correction = create_review_fixture(tmp_path)
    authority = ReviewAuthority({"reviewer": b"a" * 32}, clock=lambda: 1000)
    draft = prepare_review(original, [correction], rationale="Authored snapshot authority control")
    approval = authority.decide(draft, reviewer_id="reviewer", decision="approve")
    reviewed = ReviewedRAG(
        apply_review(original, draft, approval, authority=authority), authority=authority, source_path=path
    )
    evidence, entries = example()
    mixed = EvidenceSet([source("a"), EvidenceSource("reviewed", query().scope, reviewed)])
    with pytest.raises(ClaimBundleError):
        build_claim_replay_bundle(mixed, entries)
    assert reviewed.is_authorized() and evidence.verify_fact(evidence.lookup(query()))
