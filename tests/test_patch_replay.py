import hashlib
from pathlib import Path

import pytest

from code_agent.patch_inputs import PatchVerifySpec
from code_agent.patch_verify import reproducer_spec, tree_fingerprint, verify_patch
from code_agent.upgrade_models import fingerprint

FIXTURE = Path(__file__).parent / "fixtures/patch_structured"


def evidence():
    spec = PatchVerifySpec.model_validate_json((FIXTURE / "spec.json").read_bytes())
    report = {
        "status": "regression_found",
        "source_unchanged": True,
        "source_sha256": tree_fingerprint(FIXTURE),
        "patch_sha256": hashlib.sha256((FIXTURE / "regression.diff").read_text().encode()).hexdigest(),
        "spec_sha256": fingerprint(spec.model_dump()),
        "image_id": "sha256:" + "a" * 64,
        "minimizations": [
            {
                "confirmed_case": spec.json_domains[0]
                .case({"items": [{"quantity": 0, "price": 1}]}, "generated.cart.0")
                .model_dump()
            }
        ],
    }
    report["fingerprint"] = fingerprint(report)
    return spec, report


def test_replay_pins_inputs_and_preserves_owner_scope():
    spec, report = evidence()
    replay = reproducer_spec(spec, report)
    assert replay.reference_source_sha256 == report["source_sha256"]
    assert replay.reference_patch_sha256 == report["patch_sha256"]
    assert replay.image == report["image_id"]
    assert replay.source_paths == spec.source_paths and replay.test_paths == spec.test_paths
    assert replay.preservation_authorized and not replay.reduction.enabled
    assert not replay.json_domains and not replay.integer_domains and len(replay.contracts) == 1
    assert replay.contracts[0].case_id.startswith("replay.")


@pytest.mark.parametrize("change", ["status", "source_unchanged", "minimizations"])
def test_unconfirmed_evidence_is_not_exported(change):
    spec, report = evidence()
    report[change] = {"status": "checks_passed", "source_unchanged": False, "minimizations": []}[change]
    report["fingerprint"] = fingerprint({key: value for key, value in report.items() if key != "fingerprint"})
    assert reproducer_spec(spec, report) is None


def test_tampered_report_and_wrong_spec_are_rejected():
    spec, report = evidence()
    report["image_id"] = "sha256:" + "b" * 64
    with pytest.raises(ValueError, match="binding"):
        reproducer_spec(spec, report)
    _, report = evidence()
    with pytest.raises(ValueError, match="binding"):
        reproducer_spec(spec.model_copy(update={"seed": 1}), report)


@pytest.mark.asyncio
@pytest.mark.parametrize("changed", ["source", "patch"])
async def test_stale_reproducer_does_not_start_docker(changed):
    spec, report = evidence()
    replay = reproducer_spec(spec, report)
    if changed == "source":
        replay = replay.model_copy(update={"reference_source_sha256": "f" * 64})
    else:
        replay = replay.model_copy(update={"reference_patch_sha256": "f" * 64})

    def forbidden(*args):
        raise AssertionError("stale replay must be rejected before Docker")

    result = await verify_patch(
        FIXTURE,
        (FIXTURE / "regression.diff").read_text(),
        replay,
        sandbox_factory=forbidden,
        image_resolver=forbidden,
    )
    assert result["status"] == "inconclusive" and not result["trials"] and result["source_unchanged"]


def test_cli_exports_a_private_reproducer_and_keeps_regression_exit_code(tmp_path, monkeypatch):
    import json
    import sys

    from code_agent import patch_verify

    spec, report = evidence()
    report.update(
        reason="owner_preservation_violation",
        case_count=1,
        counterexamples=[report["minimizations"][0]["confirmed_case"]],
    )
    report["fingerprint"] = fingerprint({key: value for key, value in report.items() if key != "fingerprint"})

    async def controlled_verification(*args, **kwargs):
        return report

    monkeypatch.chdir(tmp_path)
    output = tmp_path / "data/patch-verify/cli"
    fixture = FIXTURE.resolve()
    monkeypatch.setattr(patch_verify, "verify_patch", controlled_verification)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "agentforge-verify",
            "--repository",
            str(fixture),
            "--spec",
            str(fixture / "spec.json"),
            "--patch",
            str(fixture / "regression.diff"),
            "--output-dir",
            str(output),
        ],
    )
    with pytest.raises(SystemExit) as stopped:
        patch_verify.main()
    assert stopped.value.code == 2
    assert (output / ".gitignore").read_text() == "*\n"
    replay = PatchVerifySpec.model_validate_json((output / "reproducer.spec.json").read_bytes())
    assert replay == reproducer_spec(spec, report)
    envelope = json.loads((output / "report.json").read_bytes())
    assert envelope["fingerprint"] == fingerprint(
        {key: value for key, value in envelope.items() if key != "fingerprint"}
    )
