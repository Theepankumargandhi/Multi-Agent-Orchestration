from pathlib import Path

import pytest

from code_agent.patch_inputs import PatchVerifySpec
from code_agent.patch_verify import reproducer_spec, tree_fingerprint, verify_patch
from code_agent.sandbox import DockerSandbox
from code_agent.upgrade_models import fingerprint

FIXTURE = Path(__file__).parent / "fixtures/patch_structured"
pytestmark = pytest.mark.integration


@pytest.mark.asyncio
async def test_green_owner_tests_do_not_hide_a_structured_regression_and_reproducer_is_reduced():
    if not DockerSandbox.available() or not DockerSandbox.image_available("agentforge-code-sandbox:local"):
        pytest.skip("requires the prebuilt offline coding sandbox")
    spec = PatchVerifySpec.model_validate_json((FIXTURE / "spec.json").read_bytes())
    before = tree_fingerprint(FIXTURE)
    report = await verify_patch(FIXTURE, (FIXTURE / "regression.diff").read_text(), spec)
    assert report["status"] == "regression_found", report["reason"]
    assert all(row["tests"]["passed"] for row in report["trials"]) and len(report["trials"]) == 4
    reduction = report["minimizations"][0]
    assert reduction["status"] == "reduced"
    assert reduction["input_bytes_after"] < reduction["input_bytes_before"]
    assert reduction["fingerprint"] == fingerprint(
        {key: value for key, value in reduction.items() if key != "fingerprint"}
    )
    assert report["counterexamples"] == [reduction["confirmed_case"]]
    assert report["json_campaign_sha256"] and report["live_provider_calls"] == 0
    assert before == tree_fingerprint(FIXTURE) and report["source_unchanged"]
    replay = reproducer_spec(spec, report)
    assert replay is not None and not replay.json_domains and not replay.integer_domains
    repeated = await verify_patch(
        FIXTURE, (FIXTURE / "regression.diff").read_text(), replay, mode="contracts"
    )
    assert repeated["status"] == "regression_found" and repeated["source_unchanged"]
    assert len(repeated["counterexamples"]) == 1 and all(row["tests"]["passed"] for row in repeated["trials"])
