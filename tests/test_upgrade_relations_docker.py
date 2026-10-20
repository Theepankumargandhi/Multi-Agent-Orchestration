"""Real old/new agreement can coexist with a violated owner law; never import applications on host."""
from dataclasses import replace

import pytest

from code_agent.models import SandboxPolicy
from code_agent.sandbox import DockerSandbox
from code_agent.upgrade_models import SOURCE_VERSION, TARGET_VERSION, V1_PYTHON, V2_PYTHON
from code_agent.upgrade_relations import assess_relation, generate_relations
from code_agent.upgrade_runner import resolve_image
from code_agent.upgrade_session_runtime import SessionReader, compare_sessions
from code_agent.upgrade_sessions import SESSION_IMAGE
from code_agent.verification import repository_fingerprint
from evals.upgrade_relation_evaluation import FIXTURE, policy, recipe


@pytest.mark.asyncio
async def test_matching_old_new_traces_still_fail_the_rollback_law():
    if not DockerSandbox.available() or not DockerSandbox.image_available(SESSION_IMAGE):
        pytest.skip("Dedicated session image required; never substitute host application imports")
    spec = recipe()
    campaign = generate_relations(spec, policy())
    original = next(row for row in campaign.discovery if row.relation == "rejected_call_rollback")
    case = replace(original, scenarios=tuple(row.model_copy(update={"factory": "app.models:create_broken_wallet"})
        for row in original.scenarios))
    image = resolve_image(SESSION_IMAGE)
    sandbox_policy = SandboxPolicy(image=image, allowed_images={image}, max_output_chars=100000,
        allowed_test_executables={V1_PYTHON, V2_PYTHON})
    before = repository_fingerprint(FIXTURE, 5000, 50000000)
    reader = SessionReader(spec)
    with DockerSandbox(FIXTURE, sandbox_policy) as sandbox:
        await reader.attest_one(sandbox)
        old = await reader.observe(sandbox, case.scenarios[0], SOURCE_VERSION)
        new = await reader.observe(sandbox, case.scenarios[0], TARGET_VERSION)
        assert compare_sessions(case.scenarios[0], old, new)["passed"]
        for observations in (old, new):
            result = assess_relation(case, [observations])
            assert not result["passed"] and result["violations"] == ["rejected_call_changed_state"]
    assert repository_fingerprint(FIXTURE, 5000, 50000000) == before
