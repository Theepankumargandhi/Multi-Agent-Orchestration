"""Real schema extraction in pinned old/new environments; never import applications on host."""
import pytest

from code_agent.models import SandboxPolicy
from code_agent.sandbox import DockerSandbox
from code_agent.upgrade_inputs import CounterexamplePolicy
from code_agent.upgrade_models import SOURCE_VERSION, TARGET_VERSION, V1_PYTHON, V2_PYTHON, fingerprint
from code_agent.upgrade_runner import resolve_image
from code_agent.upgrade_schema import SCHEMA_CONFIG, SCHEMA_IMAGE, SchemaPolicy, declaration_changes
from code_agent.upgrade_schema_runtime import prepare_schema_campaign, read_schemas
from code_agent.verification import repository_fingerprint
from tests.test_upgrade_schema import FIXTURE, policy, recipe


def test_real_schema_extraction_detects_requiredness_and_preserves_source():
    if not DockerSandbox.available() or not DockerSandbox.image_available(SCHEMA_IMAGE):
        pytest.skip("Dedicated schema image required; never substitute host imports")
    source_before = repository_fingerprint(FIXTURE, 5000, 50000000)
    image = resolve_image(SCHEMA_IMAGE)
    sandbox_policy = SandboxPolicy(image=image, allowed_images={image}, allowed_test_executables={V1_PYTHON, V2_PYTHON}, max_output_chars=100000)
    schema_policy = SchemaPolicy(targets=[{"seed_case_id": "empty-order", "model": "app.models:NullableProfile"},
        {"seed_case_id": "empty-order", "model": "app.models:Request"}])
    with DockerSandbox(FIXTURE, sandbox_policy) as sandbox:
        sandbox.workspace.write_file(SCHEMA_CONFIG, schema_policy.model_dump_json())
        old, old_receipt = read_schemas(sandbox, schema_policy, SOURCE_VERSION)
        new, new_receipt = read_schemas(sandbox, schema_policy, TARGET_VERSION)
        change = declaration_changes(old["app.models:NullableProfile"], new["app.models:NullableProfile"])
        assert any(row.get("added_fields") == ["label"] for row in change["differences"])
        assert not declaration_changes(old["app.models:Request"], new["app.models:Request"])["changed"]
        assert old_receipt["fresh_process_repeats"] == new_receipt["fresh_process_repeats"] == 2
    assert repository_fingerprint(FIXTURE, 5000, 50000000) == source_before
    campaign, receipt = prepare_schema_campaign(FIXTURE, recipe(), policy(), CounterexamplePolicy())
    assert receipt["fingerprint"] == fingerprint({"app.models:Request": old["app.models:Request"]})
    assert len(campaign.discovery) == 24 and len(campaign.final) == 16
    assert repository_fingerprint(FIXTURE, 5000, 50000000) == source_before
