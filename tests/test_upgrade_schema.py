"""Schema analysis never imports fixture applications into the host test process."""
import copy
import json
from pathlib import Path

import pytest

from code_agent.upgrade_contracts import result_hash
from code_agent.upgrade_inputs import CounterexamplePolicy, generate_campaign, input_key
from code_agent.upgrade_models import SOURCE_VERSION, MigrationSpec
from code_agent.upgrade_schema import (
    SchemaPolicy,
    declaration_changes,
    normalize_schema,
    schema_campaign,
    schema_samples,
    validate_schema_targets,
)
from code_agent.upgrade_schema_runtime import parse_schemas

FIXTURE = Path(__file__).parent / "fixtures/upgrade_schema"
MODEL = "app.models:Request"


def recipe():
    return MigrationSpec.model_validate_json((FIXTURE / "migration.json").read_bytes())


def policy():
    return SchemaPolicy.model_validate_json((FIXTURE / "schema-policy.json").read_bytes())


def nested_schema():
    return {"type": "object", "properties": {"items": {"type": "array", "items": {"$ref": "#/definitions/Item"}}},
        "definitions": {"Item": {"type": "object", "properties": {"quantity": {"anyOf": [{"type": "integer"}, {"type": "string"}]}},
            "required": ["quantity"]}}}


def test_reference_normalization_preserves_property_names_and_default_values():
    old = {"title": "Not behavioral", "type": "object", "properties": {"description": {"type": "string", "default": "#/definitions/Literal"},
        "title": {"$ref": "#/definitions/Child"}}, "definitions": {"Child": {"type": "string"}}}
    normalized = normalize_schema(old)
    assert normalized["properties"]["description"]["default"] == "#/definitions/Literal"
    assert normalized["properties"]["title"]["$ref"] == "#/$defs/Child"
    assert "title" not in normalized and "$defs" in normalized
    new = copy.deepcopy(normalized)
    assert not declaration_changes(old, new)["changed"]


def test_source_schema_discovers_nested_valid_shapes_invisible_to_blind_mutation():
    source = nested_schema()
    samples = schema_samples(source)
    assert {"items": [{"quantity": "0"}]} in samples
    assert {"items": [{"quantity": 0}]} in samples
    assert all(isinstance(value, dict) for value in samples)
    spec, budget = recipe(), CounterexamplePolicy()
    blind = generate_campaign(spec, budget)
    guided = schema_campaign(spec, budget, policy(), {MODEL: source})
    assert guided == schema_campaign(spec, budget, policy(), {MODEL: source})
    assert not {input_key(row) for row in guided.discovery} & {input_key(row) for row in guided.final}
    assert not any(isinstance(value, dict) and value.get("items") and isinstance(value["items"], list)
        and isinstance(value["items"][0], dict) for case in (*blind.discovery, *blind.final) for value in case.args)
    for cases in (guided.discovery, guided.final):
        assert any(case.args and isinstance(case.args[0], dict) and isinstance(case.args[0].get("items"), list)
            and case.args[0]["items"] and isinstance(case.args[0]["items"][0], dict) for case in cases)


def test_schema_changes_are_declaration_evidence_not_an_equivalence_verdict():
    old = {"type": "object", "properties": {"label": {"type": "string"}}}
    new = {**old, "required": ["label"]}
    comparison = declaration_changes(old, new)
    assert comparison["changed"] and comparison["differences"][0]["path"] == "/required"
    assert "not_behavioral_equivalence" in comparison["interpretation"]


@pytest.mark.parametrize("schema", [
    {"$ref": "https://external.test/schema"},
    {"$ref": "#/definitions/Node", "definitions": {"Node": {"$ref": "#/definitions/Node"}}},
    {"type": "string", "pattern": "^complex.*$"},
    {"type": "object", "additionalProperties": {"type": "string"}},
    {"oneOf": [{"type": "string"}]},
    {"type": "array", "minItems": 5, "items": {"type": "integer"}},
    {"type": "object", "required": ["missing"]},
])
def test_unsupported_or_recursive_schema_holds_instead_of_silent_fallback(schema):
    with pytest.raises(ValueError):
        schema_samples(schema)


def test_numeric_string_enum_alias_and_array_bounds_are_bounded():
    assert all(value >= 100 for value in schema_samples({"type": "integer", "minimum": 100}))
    assert all(value <= -100 for value in schema_samples({"type": "number", "maximum": -100}))
    assert schema_samples({"enum": ["publishedAlias", "second"]}) == ["publishedAlias", "second"]
    assert all(2 <= len(value) <= 3 for value in schema_samples({"type": "string", "minLength": 2, "maxLength": 3}))
    assert all(len(value) == 2 for value in schema_samples({"type": "array", "minItems": 2, "maxItems": 2, "items": {"type": "boolean"}}))


def test_model_mapping_is_owner_allowlisted_and_confined_without_imports(tmp_path):
    validate_schema_targets(FIXTURE, recipe(), policy())
    invalid = SchemaPolicy(targets=[{"seed_case_id": "empty-order", "model": "outside.module:Model"}])
    with pytest.raises(ValueError):
        validate_schema_targets(FIXTURE, recipe(), invalid)
    (tmp_path / "app").mkdir()
    (tmp_path / "app/models.py").write_text("raise RuntimeError('must not import on host')\n")
    validate_schema_targets(tmp_path, recipe(), policy())


def test_schema_protocol_is_exact_repeated_membership_and_hash_bound():
    payload = {"protocol": "upgrade-model-schema-v1", "pydantic_version": SOURCE_VERSION,
        "models": [{"model": MODEL, "schema": nested_schema(), "sha256": result_hash(nested_schema())}]}
    assert parse_schemas(json.dumps(payload), policy(), SOURCE_VERSION) == {MODEL: nested_schema()}
    for change in ({"protocol": "wrong"}, {"pydantic_version": "2.0"}, {"models": []}, {"models": [None]}):
        with pytest.raises(ValueError):
            parse_schemas(json.dumps({**payload, **change}), policy(), SOURCE_VERSION)
    payload["models"][0]["sha256"] = "0" * 64
    with pytest.raises(ValueError):
        parse_schemas(json.dumps(payload), policy(), SOURCE_VERSION)
    with pytest.raises(ValueError):
        parse_schemas('{"protocol":"a","protocol":"b"}', policy(), SOURCE_VERSION)


def test_required_declaration_order_is_not_a_change_and_numeric_bound_types_are_checked():
    assert not declaration_changes({"required": ["a", "b"]}, {"required": ["b", "a"]})["changed"]
    assert declaration_changes({}, {"required": ["label"]})["differences"][0]["added_fields"] == ["label"]
    for value in (True, "invalid", float("inf")):
        with pytest.raises(ValueError):
            schema_samples({"type": "number", "minimum": value})


def test_schema_policy_rejects_duplicate_models_and_private_model_names():
    with pytest.raises(ValueError):
        SchemaPolicy(targets=[{"seed_case_id": "empty-order", "model": MODEL}] * 2)
    with pytest.raises(ValueError):
        SchemaPolicy(targets=[{"seed_case_id": "empty-order", "model": "app.models:_Private"}])


@pytest.mark.parametrize("failure", ["attestation", "timeout", "unstable", "mutation"])
def test_schema_runtime_rejects_unavailable_or_unstable_evidence(monkeypatch, failure):
    import hashlib
    from types import SimpleNamespace

    import code_agent.upgrade_schema_runtime as runtime
    from code_agent.upgrade_schema_runtime import read_schemas

    expected = hashlib.sha256(Path(runtime.__file__).with_name("upgrade_schema_probe.py").read_bytes()).hexdigest()
    calls = []
    manifests = iter(["before", "after"]) if failure == "mutation" else iter(["same"] * 8)
    monkeypatch.setattr(runtime, "manifest", lambda _: next(manifests))
    monkeypatch.setattr(runtime, "digest", lambda value: value)

    def run(_command):
        calls.append(True)
        if len(calls) == 1:
            return SimpleNamespace(exit_code=0, timed_out=False, stdout="wrong" if failure == "attestation" else expected)
        schema = {"type": "integer", "minimum": len(calls) if failure == "unstable" else 0}
        payload = {"protocol": "upgrade-model-schema-v1", "pydantic_version": SOURCE_VERSION,
            "models": [{"model": MODEL, "schema": schema, "sha256": result_hash(schema)}]}
        return SimpleNamespace(exit_code=0, timed_out=failure == "timeout", stdout=json.dumps(payload))
    with pytest.raises(ValueError):
        read_schemas(SimpleNamespace(workspace=object(), run=run), policy(), SOURCE_VERSION)


def test_schema_campaign_fingerprint_changes_with_source_declarations_and_policy():
    source = nested_schema()
    original = schema_campaign(recipe(), CounterexamplePolicy(), policy(), {MODEL: source})
    changed = copy.deepcopy(source)
    changed["description"] = "Source schema metadata is still bound in provenance."
    assert schema_campaign(recipe(), CounterexamplePolicy(), policy(), {MODEL: changed}).commitment != original.commitment


@pytest.mark.asyncio
async def test_frozen_campaign_mismatch_holds_before_docker_or_model_admission(monkeypatch):
    from code_agent.upgrade_runner import run_migration

    monkeypatch.setattr("code_agent.upgrade_runner.resolve_image", lambda _: "sha256:" + "a" * 64)
    def forbidden(*_args, **_kwargs):
        raise AssertionError("no container or model may be admitted for a changed campaign")
    monkeypatch.setattr("code_agent.upgrade_runner.DockerSandbox.start", forbidden)
    report = await run_migration(FIXTURE, recipe(), counterexample_policy=CounterexamplePolicy(),
        model="groq:never-called", authorize_provider=True, expected_campaign_commitment="0" * 64)
    assert report["status"] == "held" and report["reason"] == "hold_at:impact_analysis"
    assert report["source_unchanged"] and report["model_calls"] == report["live_provider_calls"] == 0


def test_authored_corpus_identity_is_portable_across_checkout_line_endings(tmp_path, monkeypatch):
    import shutil

    import evals.upgrade_schema_evaluation as evaluation

    before = evaluation.corpus_fingerprint()
    destination = tmp_path / "fixture"
    shutil.copytree(FIXTURE, destination)
    for path in destination.rglob("*"):
        if path.is_file() and "__pycache__" not in path.parts:
            path.write_bytes(path.read_bytes().replace(b"\r\n", b"\n").replace(b"\n", b"\r\n"))
    monkeypatch.setattr(evaluation, "FIXTURE", destination)
    assert evaluation.corpus_fingerprint() == before
