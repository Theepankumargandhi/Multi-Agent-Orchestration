from __future__ import annotations

import json

import pytest

from code_agent.patch_inputs import PatchVerifySpec, generated_cases
from code_agent.patch_json import (
    JsonDomain,
    ReductionPolicy,
    complexity,
    generate_json_cases,
    matches,
    reduction_candidates,
)
from code_agent.patch_reduction import reduce_counterexample


def cart_schema():
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["items"],
        "properties": {
            "items": {
                "type": "array",
                "minItems": 0,
                "maxItems": 4,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["quantity", "price"],
                    "properties": {
                        "quantity": {"type": "integer", "minimum": 0, "maximum": 5},
                        "price": {"type": "integer", "minimum": 1, "maximum": 5},
                        "label": {"type": "string", "maxLength": 8},
                    },
                },
            },
            "coupon": {"type": ["string", "null"], "maxLength": 8},
            "priority": {"type": "boolean"},
        },
    }


def domain(**updates):
    return JsonDomain(domain_id="cart", entrypoint="app:total", input_schema=cart_schema(), **updates)


@pytest.mark.parametrize("seed", range(12))
def test_nested_generation_is_valid_seeded_bounded_and_covers_boundaries(seed):
    selected = domain()
    cases = generate_json_cases(selected, seed)
    assert cases == generate_json_cases(selected, seed)
    assert 1 <= len(cases) <= selected.samples
    assert len({json.dumps(row.args, sort_keys=True) for row in cases}) == len(cases)
    values = [row.args[0] for row in cases]
    assert all(matches(selected.input_schema, value) for value in values)
    assert any(not value["items"] for value in values)
    assert any(any(item["quantity"] == 0 for item in value["items"]) for value in values)
    assert any(value.get("coupon", "missing") is None for value in values)
    assert all(len(row.model_dump_json().encode()) < 8000 for row in cases)


@pytest.mark.parametrize(
    "schema",
    [
        {"type": "null"},
        {"type": "boolean"},
        {"type": "integer", "minimum": -2, "maximum": 2},
        {"type": "number", "minimum": -0.5, "maximum": 0.5},
        {"type": "string", "minLength": 1, "maxLength": 4},
        {"type": ["integer", "null"], "enum": [None, 0, 2]},
        {"type": "array", "items": {"type": "boolean"}, "minItems": 2, "maxItems": 3},
    ],
)
def test_all_supported_scalar_and_array_shapes(schema):
    selected = JsonDomain(domain_id="value", entrypoint="app:check", input_schema=schema, keyword="payload")
    for case in generate_json_cases(selected, 11):
        assert not case.args and matches(schema, case.kwargs["payload"])


@pytest.mark.parametrize(
    "schema",
    [
        {},
        {"type": "any"},
        {"type": ["string", "integer"]},
        {"type": ["null", "null"]},
        {"type": "integer", "minimum": True},
        {"type": "integer", "minimum": 2, "maximum": 1},
        {"type": "integer", "maximum": 10**1000},
        {"type": "number", "minimum": float("nan")},
        {"type": "string", "maxLength": 65},
        {"type": "string", "pattern": ".*"},
        {"type": "integer", "$ref": "https://must-not-fetch.invalid/schema"},
        {"type": "array", "maxItems": 7, "items": {"type": "boolean"}},
        {"type": "object", "properties": {}, "additionalProperties": True},
        {"type": "object", "properties": {}, "required": ["missing"], "additionalProperties": False},
        {"type": "number", "enum": [0, 0.0]},
        {"type": "integer", "enum": [True]},
        {"type": "string", "enum": ["duplicate", "duplicate"]},
        {"type": "array", "items": {"type": "integer"}, "enum": [[]]},
    ],
)
def test_unsupported_or_ambiguous_schemas_fail_closed(schema):
    with pytest.raises(ValueError):
        JsonDomain(domain_id="bad", entrypoint="app:check", input_schema=schema)


def test_json_type_semantics_do_not_confuse_booleans_and_numbers():
    assert matches({"type": "number", "enum": [0]}, 0.0)
    assert not matches({"type": "integer"}, True)
    assert not matches({"type": "integer", "enum": [1]}, True)


@pytest.mark.parametrize(
    "leaf,levels",
    [
        ({"type": "string", "minLength": 10, "maxLength": 10}, 3),
        ({"type": "integer", "minimum": -1_000_000, "maximum": 0}, 4),
    ],
)
def test_admission_bounds_escaped_unicode_and_negative_numeric_payloads(leaf, levels):
    schema = leaf
    for _ in range(levels):
        schema = {"type": "array", "minItems": 6, "maxItems": 6, "items": schema}
    with pytest.raises(ValueError, match="byte budget"):
        JsonDomain(domain_id="oversized", entrypoint="app:check", input_schema=schema)


def test_deep_recursive_and_alias_schemas_are_rejected():
    cyclic = {"type": "array"}
    cyclic["items"] = cyclic
    with pytest.raises(ValueError):
        JsonDomain(domain_id="bad", entrypoint="app:check", input_schema=cyclic)
    deep = {"type": "integer"}
    for _ in range(6):
        deep = {"type": "array", "items": deep}
    with pytest.raises(ValueError):
        JsonDomain(domain_id="bad", entrypoint="app:check", input_schema=deep)


def test_reducer_only_proposes_strictly_smaller_in_domain_inputs():
    selected = domain()
    value = {
        "items": [{"quantity": 0, "price": 5, "label": "abcdefgh"}] * 4,
        "coupon": None,
        "priority": True,
    }
    values = reduction_candidates(selected.input_schema, value, 12)
    assert values and all(
        matches(selected.input_schema, item) and complexity(item) < complexity(value) for item in values
    )
    assert value["items"][0]["label"] == "abcdefgh"
    assert values == reduction_candidates(selected.input_schema, value, 12)


def test_mixed_domains_have_unique_ids_and_one_total_budget():
    base = {
        "task_id": "cart",
        "source_paths": ["app.py"],
        "test_paths": ["owner_checks.py"],
        "json_domains": [domain().model_dump()],
        "preservation_authorized": True,
    }
    spec = PatchVerifySpec.model_validate(base)
    assert generated_cases(spec) == generate_json_cases(domain(), spec.seed)
    with pytest.raises(ValueError):
        PatchVerifySpec.model_validate(
            {
                **base,
                "integer_domains": [
                    {"domain_id": "cart", "entrypoint": "app:total", "minimum": 0, "maximum": 5}
                ],
            }
        )
    with pytest.raises(ValueError):
        PatchVerifySpec.model_validate(
            {
                **base,
                "json_domains": [
                    {**domain().model_dump(), "domain_id": f"domain{n}", "samples": 24} for n in range(4)
                ],
            }
        )


@pytest.mark.asyncio
async def test_reduction_requires_isolated_confirmation_and_retains_provenance():
    selected = domain()
    parent = selected.case(
        {"items": [{"quantity": 0, "price": 5, "label": "abcdefgh"}] * 4, "coupon": None}, "generated.cart.0"
    )
    calls = []

    async def evaluate(cases, deadline):
        calls.append(len(cases))
        hits = [
            case.case_id for case in cases if any(item["quantity"] == 0 for item in case.args[0]["items"])
        ]
        return {
            "status": "regression_found" if hits else "checks_passed",
            "reason": "authored_simulation",
            "changed": hits,
            "trials": [],
        }

    result = await reduce_counterexample(selected, parent, ReductionPolicy(), evaluate)
    assert result["status"] == "reduced" and result["input_bytes_after"] < result["input_bytes_before"]
    assert calls[0] == calls[-1] == 1 and result["confirmations"][-1]["changed"]
    assert result["parent_case_id"] == parent.case_id and result["fingerprint"]
    assert result["minimality"] == "bounded_greedy_not_global"


@pytest.mark.asyncio
async def test_batch_only_stateful_difference_cannot_be_a_standalone_counterexample():
    selected = domain()
    parent = generate_json_cases(selected, 1)[0]

    async def evaluate(cases, deadline):
        return {"status": "checks_passed", "reason": "not_reproduced", "changed": [], "trials": []}

    result = await reduce_counterexample(selected, parent, ReductionPolicy(), evaluate)
    assert result["status"] == "unconfirmed" and result["confirmed_case"] is None and result["fingerprint"]


@pytest.mark.asyncio
async def test_failed_smaller_confirmation_retains_only_the_confirmed_parent():
    selected = domain()
    parent = selected.case({"items": [{"quantity": 0, "price": 5}] * 4}, "generated.cart.0")

    async def evaluate(cases, deadline):
        hits = [case.case_id for case in cases if len(cases) > 1 or case.case_id == parent.case_id]
        return {
            "status": "regression_found" if hits else "inconclusive",
            "reason": "simulation",
            "changed": hits,
            "trials": [],
        }

    result = await reduce_counterexample(selected, parent, ReductionPolicy(), evaluate)
    assert result["confirmed_case"] == parent.model_dump()
    assert result["status"] == "confirmed_unchanged"
