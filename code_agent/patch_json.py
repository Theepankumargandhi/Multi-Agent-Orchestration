"""Bounded JSON-schema inputs and schema-preserving reductions; no repository execution.

This deliberately small profile rejects unsupported JSON Schema keywords rather
than silently pretending to implement a complete validator.
"""

from __future__ import annotations

import copy
import json
import math
import random
import re
from typing import Any, Literal

from pydantic import Field, model_validator

from code_agent.upgrade_models import ContractCase, StrictModel, fingerprint, safe_relative

GENERATOR = "owner-json-domain-v1"
REDUCER = "schema-preserving-greedy-v1"
TYPES = {"null", "boolean", "integer", "number", "string", "array", "object"}
KEYS = {
    "null": set(),
    "boolean": set(),
    "integer": {"minimum", "maximum"},
    "number": {"minimum", "maximum"},
    "string": {"minLength", "maxLength"},
    "array": {"items", "minItems", "maxItems"},
    "object": {"properties", "required", "additionalProperties"},
}
ENTRYPOINT = r"^[A-Za-z][A-Za-z0-9_]*(?:\.[A-Za-z][A-Za-z0-9_]*)*:[A-Za-z][A-Za-z0-9_]*$"


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def _kind(schema: dict) -> str:
    value = schema["type"]
    return value if isinstance(value, str) else next(item for item in value if item != "null")


def _max_input_bytes(schema: dict) -> int:
    """Conservative default-json encoding bound, including escaping and spaces."""
    if "enum" in schema:
        return max(len(json.dumps(item, allow_nan=False).encode()) for item in schema["enum"])
    kind = _kind(schema)
    if kind == "null":
        size = 4
    elif kind == "boolean":
        size = 5
    elif kind == "integer":
        size = max(
            len(str(schema.get(key, default))) for key, default in (("minimum", -100), ("maximum", 100))
        )
    elif kind == "number":
        size = 32  # Finite Python double, including sign, fraction and exponent.
    elif kind == "string":
        size = 2 + 12 * schema.get("maxLength", 64)  # Astral code point: two escaped surrogates.
    elif kind == "array":
        count = schema.get("maxItems", 6)
        size = 2 + count * _max_input_bytes(schema["items"]) + max(0, count - 1) * 2
    else:
        properties = schema["properties"]
        size = 2 + sum(
            len(json.dumps(key)) + 2 + _max_input_bytes(child) for key, child in properties.items()
        )
        size += max(0, len(properties) - 1) * 2
    return max(size, 4) if type(schema["type"]) is list else size


def validate_schema(schema: dict) -> None:
    """Validate a closed, acyclic profile, including a bound on worst-case input size."""
    visited = set()

    def visit(node, depth):
        if type(node) is not dict or depth > 4 or id(node) in visited or len(visited) >= 48:
            raise ValueError("JSON schema exceeds its shape budget or contains aliases")
        visited.add(id(node))
        types = node.get("type")
        nullable = type(types) is list
        if nullable:
            if len(types) != 2 or any(type(item) is not str for item in types) or types.count("null") != 1:
                raise ValueError("only one concrete type plus null is supported")
            types = next(item for item in types if item != "null")
        if type(types) is not str or types not in TYPES:
            raise ValueError("a supported explicit JSON type is required")
        if set(node) - ({"type", "enum"} | KEYS[types]):
            raise ValueError("unsupported JSON schema keyword")
        if types in {"integer", "number"}:
            lower, upper = node.get("minimum", -100), node.get("maximum", 100)
            number_types = (int,) if types == "integer" else (int, float)
            if (
                type(lower) not in number_types
                or type(upper) not in number_types
                or not -1_000_000 <= lower <= upper <= 1_000_000
                or not math.isfinite(lower)
                or not math.isfinite(upper)
            ):
                raise ValueError("numeric bounds are invalid")
        if types in {"string", "array"}:
            first, last, maximum = (
                ("minLength", "maxLength", 64) if types == "string" else ("minItems", "maxItems", 6)
            )
            lower, upper = node.get(first, 0), node.get(last, maximum)
            if type(lower) is not int or type(upper) is not int or not 0 <= lower <= upper <= maximum:
                raise ValueError("length bounds are invalid")
        if types == "array":
            visit(node.get("items"), depth + 1)
        if types == "object":
            properties, required = node.get("properties"), node.get("required", [])
            if (
                type(properties) is not dict
                or len(properties) > 8
                or node.get("additionalProperties") is not False
                or any(
                    type(key) is not str or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]{0,39}", key)
                    for key in properties
                )
                or type(required) is not list
                or any(type(key) is not str for key in required)
                or len(set(required)) != len(required)
                or not set(required) <= set(properties)
            ):
                raise ValueError("object schemas must have bounded, closed properties")
            for child in properties.values():
                visit(child, depth + 1)
        if "enum" in node:
            values = node["enum"]
            if (
                types in {"object", "array"}
                or type(values) is not list
                or not 1 <= len(values) <= 8
                or any(
                    _equal(item, previous) for index, item in enumerate(values) for previous in values[:index]
                )
                or any(
                    not matches({key: value for key, value in node.items() if key != "enum"}, item)
                    for item in values
                )
            ):
                raise ValueError("enum must contain unique supported scalar values")

    visit(schema, 0)
    # Multiplicative array bounds can otherwise hide large generated payloads.
    if _max_input_bytes(schema) > 6000 or len(canonical(schema).encode()) > 6000:
        raise ValueError("JSON schema or worst-case input exceeds its byte budget")


def _equal(left, right) -> bool:
    if type(left) in (int, float) and type(right) in (int, float):
        return left == right
    return type(left) is type(right) and left == right


def matches(schema: dict, value: Any) -> bool:
    if "enum" in schema and not any(_equal(value, item) for item in schema["enum"]):
        return False
    if value is None:
        return schema["type"] == "null" or isinstance(schema["type"], list) and "null" in schema["type"]
    kind = _kind(schema)
    if kind == "null":
        return False
    if kind == "boolean":
        return type(value) is bool
    if kind in {"integer", "number"}:
        allowed = (int,) if kind == "integer" else (int, float)
        return (
            type(value) in allowed
            and schema.get("minimum", -100) <= value <= schema.get("maximum", 100)
            and math.isfinite(value)
        )
    if kind == "string":
        return type(value) is str and schema.get("minLength", 0) <= len(value) <= schema.get("maxLength", 64)
    if kind == "array":
        return (
            type(value) is list
            and schema.get("minItems", 0) <= len(value) <= schema.get("maxItems", 6)
            and all(matches(schema["items"], item) for item in value)
        )
    if kind == "object":
        return (
            type(value) is dict
            and set(schema.get("required", [])) <= set(value) <= set(schema["properties"])
            and all(matches(schema["properties"][key], item) for key, item in value.items())
        )
    return False


def _extreme(schema: dict, *, high: bool):
    if "enum" in schema:
        return copy.deepcopy(
            max(schema["enum"], key=lambda item: len(canonical(item))) if high else schema["enum"][0]
        )
    kind = _kind(schema)
    if kind == "null":
        return None
    if kind == "boolean":
        return high
    if kind in {"integer", "number"}:
        return schema.get("maximum", 100) if high else schema.get("minimum", -100)
    if kind == "string":
        return "x" * schema.get("maxLength" if high else "minLength", 64 if high else 0)
    if kind == "array":
        return [
            _extreme(schema["items"], high=high)
            for _ in range(schema.get("maxItems" if high else "minItems", 6 if high else 0))
        ]
    required = schema.get("required", [])
    return {
        key: _extreme(child, high=high)
        for key, child in sorted(schema["properties"].items())
        if high or key in required
    }


def _unique(values, limit=32):
    result, seen = [], set()
    for value in values:
        identity = canonical(value)
        if identity not in seen:
            seen.add(identity)
            result.append(value)
        if len(result) >= limit:
            break
    return result


def _witnesses(schema: dict) -> list:
    if "enum" in schema:
        return copy.deepcopy(schema["enum"])
    kind = _kind(schema)
    values = [None] if type(schema["type"]) is list else []
    values.extend([_extreme(schema, high=False), _extreme(schema, high=True)])
    if kind == "boolean":
        values.extend([False, True])
    elif kind in {"integer", "number"}:
        low, high = schema.get("minimum", -100), schema.get("maximum", 100)
        values.extend([0, -1, 1, low + 1, high - 1, (low + high) // 2])
        if kind == "number":
            values.extend([0.5, -0.5, (low + high) / 2])
    elif kind == "string":
        length = max(1, schema.get("minLength", 0))
        values.extend(char * length for char in ("a", "A", "0", " ", "é", "\n", '"', "\\"))
    elif kind == "array":
        children = _witnesses(schema["items"])
        size = max(1, schema.get("minItems", 0))
        values.extend([[child] * size for child in children])
    elif kind == "object":
        base = _extreme(schema, high=True)
        variants = [(key, _witnesses(child)) for key, child in sorted(schema["properties"].items())]
        # Round-robin prevents the first property from consuming the entire campaign.
        for index in range(max((len(rows) for _, rows in variants), default=0)):
            for key, rows in variants:
                if index < len(rows):
                    values.append({**base, key: rows[index]})
        values.extend(
            {key: value for key, value in base.items() if key != optional}
            for optional in base
            if optional not in schema.get("required", [])
        )
    return _unique(value for value in values if matches(schema, value))


def _random_value(schema: dict, rng: random.Random):
    if "enum" in schema:
        return copy.deepcopy(rng.choice(schema["enum"]))
    if type(schema["type"]) is list and rng.random() < 0.2:
        return None
    kind = _kind(schema)
    if kind == "boolean":
        return bool(rng.getrandbits(1))
    if kind == "integer":
        return rng.randint(schema.get("minimum", -100), schema.get("maximum", 100))
    if kind == "number":
        return rng.uniform(schema.get("minimum", -100), schema.get("maximum", 100))
    if kind == "string":
        return "".join(
            rng.choice("abAZ09 é")
            for _ in range(rng.randint(schema.get("minLength", 0), schema.get("maxLength", 64)))
        )
    if kind == "array":
        return [
            _random_value(schema["items"], rng)
            for _ in range(rng.randint(schema.get("minItems", 0), schema.get("maxItems", 6)))
        ]
    if kind == "object":
        return {
            key: _random_value(child, rng)
            for key, child in sorted(schema["properties"].items())
            if key in schema.get("required", []) or rng.random() < 0.5
        }
    return None


class JsonDomain(StrictModel):
    domain_id: str = Field(pattern=r"^[A-Za-z0-9_.-]{1,60}$")
    entrypoint: str = Field(pattern=ENTRYPOINT, max_length=200)
    input_schema: dict[str, Any]
    keyword: str | None = Field(default=None, pattern=r"^[A-Za-z][A-Za-z0-9_]{0,39}$")
    samples: int = Field(default=16, ge=2, le=24)
    expected_kind: Literal["value", "error"] = "value"
    expected_error: str | None = Field(default=None, pattern=r"^[A-Za-z][A-Za-z0-9_]{0,100}$")

    @model_validator(mode="after")
    def supported(self):
        validate_schema(self.input_schema)
        if (self.expected_kind == "error") != bool(self.expected_error):
            raise ValueError("error domains require an exact expected exception class")
        return self

    def case(self, value: Any, case_id: str) -> ContractCase:
        if not matches(self.input_schema, value):
            raise ValueError("input leaves the owner domain")
        return ContractCase(
            case_id=case_id,
            entrypoint=self.entrypoint,
            args=[copy.deepcopy(value)] if self.keyword is None else [],
            kwargs={} if self.keyword is None else {self.keyword: copy.deepcopy(value)},
            expected_kind=self.expected_kind,
            expected_error=self.expected_error,
        )

    def value(self, case: ContractCase):
        return case.args[0] if self.keyword is None else case.kwargs[self.keyword]


class ReductionPolicy(StrictModel):
    enabled: bool = True
    max_rounds: int = Field(default=3, ge=0, le=6)
    max_candidates: int = Field(default=8, ge=1, le=12)
    max_seconds: int = Field(default=120, ge=30, le=300)


class ContractApprovalBinding(StrictModel):
    patch_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    source_files: dict[str, str] = Field(min_length=1, max_length=20)
    review_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")

    @model_validator(mode="after")
    def bounded_sources(self):
        for path, digest in self.source_files.items():
            if (
                safe_relative(path) != path
                or not path.endswith(".py")
                or not re.fullmatch(r"[a-f0-9]{64}", digest)
            ):
                raise ValueError("review bindings require bounded Python source hashes")
        return self


def generate_json_cases(domain: JsonDomain, seed: int) -> list[ContractCase]:
    rng = random.Random(f"{GENERATOR}:{seed}:{domain.domain_id}")
    values = _witnesses(domain.input_schema)
    for _ in range(domain.samples * 2):
        values.append(_random_value(domain.input_schema, rng))
    return [
        domain.case(value, f"generated.{domain.domain_id}.{index}")
        for index, value in enumerate(_unique(values, domain.samples))
    ]


def complexity(value: Any) -> tuple[int, str]:
    text = canonical(value)
    return len(text.encode()), text


def reduction_candidates(schema: dict, value: Any, limit: int) -> list:
    """Strictly smaller in-domain values. Greedy search is not global minimality."""
    candidates = _witnesses(schema)
    kind = _kind(schema)
    if type(value) is dict and kind == "object":
        required = set(schema.get("required", []))
        candidates.append({key: item for key, item in value.items() if key in required})
        for key, item in sorted(value.items()):
            if key not in required:
                candidates.append({name: child for name, child in value.items() if name != key})
            candidates.extend(
                {**value, key: smaller}
                for smaller in reduction_candidates(schema["properties"][key], item, limit)
            )
    elif type(value) is list and kind == "array":
        candidates.extend([value[: len(value) // 2], value[len(value) // 2 :], *[[item] for item in value]])
        for index, item in enumerate(value):
            candidates.append(value[:index] + value[index + 1 :])
            candidates.extend(
                value[:index] + [smaller] + value[index + 1 :]
                for smaller in reduction_candidates(schema["items"], item, limit)
            )
    elif type(value) is str:
        candidates.extend([value[: len(value) // 2], value[: schema.get("minLength", 0)]])
    elif type(value) in (int, float):
        candidates.append(int(value / 2) if kind == "integer" else value / 2)
    valid = _unique(
        item for item in candidates if matches(schema, item) and complexity(item) < complexity(value)
    )
    return sorted(valid, key=complexity)[:limit]


def campaign_fingerprint(domains: list[JsonDomain], cases: list[ContractCase], seed: int) -> str:
    return fingerprint(
        {
            "generator": GENERATOR,
            "seed": seed,
            "domains": [row.model_dump() for row in domains],
            "cases": [row.model_dump() for row in cases],
        }
    )
