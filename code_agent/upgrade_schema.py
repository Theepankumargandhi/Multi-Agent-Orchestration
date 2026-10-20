"""Bounded schema intelligence over JSON only; application introspection stays in Docker."""
from __future__ import annotations

import copy
import math
from itertools import islice

from pydantic import Field, model_validator

from code_agent.upgrade_contracts import canonical
from code_agent.upgrade_inputs import campaign_from_pool, input_key, variants
from code_agent.upgrade_models import ContractCase, StrictModel, fingerprint

SCHEMA_GENERATOR = "schema-grounded-boundaries-v1"
SCHEMA_CONFIG = "upgradepilot-models.json"
SCHEMA_PROBE = "/opt/upgradepilot/schema_probe.py"
SCHEMA_IMAGE = "agentforge-upgradepilot-schema:local"


class SchemaTarget(StrictModel):
    seed_case_id: str = Field(pattern=r"^[A-Za-z0-9_.-]{1,80}$")
    model: str = Field(pattern=r"^[A-Za-z][A-Za-z0-9_]*(?:\.[A-Za-z][A-Za-z0-9_]*)*:[A-Za-z][A-Za-z0-9_]*$", max_length=200)


class SchemaPolicy(StrictModel):
    targets: list[SchemaTarget] = Field(min_length=1, max_length=4)

    @model_validator(mode="after")
    def unique_models(self):
        if len({row.model for row in self.targets}) != len(self.targets):
            raise ValueError("schema models must be unique")
        return self


def validate_schema_targets(root, spec, policy):
    from code_agent.upgrade_plan import source_files
    allowed = {path.resolve() for path in source_files(root, spec)}
    seeds = {case.case_id: case for case in spec.contracts}
    for target in policy.targets:
        case = seeds.get(target.seed_case_id)
        module = target.model.split(":")[0].replace(".", "/")
        if not any((root / name).resolve() in allowed for name in (module + ".py", module + "/__init__.py")):
            raise ValueError("schema model module is outside selected source scope")
        if case is None or case.expected_kind != "value" or len(case.args) != 1 or not isinstance(case.args[0], dict) or case.kwargs:
            raise ValueError("schema targets require an owner value contract with one object argument")


def normalize_schema(schema):
    nodes = 0

    def walk(value, depth=0, *, schema_node=False):
        nonlocal nodes
        nodes += 1
        if nodes > 2000 or depth > 20:
            raise ValueError("schema structure exceeds analysis limits")
        if isinstance(value, dict):
            if schema_node and "definitions" in value and "$defs" in value:
                raise ValueError("ambiguous schema definition namespaces")
            result = {}
            for key, item in value.items():
                if schema_node and key in {"title", "description", "examples"}:
                    continue
                normalized_key = "$defs" if schema_node and key == "definitions" else key
                if schema_node and key in {"properties", "definitions", "$defs"} and isinstance(item, dict):
                    result[normalized_key] = {name: walk(child, depth + 1, schema_node=True) for name, child in item.items()}
                elif schema_node and key in {"anyOf", "allOf", "oneOf"} and isinstance(item, list):
                    result[key] = [walk(child, depth + 1, schema_node=True) for child in item]
                elif schema_node and key in {"items", "additionalProperties"} and isinstance(item, dict):
                    result[key] = walk(item, depth + 1, schema_node=True)
                elif schema_node and key == "$ref" and isinstance(item, str) and item.startswith("#/definitions/"):
                    result[key] = "#/$defs/" + item.removeprefix("#/definitions/")
                else:
                    result[normalized_key] = walk(item, depth + 1)
            return result
        if isinstance(value, list):
            return [walk(item, depth + 1) for item in value]
        return value
    return walk(schema, schema_node=True)


def schema_samples(raw):
    """Construct candidates from a conservative subset; runtime observations remain the oracle."""
    root = normalize_schema(raw)
    visits = 0
    supported = {"$defs", "$ref", "type", "properties", "required", "additionalProperties", "items", "minItems", "maxItems",
        "minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum", "minLength", "maxLength", "enum", "const", "anyOf", "allOf", "default"}

    def sample(node, stack=(), depth=0):
        nonlocal visits
        visits += 1
        if visits > 500 or depth > 8 or not isinstance(node, dict):
            raise ValueError("unsupported or oversized schema graph")
        if set(node) - supported:
            raise ValueError("schema contains unsupported generation keywords")
        if "$ref" in node:
            ref = node["$ref"]
            if not isinstance(ref, str) or not ref.startswith("#/$defs/") or ref in stack:
                raise ValueError("external or recursive schema references are unsupported")
            name = ref.removeprefix("#/$defs/")
            if "/" in name or "~" in name or name not in root.get("$defs", {}) or set(node) - {"$ref", "default"}:
                raise ValueError("ambiguous schema reference")
            return sample(root["$defs"][name], (*stack, ref), depth + 1)
        if "allOf" in node:
            if len(node["allOf"]) != 1 or set(node) - {"allOf", "default", "$defs"}:
                raise ValueError("only a single-reference allOf is supported")
            return sample(node["allOf"][0], stack, depth + 1)
        if "anyOf" in node:
            if not 1 <= len(node["anyOf"]) <= 4 or set(node) - {"anyOf", "default", "$defs"}:
                raise ValueError("unsupported union schema")
            return unique([value for branch in node["anyOf"] for value in sample(branch, stack, depth + 1)])
        if "const" in node or "enum" in node:
            values = [node["const"]] if "const" in node else node["enum"]
            if not isinstance(values, list) or not 1 <= len(values) <= 16:
                raise ValueError("bounded nonempty enum required")
            return unique(values)
        kind = node.get("type")
        if kind == "object":
            properties, required = node.get("properties", {}), node.get("required", [])
            if (not isinstance(properties, dict) or not isinstance(required, list) or len(properties) > 12
                    or not all(isinstance(key, str) for key in required) or len(set(required)) != len(required) or not set(required) <= set(properties)
                    or isinstance(node.get("additionalProperties"), dict)):
                raise ValueError("bounded declared object properties required")
            options = {key: sample(value, stack, depth + 1) for key, value in sorted(properties.items())}
            base = {key: copy.deepcopy(options[key][0]) for key in required}
            rows = [base]
            for key, values in options.items():
                rows.extend({**copy.deepcopy(base), key: copy.deepcopy(value)} for value in values)
            return unique(rows)
        if kind == "array":
            low, high = node.get("minItems", 0), node.get("maxItems", 4)
            if not isinstance(low, int) or not isinstance(high, int) or not 0 <= low <= min(high, 4):
                raise ValueError("unsupported array bounds")
            values = sample(node.get("items"), stack, depth + 1)
            rows = [[]] if low == 0 else []
            if high:
                rows.extend([[copy.deepcopy(value) for _ in range(max(1, low))] for value in values])
            return unique(rows)
        if kind in {"integer", "number"}:
            for key in ("minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum"):
                if key in node and (type(node[key]) not in {int, float} or not math.isfinite(node[key])):
                    raise ValueError("finite numeric schema bounds required")
            low = node.get("minimum", min(-1, node.get("maximum", 1) - 2))
            high = node.get("maximum", max(2, low + 2))
            if "exclusiveMinimum" in node:
                low = node["exclusiveMinimum"] + 1
            if "exclusiveMaximum" in node:
                high = node["exclusiveMaximum"] - 1
            if not isinstance(low, (int, float)) or not isinstance(high, (int, float)) or not math.isfinite(low) or not math.isfinite(high):
                raise ValueError("nonfinite schema bounds")
            candidates = [0, 1, -1, low, high, 1.5] if kind == "number" else [0, 1, -1, math.ceil(low), math.floor(high)]
            values = [value for value in candidates if low <= value <= high]
            if not values:
                raise ValueError("unsupported or contradictory numeric bounds")
            return unique(values)
        if kind == "string":
            low, high = node.get("minLength", 0), node.get("maxLength", 64)
            if not isinstance(low, int) or not isinstance(high, int) or not 0 <= low <= min(high, 128):
                raise ValueError("unsupported string bounds")
            return unique([value for value in ("0", "1", "7", "42", "text", "é", "", "x" * low, "x" * min(high, 128)) if low <= len(value) <= high])
        if kind == "boolean":
            return [False, True]
        if kind == "null":
            return [None]
        raise ValueError("untyped or unsupported schema node")

    def unique(values):
        result = {}
        for value in values:
            encoded = canonical(value)
            if len(encoded.encode()) <= 7000:
                result.setdefault(encoded, value)
            if len(result) >= 48:
                break
        if not result:
            raise ValueError("no bounded schema candidates available")
        return list(result.values())
    return sample(root)


def schema_campaign(spec, policy, schema_policy, schemas):
    seeds = {case.case_id: case for case in spec.contracts}
    excluded = {input_key(case) for case in spec.contracts}
    pool, preferred = {}, set()
    for target in schema_policy.targets:
        seed = seeds[target.seed_case_id]
        candidates = []
        for value in schema_samples(schemas[target.model]):
            if not isinstance(value, dict):
                raise ValueError("top-level schema must describe an object payload")
            case = ContractCase(case_id="schema", entrypoint=seed.entrypoint, args=[value])
            candidates.append(case)
            if input_key(case) not in excluded:
                preferred.add(input_key(case))
                pool[input_key(case)] = case
        for seed in candidates[:8]:
            for case in islice(variants(seed), 256):
                if input_key(case) not in excluded:
                    pool.setdefault(input_key(case), case)
    pool = {key: case.model_copy(update={"case_id": "generated-" + key[:20]}) for key, case in pool.items()}
    provenance = {"schema_policy": schema_policy.model_dump(), "source_schema_fingerprint": fingerprint(schemas)}
    return campaign_from_pool(spec, policy, pool, generator=SCHEMA_GENERATOR, provenance=provenance, preferred=preferred)


def declaration_changes(source, target):
    """Bounded declaration differences, not automatic semantic incompatibility verdicts."""
    rows = []

    def walk(old, new, path):
        if len(rows) >= 80 or old == new:
            return
        if isinstance(old, dict) and isinstance(new, dict):
            for key in sorted(set(old) | set(new)):
                location = path + "/" + key.replace("~", "~0").replace("/", "~1")
                if key == "required" and all(isinstance(value, list) and all(isinstance(name, str) for name in value)
                        for value in (old.get(key, []), new.get(key, []))):
                    added, removed = sorted(set(new.get(key, [])) - set(old.get(key, []))), sorted(set(old.get(key, [])) - set(new.get(key, [])))
                    if added or removed:
                        rows.append({"path": location, "change": "required_fields_changed", "added_fields": added, "removed_fields": removed})
                elif key not in old or key not in new:
                    rows.append({"path": location, "change": "added" if key not in old else "removed",
                        "source_sha256": fingerprint(old.get(key)), "target_sha256": fingerprint(new.get(key))})
                else:
                    walk(old[key], new[key], location)
                if len(rows) >= 80:
                    break
        else:
            rows.append({"path": path, "change": "changed", "source_sha256": fingerprint(old), "target_sha256": fingerprint(new)})
    walk(normalize_schema(source), normalize_schema(target), "")
    return {"changed": bool(rows), "differences": rows, "difference_limit_reached": len(rows) == 80,
        "interpretation": "schema_declarations_only_not_behavioral_equivalence"}
