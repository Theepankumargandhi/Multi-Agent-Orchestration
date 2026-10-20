"""Deterministic bounded JSON exploration; never import the application being migrated."""
from __future__ import annotations

import copy
import random
import re
from dataclasses import dataclass
from itertools import islice
from typing import Any

from pydantic import Field, model_validator

from code_agent.upgrade_contracts import canonical
from code_agent.upgrade_models import ContractCase, MigrationSpec, StrictModel, fingerprint

GENERATOR_VERSION = "json-boundaries-stratified-v1"
REDUCER_VERSION = "shape-preserving-bounded-v1"
BOUNDARIES = (None, False, True, 0, 1, -1, 2**53 + 1, 0.0, -0.1, 1.5,
    "", "0", "1", "-1", "7", "42", "1.5", " 1 ", "null", "text", "é", [], [None], {})


class CounterexamplePolicy(StrictModel):
    seed: int = Field(default=1729, ge=0, le=2**32 - 1)
    discovery_cases: int = Field(default=24, ge=1, le=50)
    final_cases: int = Field(default=16, ge=1, le=50)
    max_counterexamples: int = Field(default=2, ge=1, le=4)
    max_shrink_evaluations: int = Field(default=12, ge=0, le=40)

    @model_validator(mode="after")
    def bounded_total(self):
        if self.discovery_cases + self.final_cases > 80:
            raise ValueError("at most 80 generated cases per campaign")
        return self


@dataclass(frozen=True)
class InputCampaign:
    discovery: tuple[ContractCase, ...]
    final: tuple[ContractCase, ...]
    commitment: str
    discovery_fingerprint: str
    final_fingerprint: str


def input_key(case: ContractCase) -> str:
    return fingerprint([case.entrypoint, case.args, case.kwargs])


def _paths(value: Any, prefix: tuple = (), depth: int = 0):
    """Stable shallow traversal, capped independently of input size."""
    yield prefix
    if depth >= 5:
        return
    children = sorted(value)[:12] if isinstance(value, dict) else range(min(len(value), 12)) if isinstance(value, list) else ()
    for key in children:
        yield from _paths(value[key], (*prefix, key), depth + 1)


def _replace(value, path, replacement=None, *, remove=False):
    result = copy.deepcopy(value)
    if not path:
        return copy.deepcopy(replacement)
    parent = result
    for part in path[:-1]:
        parent = parent[part]
    if remove:
        del parent[path[-1]]
    else:
        parent[path[-1]] = copy.deepcopy(replacement)
    return result


def variants(case: ContractCase):
    """Single-site changes cover nested values, omissions, arrays and extra keys."""
    payload = {"args": case.args, "kwargs": case.kwargs}
    for path in islice(_paths(payload), 64):
        if len(path) < 2:  # Keep the invocation envelope and positional/keyword containers valid.
            continue
        for value in BOUNDARIES:
            changed = _replace(payload, path, value)
            try:
                yield ContractCase(case_id="generated", entrypoint=case.entrypoint, **changed)
            except ValueError:
                continue
        changed = _replace(payload, path, remove=True)
        yield ContractCase(case_id="generated", entrypoint=case.entrypoint, **changed)
        current = payload
        for part in path:
            current = current[part]
        if isinstance(current, dict):
            extra = {**current, "upgradepilot_extra": "unexpected"}
            changed = _replace(payload, path, extra)
            try:
                yield ContractCase(case_id="generated", entrypoint=case.entrypoint, **changed)
            except ValueError:
                continue


def _shape(value, depth=0):
    if depth >= 5:
        return type(value).__name__
    if isinstance(value, dict):
        return {key: _shape(value[key], depth + 1) for key in sorted(value)[:12]}
    if isinstance(value, list):
        return [_shape(item, depth + 1) for item in value[:12]]
    if isinstance(value, str) and re.fullmatch(r"\s*[+-]?\d+(?:\.\d+)?\s*", value):
        return "numeric-string"
    if isinstance(value, float) and not value.is_integer():
        return "fractional-float"
    return type(value).__name__


def generate_campaign(spec: MigrationSpec, policy: CounterexamplePolicy, *, excluded_inputs=frozenset()) -> InputCampaign:
    excluded = {input_key(case) for case in spec.contracts} | set(excluded_inputs)
    pool = {}
    # Fixed ceilings prevent an owner seed from causing unbounded traversal/work.
    for seed in spec.contracts[:8]:
        for case in islice(variants(seed), 512):
            key = input_key(case)
            if key not in excluded:
                pool.setdefault(key, case.model_copy(update={"case_id": "generated-" + key[:20]}))
    provenance = {"excluded_inputs": sorted(excluded_inputs)} if excluded_inputs else None
    return campaign_from_pool(spec, policy, pool, provenance=provenance)


def campaign_from_pool(spec, policy, pool, *, generator=GENERATOR_VERSION, provenance=None, preferred=frozenset()):
    keys = sorted(pool)
    random.Random(policy.seed).shuffle(keys)
    count = policy.discovery_cases + policy.final_cases
    if len(keys) < count:
        raise ValueError("insufficient distinct generated inputs; add representative owner seeds")
    groups = {}
    for key in keys:
        row = pool[key]
        groups.setdefault(canonical([row.entrypoint, _shape(row.args), _shape(row.kwargs)]), []).append(key)
    discovery_keys, final_keys = [], []
    # Cover common input-shape/type families in both sets, without observing outcomes.
    for group in sorted(groups.values(), key=lambda group: (not bool(set(group) & preferred), -len(group))):
        group = sorted(group, key=lambda key: key not in preferred)
        if len(group) >= 2 and len(discovery_keys) < policy.discovery_cases and len(final_keys) < policy.final_cases:
            discovery_keys.append(group[0])
            final_keys.append(group[1])
    selected = set(discovery_keys + final_keys)
    remaining = sorted((key for key in keys if key not in selected), key=lambda key: key not in preferred)
    needed = policy.discovery_cases - len(discovery_keys)
    discovery_keys.extend(remaining[:needed])
    final_keys.extend(remaining[needed:needed + policy.final_cases - len(final_keys)])
    discovery = tuple(pool[key] for key in discovery_keys)
    final = tuple(pool[key] for key in final_keys)
    discovery_hash = fingerprint([case.model_dump() for case in discovery])
    final_hash = fingerprint([case.model_dump() for case in final])
    identity = {"generator": generator, "policy": policy.model_dump(),
        "spec": fingerprint(spec.model_dump()), "discovery": discovery_hash, "final": final_hash}
    if provenance is not None:
        identity["provenance"] = provenance
    commitment = fingerprint(identity)
    return InputCampaign(discovery, final, commitment, discovery_hash, final_hash)


def complexity(case: ContractCase):
    encoded = canonical([case.args, case.kwargs])
    return len(encoded.encode()), encoded


def shrink_candidates(case: ContractCase, limit: int) -> list[ContractCase]:
    if limit <= 0:
        return []
    seen, result = {input_key(case)}, []
    for candidate in variants(case):
        key = input_key(candidate)
        if key in seen or complexity(candidate) >= complexity(case):
            continue
        seen.add(key)
        result.append(candidate.model_copy(update={"case_id": "generated-" + key[:20]}))
        if len(result) >= 512:
            break
    original_shape = _shape([case.args, case.kwargs])
    return sorted(result, key=lambda row: (_shape([row.args, row.kwargs]) != original_shape, complexity(row)))[:limit]
