"""Strict parsing and typed differential comparison; no repository imports on the host."""
from __future__ import annotations

import hashlib
import json

from code_agent.upgrade_models import MigrationSpec


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False, ensure_ascii=False)


def result_hash(result):
    return hashlib.sha256(canonical(result).encode()).hexdigest()


def _unique_pairs(items):
    result = {}
    for key, value in items:
        if key in result:
            raise ValueError("duplicate JSON keys")
        result[key] = value
    return result


def parse_probe(stdout: str, spec: MigrationSpec, expected_version: str) -> dict:
    if len(stdout.encode()) > 80000:
        raise ValueError("probe output exceeds budget")
    data = json.loads(stdout, object_pairs_hook=_unique_pairs,
        parse_constant=lambda _: (_ for _ in ()).throw(ValueError("nonfinite JSON")))
    if (not isinstance(data, dict) or set(data) != {"protocol", "pydantic_version", "outcomes"} or data["protocol"] != "upgradepilot-probe-v1"
            or data["pydantic_version"] != expected_version or not isinstance(data["outcomes"], list)):
        raise ValueError("probe protocol or dependency identity mismatch")
    expected = [case.case_id for case in spec.contracts]
    if any(not isinstance(row, dict) for row in data["outcomes"]) or [row.get("case_id") for row in data["outcomes"]] != expected:
        raise ValueError("probe cases are missing, duplicated or reordered")
    for row in data["outcomes"]:
        if set(row) != {"case_id", "result", "sha256"} or not isinstance(row["result"], dict) or row["sha256"] != result_hash(row["result"]):
            raise ValueError("probe outcome integrity mismatch")
        result = row["result"]
        if result.get("kind") == "value" and set(result) == {"kind", "value"}:
            continue
        if result.get("kind") == "error" and set(result) == {"kind", "type"} and isinstance(result["type"], str):
            continue
        raise ValueError("probe import, timeout or serialization was unavailable")
    return data


def validate_source_contracts(batch, spec):
    for case, row in zip(spec.contracts, batch["outcomes"], strict=True):
        result = row["result"]
        if result["kind"] != case.expected_kind or (case.expected_error and result.get("type") != case.expected_error):
            return False
    return True


def compare_contracts(source, target, spec):
    expected = [case.case_id for case in spec.contracts]
    if [row["case_id"] for row in source["outcomes"]] != expected or [row["case_id"] for row in target["outcomes"]] != expected:
        raise ValueError("comparison cases must match the frozen owner specification")
    approved = {row.case_id: row for row in spec.accepted_differences}
    rows = []
    for old, new in zip(source["outcomes"], target["outcomes"], strict=True):
        accepted = approved.get(old["case_id"])
        equal = old["sha256"] == new["sha256"]
        waived = bool(accepted and accepted.source_result_sha256 == old["sha256"] and accepted.target_result_sha256 == new["sha256"])
        rows.append({"case_id": old["case_id"], "status": "preserved" if equal else "owner_accepted_change" if waived else "regression",
            "source_sha256": old["sha256"], "target_sha256": new["sha256"]})
    return {"passed": all(row["status"] != "regression" for row in rows), "cases": rows,
            "preserved": sum(row["status"] == "preserved" for row in rows),
            "accepted_changes": sum(row["status"] == "owner_accepted_change" for row in rows),
            "regressions": sum(row["status"] == "regression" for row in rows)}
