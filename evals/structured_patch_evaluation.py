"""Authored JSON regression controls using real offline Docker, not real maintainer outcomes."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from code_agent.patch_inputs import PatchVerifySpec
from code_agent.patch_verify import (
    reserve_private_directory,
    tree_fingerprint,
    verify_patch,
    write_private_json,
)
from code_agent.upgrade_models import fingerprint

FIXTURE = Path(__file__).resolve().parent.parent / "tests/fixtures/patch_structured"


async def evaluate(on_result=None):
    spec = PatchVerifySpec.model_validate_json((FIXTURE / "spec.json").read_bytes())
    runs = [
        ("regression", "tests_only", "inconclusive"),
        ("regression", "contracts", "checks_passed"),
        ("regression", "generated", "regression_found"),
        ("correct", "generated", "checks_passed"),
        ("intentional", "generated", "checks_passed"),
        ("invalid_owner_domain", "generated", "inconclusive"),
    ]
    before = tree_fingerprint(FIXTURE)
    controls = []
    for name, mode, expected in runs:
        selected = spec
        if name == "invalid_owner_domain":
            value = spec.model_dump()
            value["json_domains"][0]["input_schema"]["type"] = ["object", "null"]
            selected = PatchVerifySpec.model_validate(value)
        patch = (FIXTURE / f"{'regression' if name == 'invalid_owner_domain' else name}.diff").read_text()
        report = await verify_patch(FIXTURE, patch, selected, mode=mode)
        passed = (
            report["status"] == expected
            and report["source_unchanged"]
            and len(report["trials"]) == 4
            and all(row["tests"]["passed"] for row in report["trials"])
        )
        if expected == "inconclusive":
            reason = (
                "original_behavior_does_not_match_owner_scope"
                if name == "invalid_owner_domain"
                else "no_owner_authorized_behavioral_scope"
            )
            passed = passed and report["reason"] == reason
        if name == "regression" and mode == "generated":
            reduction = report["minimizations"][0] if report["minimizations"] else {}
            passed = (
                passed
                and reduction.get("status") == "reduced"
                and reduction["input_bytes_after"] < reduction["input_bytes_before"]
            )
            passed = (
                passed
                and bool(report["counterexamples"])
                and any(
                    item["case"] == reduction["confirmed_case"]
                    and item["status"] == "regression_found"
                    and len(item["trials"]) == 4
                    and item["case"]["case_id"] in item["changed"]
                    for item in reduction["confirmations"]
                )
            )
        controls.append(
            {"case": name, "mode": mode, "expected": expected, "passed": bool(passed), "report": report}
        )
        if on_result is not None:
            on_result(controls[-1])
    report = {
        "evidence_kind": "authored_json_controls_actual_docker",
        "real_external_cases": 0,
        "live_provider_calls": 0,
        "total": len(controls),
        "passed": sum(row["passed"] for row in controls),
        "source_unchanged": before == tree_fingerprint(FIXTURE),
        "controls": controls,
    }
    report["fingerprint"] = fingerprint(report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output = reserve_private_directory(args.output_dir)

    def retain(row):
        write_private_json(output / f"{row['case']}-{row['mode']}.json", row)
        print(
            json.dumps(
                {
                    "case": row["case"],
                    "mode": row["mode"],
                    "passed": row["passed"],
                    "status": row["report"]["status"],
                }
            ),
            flush=True,
        )

    report = asyncio.run(evaluate(retain))
    write_private_json(output / "controls.json", report)
    print(json.dumps({key: value for key, value in report.items() if key != "controls"}, indent=2))
    if report["passed"] != report["total"] or not report["source_unchanged"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
