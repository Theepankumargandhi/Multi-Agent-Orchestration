"""Authored pilot lifecycle control with real migration Docker execution, not maintainer adoption."""
from __future__ import annotations

import argparse
import asyncio
import json
import shutil
from pathlib import Path

from code_agent.maintainer_pilot import (
    PilotRequest,
    PilotReview,
    execute_pilot,
    pilot_status,
    record_review,
    register_pilot,
    revoke_consent,
)
from code_agent.patch_inputs import read_bounded, strict_json
from code_agent.patch_verify import reserve_private_directory, write_private_json

ROOT = Path(__file__).resolve().parent.parent


async def evaluate(output: Path):
    repository = output / "repositories/authored-control"
    shutil.copytree(ROOT / "tests/fixtures/upgradepilot/constraints", repository)
    shutil.copy2(ROOT / "LICENSE", repository / "LICENSE")
    spec = strict_json(read_bounded(repository / "migration.json", 40_000))
    spec["task_id"] = "migration-pilot-control"
    request = PilotRequest.model_validate({"pilot_id": spec["task_id"], "evidence_kind": "authored_control",
        "repository": "repositories/authored-control", "repository_url": "https://github.com/example/authored-control",
        "upstream_commit": "0" * 40, "license_identifier": "MIT", "spec": spec,
        "consent": {"granted": True, "maintainer_handle": "AuthoredControl", "repository_url": "https://github.com/example/authored-control",
                    "upstream_commit": "0" * 40, "evidence_reference": "Synthetic consent for an authored fixture; no maintainer contacted."}})
    pilot = output / "pilot"
    registration = register_pilot(request, output, pilot)
    report = await execute_pilot(pilot, output)
    checks = [{"name": "real_docker_migration_ready", "passed": report["status"] == "ready_for_owner_review" and report["source_unchanged"]},
              {"name": "zero_provider_calls", "passed": report["live_provider_calls"] == 0}]
    try:
        await execute_pilot(pilot, output)
    except ValueError:
        checks.append({"name": "implicit_rerun_blocked", "passed": True})
    else:
        checks.append({"name": "implicit_rerun_blocked", "passed": False})
    if report["status"] == "ready_for_owner_review":
        review = PilotReview(pilot_id=request.pilot_id, registration_sha256=registration["fingerprint"],
            report_sha256=report["fingerprint"], patch_sha256=report["patch_sha256"], maintainer_handle="AuthoredControl",
            decision="accepted", evidence_url="https://github.com/example/authored-control/pull/1")
        record_review(pilot, output, review)
        status = pilot_status(pilot, output)
        checks.append({"name": "synthetic_review_not_real_acceptance", "passed": not status["real_maintainer_reported_acceptance"] and not status["github_status_verified"]})
    else:
        checks.append({"name": "synthetic_review_not_real_acceptance", "passed": False})
    try:
        pilot_status(pilot, output, public=True)
    except ValueError:
        checks.append({"name": "unconsented_public_summary_blocked", "passed": True})
    else:
        checks.append({"name": "unconsented_public_summary_blocked", "passed": False})
    revoke_consent(pilot)
    try:
        pilot_status(pilot, output)
    except ValueError:
        checks.append({"name": "withdrawn_consent_blocks_use", "passed": True})
    else:
        checks.append({"name": "withdrawn_consent_blocks_use", "passed": False})
    return {"evidence_kind": "authored_pilot_control_real_docker", "total": len(checks), "passed": sum(row["passed"] for row in checks),
            "checks": checks, "migration_status": report["status"], "live_provider_calls": report["live_provider_calls"],
            "real_maintainer_acceptances": 0, "real_maintainers_contacted": 0, "external_writes": 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output = reserve_private_directory(args.output_dir, "maintainer-pilots")
    report = asyncio.run(evaluate(output))
    write_private_json(output / "controls.json", report)
    print(json.dumps({key: value for key, value in report.items() if key != "checks"}, indent=2))
    if report["passed"] != report["total"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
