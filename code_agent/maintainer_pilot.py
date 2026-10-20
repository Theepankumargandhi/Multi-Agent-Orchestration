"""Consented, local migration pilots with immutable evidence and explicit human outcomes.

No cloning, contacting maintainers, PR creation, pushes or automatic merges. Local
review/consent attestations are recorded as such, never authenticated GitHub facts.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator

from code_agent.benchmark_checkpoint import BenchmarkCheckpoint
from code_agent.patch_inputs import read_bounded, strict_json
from code_agent.patch_verify import (
    implementation_fingerprint,
    private_directory,
    protect_private_directory,
    tree_fingerprint,
)
from code_agent.regression_benchmark import confined
from code_agent.upgrade_cli import save_run
from code_agent.upgrade_inputs import CounterexamplePolicy
from code_agent.upgrade_models import MigrationSpec, StrictModel, fingerprint, safe_relative
from code_agent.upgrade_plan import plan_migration
from code_agent.upgrade_runner import resolve_image, run_migration


class PilotConsent(StrictModel):
    granted: bool = False
    maintainer_handle: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9-]{0,38}$")
    repository_url: str = Field(pattern=r"^https://github\.com/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")
    upstream_commit: str = Field(pattern=r"^[a-f0-9]{40,64}$")
    evidence_reference: str = Field(min_length=10, max_length=500)
    local_verification_only: Literal[True] = True
    publish_content_free_summary: bool = False
    identity_authenticated: Literal[False] = False


class PilotRequest(StrictModel):
    pilot_id: str = Field(pattern=r"^[A-Za-z0-9_.-]{1,80}$")
    evidence_kind: Literal["authored_control", "maintainer_pilot"] = "maintainer_pilot"
    repository: str
    repository_url: str = Field(pattern=r"^https://github\.com/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")
    upstream_commit: str = Field(pattern=r"^[a-f0-9]{40,64}$")
    license_file: str = "LICENSE"
    license_identifier: str = Field(min_length=1, max_length=100)
    consent: PilotConsent
    spec: MigrationSpec

    @model_validator(mode="after")
    def authorized(self):
        safe_relative(self.repository)
        safe_relative(self.license_file)
        if (not self.consent.granted or self.consent.repository_url != self.repository_url
                or self.consent.upstream_commit != self.upstream_commit or self.spec.task_id != self.pilot_id):
            raise ValueError("pilot needs matching explicit consent, revision and task identity")
        if self.evidence_kind == "maintainer_pilot" and not self.repository.startswith("repositories/"):
            raise ValueError("real pilot snapshots must be private under repositories/")
        return self


class PilotReview(StrictModel):
    pilot_id: str
    registration_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    report_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    patch_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    maintainer_handle: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9-]{0,38}$")
    decision: Literal["accepted", "changes_requested", "rejected", "merged"]
    evidence_url: str = Field(pattern=r"^https://github\.com/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+/(?:issues|pull)/[0-9]+(?:#[A-Za-z0-9_-]+)?$")
    merge_commit: str | None = Field(default=None, pattern=r"^[a-f0-9]{40,64}$")
    identity_authenticated: Literal[False] = False

    @model_validator(mode="after")
    def merged_evidence(self):
        if (self.decision == "merged") != bool(self.merge_commit) or (self.decision == "merged" and "/pull/" not in self.evidence_url):
            raise ValueError("reported merges require a PR and merge commit; other decisions cannot claim one")
        return self


def snapshot_binding(request: PilotRequest, root: Path):
    repository = confined(root, request.repository)
    license_data = read_bounded(confined(repository, request.license_file), 64_000)
    plan = plan_migration(repository, request.spec)
    if plan.status != "planned":
        raise ValueError("pilot profile cannot analyze this source scope")
    return {"source_sha256": tree_fingerprint(repository), "license_sha256": hashlib.sha256(license_data).hexdigest(),
            "plan_sha256": fingerprint(plan.model_dump()), "spec_sha256": fingerprint(request.spec.model_dump())}


def register_pilot(request: PilotRequest, root: Path, destination: Path, *, image_resolver=None):
    request = PilotRequest.model_validate(request.model_dump())
    destination = private_directory(destination, "maintainer-pilots")
    binding = snapshot_binding(request, root)
    image = (image_resolver or resolve_image)(request.spec.image)
    if not re.fullmatch(r"sha256:[a-f0-9]{64}", image):
        raise ValueError("pilot image is not content-addressed")
    registration = {"protocol": "agentforge-maintainer-pilot-v1", "request": request.model_dump(), "binding": binding,
                    "image_id": image, "registered_at": datetime.now(UTC).isoformat(), "upstream_origin_authenticated": False,
                    "pilot_implementation_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    "verification_implementation_sha256": implementation_fingerprint()}
    registration["fingerprint"] = fingerprint(registration)
    with BenchmarkCheckpoint(destination, resume=False) as journal:
        protect_private_directory(destination)
        journal.bind(registration["fingerprint"], request.pilot_id, initial_metadata={"registration": registration})
    return registration


def _open_pilot(journal, root):
    # Read the registered scope first, then let bind validate the envelope and all retained artifacts.
    envelope = strict_json(read_bounded(journal.safe_path("checkpoint.json"), 1_000_000))
    registration = envelope["state"]["registration"]
    if registration["fingerprint"] != fingerprint({key: value for key, value in registration.items() if key != "fingerprint"}):
        raise ValueError("pilot registration integrity changed")
    journal.bind(registration["fingerprint"], registration["request"]["pilot_id"])
    if journal.state.get("consent_revoked"):
        raise ValueError("maintainer consent is revoked")
    request = PilotRequest.model_validate(registration["request"])
    if snapshot_binding(request, root) != registration["binding"]:
        raise ValueError("pilot snapshot, license or owner contract changed")
    if registration["pilot_implementation_sha256"] != hashlib.sha256(Path(__file__).read_bytes()).hexdigest():
        raise ValueError("pilot implementation changed since registration")
    if registration["verification_implementation_sha256"] != implementation_fingerprint():
        raise ValueError("verification implementation changed since pilot registration")
    return registration, request


async def execute_pilot(directory: Path, root: Path, *, model=None, authorize_provider=False, runner=None, image_resolver=None):
    if bool(model) != authorize_provider:
        raise ValueError("live repair requires both explicit model and provider authorization")
    directory = private_directory(directory, "maintainer-pilots")
    with BenchmarkCheckpoint(directory, resume=True) as journal:
        registration, request = _open_pilot(journal, root)
        if journal.state.get("execution"):
            raise ValueError("pilot already attempted; no implicit retry of unknown or completed provider requests")
        if (image_resolver or resolve_image)(registration["image_id"]) != registration["image_id"]:
            raise ValueError("pilot image changed")
        journal.state["execution"] = {"status": "running", "provider_authorized": authorize_provider,
                                      "max_provider_calls": request.spec.max_model_calls if authorize_provider else 0}
        journal.save()  # Durable admission before any possible provider request.
        destination = journal.safe_path("execution")
        destination.mkdir(exist_ok=False)
        spec = MigrationSpec.model_validate({**request.spec.model_dump(), "image": registration["image_id"]})
        report = await (runner or run_migration)(confined(root, request.repository), spec, model=model,
                                                 authorize_provider=authorize_provider, counterexample_policy=CounterexamplePolicy())
        if (report.get("fingerprint") != fingerprint({key: value for key, value in report.items() if key != "fingerprint"})
                or report.get("task_id") != request.pilot_id or report.get("spec_fingerprint") != fingerprint(spec.model_dump())):
            raise ValueError("migration receipt does not match this pilot")
        if snapshot_binding(request, root) != registration["binding"] or not report["source_unchanged"]:
            raise ValueError("pilot source integrity changed during execution")
        if (report["status"] == "ready_for_owner_review" and (not report.get("patch")
                or report.get("patch_sha256") != hashlib.sha256(report["patch"].encode()).hexdigest())):
            raise ValueError("review-ready evidence must bind nonempty patch bytes")
        if not authorize_provider and report["live_provider_calls"] != 0:
            raise ValueError("unauthorized provider use cannot be accepted as pilot evidence")
        save_run(report, destination)
        artifacts = {"execution/report.json": {"sha256": hashlib.sha256((destination / "report.json").read_bytes()).hexdigest(),
                                                "bytes": (destination / "report.json").stat().st_size}}
        if report.get("patch"):
            artifacts["execution/migration.diff"] = {"sha256": hashlib.sha256((destination / "migration.diff").read_bytes()).hexdigest(),
                                                      "bytes": (destination / "migration.diff").stat().st_size}
        journal.state["execution"] = {"status": "complete", "report_sha256": report["fingerprint"],
            "patch_sha256": report.get("patch_sha256"), "ready_for_owner_review": report["status"] == "ready_for_owner_review",
            "live_provider_calls": report["live_provider_calls"], "duration_ms": report["duration_ms"]}
        journal.state["attempts"][request.pilot_id] = [{"status": "complete", "artifacts": artifacts}]
        journal.save()
    return report


def record_review(directory: Path, root: Path, review: PilotReview):
    review = PilotReview.model_validate(review.model_dump())
    with BenchmarkCheckpoint(private_directory(directory, "maintainer-pilots"), resume=True) as journal:
        registration, request = _open_pilot(journal, root)
        execution = journal.state.get("execution", {})
        if (journal.state.get("review") or execution.get("status") != "complete" or not execution.get("ready_for_owner_review")
                or review.pilot_id != request.pilot_id or review.registration_sha256 != registration["fingerprint"]
                or review.report_sha256 != execution["report_sha256"] or review.patch_sha256 != execution["patch_sha256"]
                or review.maintainer_handle.casefold() != request.consent.maintainer_handle.casefold()
                or not review.evidence_url.startswith(request.repository_url + "/")):
            raise ValueError("review must bind an unreviewed, ready patch and the consenting maintainer/repository")
        journal.state["review"] = {"attestation": review.model_dump(), "fingerprint": fingerprint(review.model_dump()),
                                   "recorded_at": datetime.now(UTC).isoformat(), "github_status_verified": False}
        journal.save()
        return journal.state["review"]


def revoke_consent(directory: Path):
    """Revocation remains possible even when source files have changed or disappeared."""
    with BenchmarkCheckpoint(private_directory(directory, "maintainer-pilots"), resume=True) as journal:
        envelope = strict_json(read_bounded(journal.safe_path("checkpoint.json"), 1_000_000))
        state = envelope["state"]
        journal.bind(state["scope"], state["run_id"])
        journal.state["consent_revoked"] = True
        journal.save()


def pilot_status(directory: Path, root: Path, *, public=False):
    with BenchmarkCheckpoint(private_directory(directory, "maintainer-pilots"), resume=True) as journal:
        registration, request = _open_pilot(journal, root)
        if public and not request.consent.publish_content_free_summary:
            raise ValueError("maintainer has not authorized public summary sharing")
        execution, review = journal.state.get("execution", {}), journal.state.get("review", {})
        decision = review.get("attestation", {}).get("decision")
        return {"protocol": "agentforge-pilot-summary-v1", "pilot_id": request.pilot_id,
                "evidence_kind": request.evidence_kind, "registration_sha256": registration["fingerprint"],
                "execution_status": execution.get("status", "not_run"),
                "ready_for_owner_review": execution.get("ready_for_owner_review", False),
                "maintainer_reported_decision": decision, "github_status_verified": False,
                "real_maintainer_reported_acceptance": request.evidence_kind == "maintainer_pilot" and decision in {"accepted", "merged"},
                "live_provider_calls": execution.get("live_provider_calls"), "duration_ms": execution.get("duration_ms"),
                "source_unchanged": True, "automatic_external_writes": 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("register", "run", "review", "status", "revoke"))
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--pilot-dir", type=Path, required=True)
    parser.add_argument("--input", type=Path)
    parser.add_argument("--model")
    parser.add_argument("--authorize-provider", action="store_true")
    parser.add_argument("--public", action="store_true")
    args = parser.parse_args()
    if (args.model or args.authorize_provider) and args.operation != "run":
        parser.error("only run accepts provider authorization")
    if args.public and args.operation != "status":
        parser.error("--public only applies to consented status summaries")
    if args.operation in {"register", "review"} and not args.input:
        parser.error("register/review require a private JSON input")
    if args.operation == "register":
        request = PilotRequest.model_validate(strict_json(read_bounded(args.input, 60_000)))
        result = register_pilot(request, args.root, args.pilot_dir)
        print(json.dumps({"registration_sha256": result["fingerprint"], "status": "registered_not_run"}))
    elif args.operation == "run":
        if args.authorize_provider:
            from dotenv import load_dotenv
            load_dotenv()
        result = asyncio.run(execute_pilot(args.pilot_dir, args.root, model=args.model, authorize_provider=args.authorize_provider))
        print(json.dumps({"status": result["status"], "reason": result["reason"], "live_provider_calls": result["live_provider_calls"]}))
        if result["status"] != "ready_for_owner_review":
            raise SystemExit(2)
    elif args.operation == "review":
        result = record_review(args.pilot_dir, args.root, PilotReview.model_validate(strict_json(read_bounded(args.input, 8000))))
        print(json.dumps({"review_sha256": result["fingerprint"], "github_status_verified": False}))
    elif args.operation == "revoke":
        revoke_consent(args.pilot_dir)
        print(json.dumps({"consent_revoked": True}))
    else:
        print(json.dumps(pilot_status(args.pilot_dir, args.root, public=args.public)))


if __name__ == "__main__":
    main()
