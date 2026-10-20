"""Frozen patch-regression comparisons; authored controls never become real-world evidence."""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import re
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator

from code_agent.patch_inputs import MAX_PATCH_BYTES, PatchVerifySpec, read_bounded, strict_json
from code_agent.patch_verify import (
    MODES,
    implementation_fingerprint,
    reserve_private_directory,
    tree_fingerprint,
    verify_patch,
    write_private_json,
)
from code_agent.upgrade_models import StrictModel, fingerprint, safe_relative
from code_agent.upgrade_runner import resolve_image


class IndependentReview(StrictModel):
    submitter_id: str = Field(min_length=1, max_length=120)
    reviewer_id: str = Field(min_length=1, max_length=120)
    evidence_url: str = Field(pattern=r"^https://[^\s]{5,500}$")
    artifacts_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    conclusion: Literal["regression", "correct"]
    # A local attestation is deliberately not represented as authenticated identity.
    identity_authenticated: Literal[False] = False

    @model_validator(mode="after")
    def independent(self):
        if self.submitter_id.strip().casefold() == self.reviewer_id.strip().casefold():
            raise ValueError("real cases require a different reviewer and submitter")
        return self


class RegressionCase(StrictModel):
    case_id: str = Field(pattern=r"^[A-Za-z0-9_.-]{1,80}$")
    family: str = Field(pattern=r"^[A-Za-z0-9_.-]{1,100}$")
    split: Literal["development", "holdout"]
    evidence_kind: Literal["authored_control", "historical_real_patch"]
    repository: str
    patch: str
    spec: str
    expected: Literal["regression", "correct"]
    license_file: str = "LICENSE"
    license_identifier: str = Field(min_length=1, max_length=100)
    upstream_url: str | None = Field(default=None, pattern=r"^https://github\.com/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")
    upstream_commit: str | None = Field(default=None, pattern=r"^[a-f0-9]{40,64}$")
    patch_origin: Literal["authored", "historical_reversal", "model_generated", "upstream_commit"] = "authored"
    review: IndependentReview | None = None

    @model_validator(mode="after")
    def paths(self):
        for value in (self.repository, self.patch, self.spec, self.license_file):
            safe_relative(value)
        if self.evidence_kind == "historical_real_patch":
            if not self.upstream_url or not self.upstream_commit or self.patch_origin == "authored":
                raise ValueError("real-case intake needs pinned upstream/license provenance")
            if self.review and self.review.conclusion != self.expected:
                raise ValueError("review conclusion differs from the case label")
        elif self.review is not None or self.patch_origin != "authored":
            raise ValueError("authored controls must not claim independent real-patch review")
        return self


class RegressionCorpus(StrictModel):
    protocol: Literal["agentforge-regression-corpus-v1"] = "agentforge-regression-corpus-v1"
    corpus_id: str = Field(pattern=r"^[A-Za-z0-9_.-]{1,80}$")
    cases: list[RegressionCase] = Field(min_length=1, max_length=100)

    @model_validator(mode="after")
    def isolated_families(self):
        if len({row.case_id for row in self.cases}) != len(self.cases):
            raise ValueError("case IDs must be unique")
        family_splits = {}
        for row in self.cases:
            if row.family in family_splits and family_splits[row.family] != row.split:
                raise ValueError("task families cannot cross development/holdout splits")
            family_splits[row.family] = row.split
        return self


def confined(root: Path, name: str) -> Path:
    path = root / safe_relative(name)
    # Reject links before resolving, including directory junctions on Windows.
    for item in (path, *path.parents):
        if item.is_symlink() or (item.exists() and getattr(item.lstat(), "st_file_attributes", 0) & 0x400):
            raise ValueError("benchmark inputs cannot traverse links")
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError("benchmark path leaves its input root")
    return path


def case_inputs(root: Path, case: RegressionCase):
    repository = confined(root, case.repository)
    spec = PatchVerifySpec.model_validate(strict_json(read_bounded(confined(root, case.spec), 40_000)))
    patch = read_bounded(confined(root, case.patch), MAX_PATCH_BYTES).decode("utf-8")
    license_hash = hashlib.sha256(read_bounded(confined(repository, case.license_file), 64_000)).hexdigest()
    if (spec.task_id != case.case_id or not spec.preservation_authorized or not spec.contracts
            or not (spec.integer_domains or spec.json_domains)):
        raise ValueError("comparison requires matching task identity and owner-authorized contracts/domains")
    artifacts = {"source_sha256": tree_fingerprint(repository), "patch_sha256": hashlib.sha256(patch.encode()).hexdigest(),
                 "spec_sha256": fingerprint(spec.model_dump()), "license_sha256": license_hash}
    if case.review and case.review.artifacts_sha256 != fingerprint(artifacts):
        raise ValueError("independent review is not bound to these exact artifacts")
    return repository, spec, patch, artifacts


def freeze_corpus(corpus: RegressionCorpus, root: Path, *, minimum_real_cases=0, image_resolver=None):
    if not 0 <= minimum_real_cases <= 100:
        raise ValueError("minimum real-case count is invalid")
    corpus = RegressionCorpus.model_validate(corpus.model_dump())
    real = [row for row in corpus.cases if row.evidence_kind == "historical_real_patch"]
    if any(row.review is None for row in real):
        raise ValueError("real cases cannot be admitted without independent artifact-bound review")
    if len(real) < minimum_real_cases:
        raise ValueError("not enough independently reviewed real cases; authored controls do not count")
    if real and {row.expected for row in real} != {"regression", "correct"}:
        raise ValueError("real comparisons require both regressions and correct-patch negative controls")
    rows = []
    images = {}
    for case in corpus.cases:
        _, spec, _, artifacts = case_inputs(root, case)
        if spec.image not in images:
            images[spec.image] = (image_resolver or resolve_image)(spec.image)
            if not re.fullmatch(r"sha256:[a-f0-9]{64}", images[spec.image]):
                raise ValueError("benchmark requires content-addressed local images")
        rows.append({"case_id": case.case_id, "artifacts": artifacts, "image_id": images[spec.image]})
    frozen = {"protocol": "agentforge-regression-freeze-v1", "corpus": corpus.model_dump(), "bindings": rows,
              "implementation_sha256": implementation_fingerprint(),
              "benchmark_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "minimum_real_cases": minimum_real_cases, "independently_reviewed_real_cases": len(real),
              "official_benchmark_result": False, "upstream_origin_authenticated": False}
    frozen["fingerprint"] = fingerprint(frozen)
    return frozen


def review_kit(case: RegressionCase, root: Path, submitter_id: str):
    """Prepare exact review bindings without fabricating a reviewer or accepting a label."""
    if case.evidence_kind != "historical_real_patch" or case.review is not None or not submitter_id.strip():
        raise ValueError("review kits require an unreviewed real-case intake and a submitter ID")
    _, _, _, artifacts = case_inputs(root, case)
    return {"status": "awaiting_independent_review_not_admitted", "case": case.model_dump(), "artifacts": artifacts,
            "artifacts_sha256": fingerprint(artifacts), "label_status": "submitter_proposed_not_independently_confirmed",
            "review_template": {"submitter_id": submitter_id, "reviewer_id": "", "evidence_url": "",
                                "artifacts_sha256": fingerprint(artifacts), "conclusion": case.expected,
                                "identity_authenticated": False}, "upstream_origin_authenticated": False}


def _rate(numerator, denominator):
    return numerator / denominator if denominator else None


def summarize(rows):
    """Every declared task remains in denominators; inconclusives are not successes."""
    result = {}
    for kind in ("authored_control", "historical_real_patch"):
        result[kind] = {}
        for mode in MODES:
            selected = [row for row in rows if row["evidence_kind"] == kind and row["mode"] == mode]
            positives = [row for row in selected if row["expected"] == "regression"]
            negatives = [row for row in selected if row["expected"] == "correct"]
            detected = sum(row["decision"] == "alarm" for row in positives)
            false_alarms = sum(row["decision"] == "alarm" for row in negatives)
            unknown_negatives = sum(row["decision"] == "inconclusive" for row in negatives)
            measured_cost = sum(row["provider_cost_usd"] for row in selected)
            result[kind][mode] = {"declared_cases": len(selected), "regression_cases": len(positives),
                "correct_patch_controls": len(negatives), "detected_regressions": detected,
                "regression_detection_rate": _rate(detected, len(positives)), "false_alarms": false_alarms,
                "false_alarm_rate": _rate(false_alarms, len(negatives)),
                "false_alarm_rate_bounds": [_rate(false_alarms, len(negatives)), _rate(false_alarms + unknown_negatives, len(negatives))],
                "decision_coverage": _rate(sum(row["decision"] != "inconclusive" for row in selected), len(selected)),
                "correct_negative_clearance_rate": _rate(sum(row["decision"] == "no_alarm" for row in negatives), len(negatives)),
                "inconclusive_cases": sum(row["decision"] == "inconclusive" for row in selected),
                "tests_green_eligible_cases": sum(row["tests_green_eligible"] for row in selected),
                "mean_latency_ms": _rate(sum(row["duration_ms"] for row in selected), len(selected)),
                "provider_cost_usd": measured_cost, "provider_cost_per_detected_regression_usd": _rate(measured_cost, detected),
                "infrastructure_cost_usd": None, "live_provider_calls": 0}
    return result


async def run_benchmark(frozen, root: Path, *, verifier=None, image_resolver=None):
    if frozen.get("fingerprint") != fingerprint({key: value for key, value in frozen.items() if key != "fingerprint"}):
        raise ValueError("benchmark freeze integrity changed")
    corpus = RegressionCorpus.model_validate(frozen["corpus"])
    current = freeze_corpus(corpus, root, minimum_real_cases=frozen["minimum_real_cases"], image_resolver=image_resolver)
    if current != frozen:
        raise ValueError("benchmark sources, runtime or reviews changed since freeze")
    rows = []
    for case, binding in zip(corpus.cases, frozen["bindings"], strict=True):
        repository, spec, patch, _ = case_inputs(root, case)
        spec = PatchVerifySpec.model_validate({**spec.model_dump(), "image": binding["image_id"]})
        for mode in MODES:
            report = await (verifier or verify_patch)(repository, patch, spec, mode=mode)
            if report.get("fingerprint") != fingerprint({key: value for key, value in report.items() if key != "fingerprint"}):
                raise ValueError("verifier returned an invalid receipt fingerprint")
            tests_green = (len(report["trials"]) == 4 and report["source_unchanged"]
                           and report["reason"] not in {"unstable_execution", "test_population_changed"}
                           and all(row["tests"]["passed"] for row in report["trials"]))
            decision = "inconclusive"
            if tests_green:
                if mode == "tests_only":
                    decision = "no_alarm"
                elif report["status"] == "regression_found":
                    decision = "alarm"
                elif report["status"] == "checks_passed":
                    decision = "no_alarm"
            rows.append({"case_id": case.case_id, "family": case.family, "split": case.split,
                         "evidence_kind": case.evidence_kind, "expected": case.expected, "mode": mode,
                         "decision": decision, "tests_green_eligible": tests_green, "report": report,
                         "duration_ms": report["duration_ms"], "provider_cost_usd": report["provider_cost_usd"]})
        # Drift invalidates the whole comparison, not merely the final arm.
        if case_inputs(root, case)[3] != binding["artifacts"]:
            raise ValueError("case artifacts changed during benchmark execution")
    if freeze_corpus(corpus, root, minimum_real_cases=frozen["minimum_real_cases"], image_resolver=image_resolver) != frozen:
        raise ValueError("benchmark runtime, images or corpus changed during execution")
    report = {"protocol": "agentforge-regression-results-v1", "freeze_sha256": frozen["fingerprint"],
              "rows": rows, "metrics": summarize(rows),
              "metrics_by_split": {split: summarize([row for row in rows if row["split"] == split]) for split in ("development", "holdout")},
              "independently_reviewed_real_cases": frozen["independently_reviewed_real_cases"],
              "official_benchmark_result": False, "maintainer_acceptance_measured": False,
              "live_provider_calls": 0, "claims": "Local verification comparison; no AI-generated-test superiority or maintainer adoption claim."}
    report["fingerprint"] = fingerprint(report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("review-kit", "freeze", "run"))
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--minimum-real-cases", type=int, default=0)
    parser.add_argument("--submitter-id")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    payload = strict_json(read_bounded(args.input, 2_000_000))
    if args.operation != "freeze" and args.minimum_real_cases:
        parser.error("minimum real-case count is fixed at freeze time")
    if (args.operation == "review-kit") != bool(args.submitter_id):
        parser.error("review-kit requires --submitter-id; other operations do not accept it")
    output = reserve_private_directory(args.output_dir, "regression-benchmark")
    if args.operation == "review-kit":
        result = review_kit(RegressionCase.model_validate(payload), args.root, args.submitter_id)
        write_private_json(output / "review-kit.json", result)
        print(json.dumps({"status": result["status"], "artifacts_sha256": result["artifacts_sha256"], "private_artifacts": str(output)}))
    elif args.operation == "freeze":
        result = freeze_corpus(RegressionCorpus.model_validate(payload), args.root, minimum_real_cases=args.minimum_real_cases)
        write_private_json(output / "freeze.json", result)
        print(json.dumps({"freeze_sha256": result["fingerprint"], "real_cases": result["independently_reviewed_real_cases"], "private_artifacts": str(output)}))
    else:
        result = asyncio.run(run_benchmark(payload, args.root))
        write_private_json(output / "results.json", result)
        print(json.dumps({"metrics": result["metrics"], "private_artifacts": str(output)}))
        if any(row["decision"] == "inconclusive" for row in result["rows"]):
            raise SystemExit(2)


if __name__ == "__main__":
    main()
