"""Public upstream patch replays, explicitly separate from independently reviewed benchmarks."""

from __future__ import annotations

import argparse
import asyncio
import difflib
import hashlib
import json
from pathlib import Path

from code_agent.patch_inputs import PatchVerifySpec, apply_file_patch, parse_patch, read_bounded, strict_json
from code_agent.patch_verify import (
    MODES,
    implementation_fingerprint,
    reproducer_spec,
    reserve_private_directory,
    verify_patch,
    write_private_json,
)
from code_agent.regression_benchmark import (
    RegressionCase,
    RegressionCorpus,
    case_inputs,
    confined,
    review_kit,
    summarize,
)
from code_agent.upgrade_models import fingerprint
from code_agent.upgrade_runner import resolve_image
from code_agent.upstream_inputs import GitSource, UpstreamCatalog


def runner_fingerprint():
    return fingerprint(
        {
            name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
            for name in ("upstream_pilot.py", "upstream_inputs.py", "regression_benchmark.py")
        }
    )


def _seal(value):
    return {**value, "fingerprint": fingerprint(value)}


def prepare(
    catalog: UpstreamCatalog, cache: Path, output: Path, *, allow_download=False, image_resolver=None
):
    """Materialize exact selected blobs and production-only commit diffs. No host imports."""
    catalog = UpstreamCatalog.model_validate(catalog.model_dump())
    output = reserve_private_directory(output, "regression-benchmark")
    sources, cases, bindings = {}, [], []
    images = {}
    for entry in catalog.cases:
        if entry.repository_url not in sources:
            sources[entry.repository_url] = GitSource(
                cache, entry.repository_url, allow_download=allow_download
            )
        source = sources[entry.repository_url]
        base_raw, candidate_raw = source.commit(entry.base_commit), source.commit(entry.candidate_commit)
        parents = [line[7:].decode() for line in candidate_raw.splitlines() if line.startswith(b"parent ")]
        if not parents or parents[0] != entry.base_commit:
            raise ValueError("pilot supports direct first-parent forward upstream commits only")
        snapshot = output / entry.case_id / "snapshot"
        snapshot.mkdir(parents=True)
        edits, file_receipts, total = [], [], 0
        for item in entry.files:
            raw = source.blob(entry.base_commit, item.upstream_path, item.sha256)
            total += len(raw)
            if total > 5_000_000:
                raise ValueError("upstream snapshot byte budget exceeded")
            destination = confined(snapshot, item.local_path)
            destination.parent.mkdir(parents=True, exist_ok=True)
            with destination.open("xb") as stream:
                stream.write(raw)
            file_receipts.append(item.model_dump())
            if item.candidate_sha256 is not None:
                candidate = source.blob(entry.candidate_commit, item.upstream_path, item.candidate_sha256)
                edits.extend(
                    difflib.unified_diff(
                        raw.decode().splitlines(keepends=True),
                        candidate.decode().splitlines(keepends=True),
                        fromfile=f"a/{item.local_path}",
                        tofile=f"b/{item.local_path}",
                    )
                )
        for name, text in entry.support_files.items():
            path = confined(snapshot, name)
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("x", encoding="utf-8", newline="") as stream:
                stream.write(text)
        changed = (
            source.read(
                "diff",
                "--name-only",
                "--no-renames",
                entry.base_commit,
                entry.candidate_commit,
                "--",
                limit=100_000,
            )
            .decode()
            .splitlines()
        )
        production = {row.upstream_path for row in entry.files if row.candidate_sha256 is not None}
        provenance = {
            "protocol": "upstream-production-projection-v1",
            "repository_url": entry.repository_url,
            "base_commit": entry.base_commit,
            "candidate_commit": entry.candidate_commit,
            "commit_bytes_sha256": {
                "base": hashlib.sha256(base_raw).hexdigest(),
                "candidate": hashlib.sha256(candidate_raw).hexdigest(),
            },
            "files": file_receipts,
            "authored_build_metadata": entry.support_files,
            "excluded_commit_paths": [path for path in changed if path not in production],
            "projection": "production_only_with_explicit_src_layout_mapping",
            "test_snapshot": "unchanged_selected_upstream_base_files_not_full_repository_suite",
            "preservation_scope_review": "local_curator_assumption_not_upstream_owner_approval",
            "candidate_label": "public_evidence_derived_not_independently_artifact_reviewed",
            "evidence_url": entry.evidence_url,
            "catalog_sha256": fingerprint(catalog.model_dump()),
        }
        write_private_json(snapshot / "UPSTREAM-PROVENANCE.json", _seal(provenance))
        patch = "".join(edits)
        for edit in parse_patch(patch, entry.spec):
            apply_file_patch(read_bounded(snapshot / edit.path, 1_000_000).decode(), edit)
        prefix = entry.case_id
        case = RegressionCase(
            case_id=prefix,
            family=entry.family,
            split="development",
            evidence_kind="historical_real_patch",
            repository=f"{prefix}/snapshot",
            patch=f"{prefix}/candidate.diff",
            spec=f"{prefix}/spec.json",
            expected=entry.expected,
            license_file=entry.license_file,
            license_identifier=entry.license_identifier,
            upstream_url=entry.repository_url,
            upstream_commit=entry.base_commit,
            patch_origin="upstream_commit",
        )
        with (output / case.patch).open("x", encoding="utf-8", newline="") as stream:
            stream.write(patch)
        write_private_json(output / case.spec, entry.spec.model_dump())
        write_private_json(output / prefix / "intake.json", case.model_dump())
        write_private_json(
            output / prefix / "review-kit.json", review_kit(case, output, "automated-public-curation")
        )
        _, spec, _, artifacts = case_inputs(output, case)
        if spec.image not in images:
            image = (image_resolver or resolve_image)(spec.image)
            # Reuse spec's strict image grammar; immutable identity is checked here.
            if (
                not image.startswith("sha256:")
                or len(image) != 71
                or any(c not in "0123456789abcdef" for c in image[7:])
            ):
                raise ValueError("pilot requires an existing content-addressed image")
            images[spec.image] = image
        cases.append(case)
        bindings.append(
            {
                "case_id": prefix,
                "artifacts": artifacts,
                "image_id": images[spec.image],
                "expectations": entry.expectations,
            }
        )
    corpus = RegressionCorpus(corpus_id=catalog.catalog_id, cases=cases)
    study = _seal(
        {
            "protocol": "agentforge-upstream-freeze-v1",
            "corpus": corpus.model_dump(),
            "catalog": catalog.model_dump(),
            "bindings": bindings,
            "implementation_sha256": implementation_fingerprint(),
            "runner_sha256": runner_fingerprint(),
            "independently_reviewed_real_cases": 0,
            "benchmark_admitted": False,
            "split": "public_development_no_hidden_holdout",
        }
    )
    write_private_json(output / "study.json", study)
    return study


def validate_study(study, root, image_resolver=None):
    if (
        study.get("fingerprint") != fingerprint({k: v for k, v in study.items() if k != "fingerprint"})
        or study.get("protocol") != "agentforge-upstream-freeze-v1"
        or study.get("implementation_sha256") != implementation_fingerprint()
        or study.get("runner_sha256") != runner_fingerprint()
        or study.get("benchmark_admitted") is not False
        or study.get("independently_reviewed_real_cases") != 0
    ):
        raise ValueError("upstream study integrity, implementation or review status changed")
    corpus = RegressionCorpus.model_validate(study["corpus"])
    catalog = UpstreamCatalog.model_validate(study["catalog"])
    if (
        len(corpus.cases) != len(study["bindings"])
        or len(corpus.cases) != len(catalog.cases)
        or corpus.corpus_id != catalog.catalog_id
    ):
        raise ValueError("upstream study case population changed")
    for case, entry, binding in zip(corpus.cases, catalog.cases, study["bindings"], strict=True):
        if (
            case.case_id != entry.case_id
            or case.case_id != binding["case_id"]
            or case.review is not None
            or case.family != entry.family
            or case.expected != entry.expected
            or case.split != "development"
            or case.patch_origin != "upstream_commit"
            or case.evidence_kind != "historical_real_patch"
            or case.upstream_url != entry.repository_url
            or case.upstream_commit != entry.base_commit
            or case.repository != f"{entry.case_id}/snapshot"
            or case.patch != f"{entry.case_id}/candidate.diff"
            or case.spec != f"{entry.case_id}/spec.json"
            or case.license_file != entry.license_file
            or case.license_identifier != entry.license_identifier
            or binding["expectations"] != entry.expectations
        ):
            raise ValueError("upstream study labels, reviews or case bindings changed")
        _, spec, _, artifacts = case_inputs(root, case)
        if artifacts != binding["artifacts"] or spec != entry.spec:
            raise ValueError("upstream inputs changed since preparation")
        if (image_resolver or resolve_image)(binding["image_id"]) != binding["image_id"]:
            raise ValueError("upstream execution image is unavailable or changed")
    return corpus


def observation(report, spec, patch, binding, mode):
    if (
        report.get("fingerprint") != fingerprint({k: v for k, v in report.items() if k != "fingerprint"})
        or report.get("spec_sha256") != fingerprint(spec.model_dump())
        or report.get("patch_sha256") != hashlib.sha256(patch.encode()).hexdigest()
        or report.get("source_sha256") != binding["artifacts"]["source_sha256"]
        or report.get("implementation_sha256") != implementation_fingerprint()
        or (
            report.get("image_id") != binding["image_id"]
            and not (
                report.get("status") == "inconclusive"
                and not report.get("trials")
                and report.get("image_id") == ""
            )
        )
        or report.get("mode") != mode
        or report.get("live_provider_calls") != 0
        or report.get("provider_cost_usd") != 0
    ):
        raise ValueError("upstream verifier receipt is not bound to the frozen execution")
    green = (
        len(report["trials"]) == 4
        and report["source_unchanged"]
        and report["reason"] not in {"unstable_execution", "test_population_changed"}
        and all(row["tests"]["passed"] for row in report["trials"])
    )
    decision = "inconclusive"
    if green:
        if mode == "tests_only" or report["status"] == "checks_passed":
            decision = "no_alarm"
        elif report["status"] == "regression_found":
            decision = "alarm"
    return decision, green


async def run(study, root: Path, output: Path, *, verifier=None, image_resolver=None, on_result=None):
    corpus = validate_study(study, root, image_resolver)
    output = reserve_private_directory(output, "regression-benchmark")
    rows, replays = [], []
    for case, binding in zip(corpus.cases, study["bindings"], strict=True):
        repository, owner, patch, _ = case_inputs(root, case)
        spec = PatchVerifySpec.model_validate({**owner.model_dump(), "image": binding["image_id"]})
        for mode in MODES:
            report = await (verifier or verify_patch)(repository, patch, spec, mode=mode)
            decision, green = observation(report, spec, patch, binding, mode)
            row = {
                "case_id": case.case_id,
                "family": case.family,
                "split": "development",
                "evidence_kind": "historical_real_patch",
                "expected": case.expected,
                "mode": mode,
                "decision": decision,
                "tests_green_eligible": green,
                "report": report,
                "duration_ms": report["duration_ms"],
                "provider_cost_usd": report["provider_cost_usd"],
                "expectation_met": decision == binding["expectations"][mode],
            }
            rows.append(row)
            write_private_json(output / f"{case.case_id}-{mode}.json", row)
            if on_result:
                on_result({key: row[key] for key in ("case_id", "mode", "decision", "expectation_met")})
            replay = reproducer_spec(spec, report)
            if mode == "generated" and replay is not None:
                write_private_json(output / f"{case.case_id}-reproducer.spec.json", replay.model_dump())
                repeated = await (verifier or verify_patch)(repository, patch, replay, mode="contracts")
                repeated_decision, repeated_green = observation(repeated, replay, patch, binding, "contracts")
                result = {
                    "case_id": case.case_id,
                    "confirmed": repeated_decision == "alarm" and repeated_green,
                    "report": repeated,
                }
                replays.append(result)
                write_private_json(output / f"{case.case_id}-replay.json", result)
        if case_inputs(root, case)[3] != binding["artifacts"]:
            raise ValueError("upstream inputs changed during execution")
    validate_study(study, root, image_resolver)
    report = _seal(
        {
            "protocol": "agentforge-upstream-results-v1",
            "study_sha256": study["fingerprint"],
            "evidence_kind": "public_upstream_development_replay_unreviewed",
            "rows": rows,
            "replays": replays,
            "provisional_metrics": summarize(rows)["historical_real_patch"],
            "expectations_met": sum(row["expectation_met"] for row in rows),
            "expected_observations": len(rows),
            "benchmark_admitted": False,
            "independently_reviewed_real_cases": 0,
            "official_benchmark_result": False,
            "maintainer_acceptance_measured": False,
            "live_provider_calls": 0,
        }
    )
    write_private_json(output / "results.json", report)
    public = {
        key: report[key]
        for key in (
            "protocol",
            "evidence_kind",
            "study_sha256",
            "expectations_met",
            "expected_observations",
            "benchmark_admitted",
            "live_provider_calls",
        )
    }
    public.update(
        replay_confirmations=sum(row["confirmed"] for row in replays),
        provisional_metrics=report["provisional_metrics"],
        results_sha256=report["fingerprint"],
    )
    write_private_json(output / "public-summary.json", public)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("prepare", "run"))
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cache", type=Path, default=Path("data/regression-benchmark/upstream-cache"))
    parser.add_argument("--allow-download", action="store_true")
    args = parser.parse_args()
    payload = strict_json(read_bounded(args.input, 2_000_000))
    if args.operation == "prepare":
        study = prepare(
            UpstreamCatalog.model_validate(payload),
            args.cache,
            args.output_dir,
            allow_download=args.allow_download,
        )
        print(
            json.dumps(
                {
                    "status": "prepared_unreviewed_public_replays",
                    "cases": len(study["bindings"]),
                    "study_sha256": study["fingerprint"],
                    "benchmark_admitted": False,
                }
            )
        )
    else:
        if args.allow_download:
            parser.error("execution never downloads source; prepare it separately")
        result = asyncio.run(
            run(
                payload,
                args.input.parent,
                args.output_dir,
                on_result=lambda row: print(json.dumps(row), flush=True),
            )
        )
        print(
            json.dumps(
                {
                    "expectations_met": result["expectations_met"],
                    "expected_observations": result["expected_observations"],
                    "replays_confirmed": sum(row["confirmed"] for row in result["replays"]),
                    "benchmark_admitted": False,
                    "live_provider_calls": 0,
                }
            )
        )
        if result["expectations_met"] != result["expected_observations"] or any(
            not row["confirmed"] for row in result["replays"]
        ):
            raise SystemExit(2)


if __name__ == "__main__":
    main()
