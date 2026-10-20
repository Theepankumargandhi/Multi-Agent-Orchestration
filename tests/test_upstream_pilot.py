"""Authored unit controls for provenance and orchestration, not real-patch results."""

import hashlib
import json
from pathlib import Path

import pytest

from code_agent import upstream_pilot as pilot
from code_agent.patch_verify import tree_fingerprint
from code_agent.regression_benchmark import RegressionCorpus, freeze_corpus
from code_agent.upgrade_models import fingerprint
from code_agent.upstream_inputs import GitSource, UpstreamCatalog

IMAGE = "sha256:" + "a" * 64
BASE, CANDIDATE = "1" * 40, "2" * 40
ORIGINAL = b"def change(value):\n    return value\n"
CHANGED = b"def change(value):\n    return value + 1\n"
FILES = {
    "app.py": ORIGINAL,
    "checks.py": b"def test_green():\n    assert True\n",
    "LICENSE": b"Authored unit fixture\r\n",
}


def catalog():
    cases = []
    for identity, label in (("positive", "regression"), ("negative", "correct")):
        files = [
            {
                "upstream_path": name,
                "local_path": name,
                "sha256": hashlib.sha256(raw).hexdigest(),
                **({"candidate_sha256": hashlib.sha256(CHANGED).hexdigest()} if name == "app.py" else {}),
            }
            for name, raw in FILES.items()
        ]
        cases.append(
            {
                "case_id": identity,
                "family": "unit-family",
                "repository_url": "https://github.com/example/unit",
                "base_commit": BASE,
                "candidate_commit": CANDIDATE,
                "expected": label,
                "evidence_url": "https://github.com/example/unit/commit/" + CANDIDATE,
                "files": files,
                "license_file": "LICENSE",
                "license_identifier": "MIT",
                "spec": {
                    "task_id": identity,
                    "source_paths": ["app.py"],
                    "test_paths": ["checks.py"],
                    "preservation_authorized": True,
                    "contracts": [{"case_id": "ordinary", "entrypoint": "app:change", "args": [1]}],
                    "json_domains": [
                        {
                            "domain_id": "values",
                            "entrypoint": "app:change",
                            "input_schema": {"type": "integer", "minimum": 0, "maximum": 3},
                        }
                    ],
                },
                "expectations": {
                    "tests_only": "no_alarm",
                    "contracts": "no_alarm",
                    "generated": "alarm" if label == "regression" else "no_alarm",
                },
            }
        )
    return UpstreamCatalog(catalog_id="authored-unit-only", cases=cases)


class DataOnlySource:
    def __init__(self, *args, **kwargs):
        pass

    def commit(self, identity):
        return (
            b"tree "
            + b"0" * 40
            + b"\n"
            + (b"parent " + BASE.encode() + b"\n" if identity == CANDIDATE else b"")
        )

    def blob(self, commit, name, expected):
        raw = CHANGED if commit == CANDIDATE and name == "app.py" else FILES[name]
        assert hashlib.sha256(raw).hexdigest() == expected
        return raw

    def read(self, *args, **kwargs):
        return b"app.py\nnew_upstream_tests.py\n"


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(pilot, "GitSource", DataOnlySource)
    root = tmp_path / "data/regression-benchmark/prepared"
    study = pilot.prepare(catalog(), tmp_path / "cache", root, image_resolver=lambda _: IMAGE)
    return root, study


def test_prepare_keeps_exact_license_and_test_bytes_and_never_fabricates_review(prepared):
    root, study = prepared
    assert pilot.validate_study(study, root, lambda _: IMAGE)
    assert not study["benchmark_admitted"] and study["independently_reviewed_real_cases"] == 0
    assert (root / "positive/snapshot/LICENSE").read_bytes() == FILES["LICENSE"]
    assert (root / "positive/snapshot/checks.py").read_bytes() == FILES["checks.py"]
    provenance = json.loads((root / "positive/snapshot/UPSTREAM-PROVENANCE.json").read_bytes())
    assert provenance["excluded_commit_paths"] == ["new_upstream_tests.py"]
    kit = json.loads((root / "positive/review-kit.json").read_bytes())
    assert kit["review_template"]["reviewer_id"] == "" and not kit["upstream_origin_authenticated"]
    with pytest.raises(ValueError, match="without independent"):
        freeze_corpus(RegressionCorpus.model_validate(study["corpus"]), root, image_resolver=lambda _: IMAGE)


@pytest.mark.parametrize("target", ["source", "license", "spec", "patch"])
def test_every_frozen_artifact_is_checked_before_execution(prepared, target):
    root, study = prepared
    path = {
        "source": "snapshot/app.py",
        "license": "snapshot/LICENSE",
        "spec": "spec.json",
        "patch": "candidate.diff",
    }[target]
    with (root / "positive" / path).open("a") as stream:
        stream.write("\nchanged\n")
    with pytest.raises(ValueError):
        pilot.validate_study(study, root, lambda _: IMAGE)


@pytest.mark.parametrize(
    "field,value",
    [
        ("upstream_commit", "f" * 40),
        ("expected", "correct"),
        ("split", "holdout"),
        ("repository", "negative/snapshot"),
    ],
)
def test_resealed_case_metadata_cannot_disagree_with_the_catalog(prepared, field, value):
    root, study = prepared
    study["corpus"]["cases"][0][field] = value
    study["fingerprint"] = fingerprint({k: v for k, v in study.items() if k != "fingerprint"})
    with pytest.raises(ValueError):
        pilot.validate_study(study, root, lambda _: IMAGE)


@pytest.mark.asyncio
async def test_matched_modes_and_independent_replay_are_checkpointed_without_public_inputs(prepared):
    root, study = prepared
    calls = []

    async def controlled(repository, patch, spec, *, mode):
        calls.append((spec.task_id, mode, bool(spec.reference_patch_sha256)))
        alarm = spec.task_id == "positive" and (mode == "generated" or spec.reference_patch_sha256)
        case = spec.json_domains[0].case(1, "generated.values.0").model_dump() if spec.json_domains else None
        report = {
            "status": "regression_found" if alarm else "checks_passed",
            "reason": "owner_preservation_violation" if alarm else "declared_checks_passed_not_a_proof",
            "trials": [{"tests": {"passed": True}}] * 4,
            "source_unchanged": True,
            "source_sha256": tree_fingerprint(repository),
            "spec_sha256": fingerprint(spec.model_dump()),
            "patch_sha256": hashlib.sha256(patch.encode()).hexdigest(),
            "mode": mode,
            "image_id": IMAGE,
            "implementation_sha256": pilot.implementation_fingerprint(),
            "live_provider_calls": 0,
            "provider_cost_usd": 0,
            "duration_ms": 1,
            "counterexamples": [case] if alarm and case else [],
            "minimizations": [{"confirmed_case": case}] if alarm and case else [],
        }
        return pilot._seal(report)

    output = Path("data/regression-benchmark/results")
    report = await pilot.run(study, root, output, verifier=controlled, image_resolver=lambda _: IMAGE)
    assert len(calls) == 7 and sum(row[2] for row in calls) == 1
    assert report["expectations_met"] == report["expected_observations"] == 6
    assert report["replays"][0]["confirmed"] and not report["benchmark_admitted"]
    assert len(list(output.glob("*-generated.json"))) == 2 and (output / ".gitignore").read_text() == "*\n"
    public = json.loads((output / "public-summary.json").read_bytes())
    assert "rows" not in public and "counterexamples" not in json.dumps(public)
    assert public["replay_confirmations"] == 1


def test_live_catalog_is_pinned_and_contains_only_public_development_cases():
    path = Path(__file__).resolve().parent.parent / "evals/datasets/upstream_patch_pilot.json"
    data = UpstreamCatalog.model_validate_json(path.read_bytes())
    assert len(data.cases) == 4 and len({row.repository_url for row in data.cases}) == 3
    assert {row.expected for row in data.cases} == {"regression", "correct"}
    assert all(row.spec.json_domains and not row.spec.integer_domains for row in data.cases)


def test_cache_download_is_opt_in_and_paths_cannot_escape(tmp_path):
    with pytest.raises(ValueError, match="explicit"):
        GitSource(tmp_path, "https://github.com/example/unit")
    with pytest.raises(ValueError, match="public GitHub"):
        GitSource(tmp_path, "file:///private")


@pytest.mark.parametrize("change", ["runner", "image", "admission"])
def test_runtime_or_review_status_drift_is_rejected(prepared, monkeypatch, change):
    root, study = prepared

    def resolver(_):
        return "sha256:" + "b" * 64 if change == "image" else IMAGE

    if change == "runner":
        monkeypatch.setattr(pilot, "runner_fingerprint", lambda: "f" * 64)
    elif change == "admission":
        study["benchmark_admitted"] = True
        study["fingerprint"] = fingerprint({k: v for k, v in study.items() if k != "fingerprint"})
    with pytest.raises(ValueError):
        pilot.validate_study(study, root, resolver)


@pytest.mark.parametrize(
    "field,value",
    [
        ("source_sha256", "f" * 64),
        ("spec_sha256", "f" * 64),
        ("patch_sha256", "f" * 64),
        ("image_id", "sha256:" + "b" * 64),
        ("mode", "contracts"),
        ("live_provider_calls", 1),
        ("provider_cost_usd", 1),
        ("implementation_sha256", "f" * 64),
    ],
)
def test_resealed_execution_receipts_cannot_cross_a_frozen_boundary(prepared, field, value):
    root, study = prepared
    case = RegressionCorpus.model_validate(study["corpus"]).cases[0]
    _, owner, patch, _ = pilot.case_inputs(root, case)
    spec = type(owner).model_validate({**owner.model_dump(), "image": IMAGE})
    binding = study["bindings"][0]
    report = {
        "status": "inconclusive",
        "reason": "docker_unavailable",
        "trials": [],
        "source_unchanged": True,
        "source_sha256": binding["artifacts"]["source_sha256"],
        "spec_sha256": fingerprint(spec.model_dump()),
        "patch_sha256": hashlib.sha256(patch.encode()).hexdigest(),
        "image_id": "",
        "mode": "generated",
        "implementation_sha256": pilot.implementation_fingerprint(),
        "live_provider_calls": 0,
        "provider_cost_usd": 0,
    }
    assert pilot.observation(pilot._seal(report), spec, patch, binding, "generated") == (
        "inconclusive",
        False,
    )
    report[field] = value
    with pytest.raises(ValueError, match="frozen execution"):
        pilot.observation(pilot._seal(report), spec, patch, binding, "generated")
