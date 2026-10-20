from __future__ import annotations

import copy
import difflib
import json
import subprocess
from pathlib import Path

import pytest

from code_agent.patch_inputs import (
    PatchVerifySpec,
    apply_file_patch,
    generated_cases,
    import_prediction,
    parse_patch,
    read_bounded,
    strict_json,
)
from code_agent.patch_verify import (
    _decode,
    assess,
    private_directory,
    reserve_private_directory,
    tree_fingerprint,
    verify_patch,
)
from code_agent.upgrade_models import fingerprint

FIXTURE = Path(__file__).parent / "fixtures/patch_verify"


def specification(name="fee-regression"):
    return PatchVerifySpec.model_validate_json((FIXTURE / f"{name}.json").read_bytes())


def patch(name="fee-regression"):
    return (FIXTURE / f"{name}.diff").read_text(encoding="utf-8")


def test_exact_patch_and_git_wrapper():
    spec = specification()
    text = (FIXTURE / "app.py").read_text(encoding="utf-8")
    expected = text.replace("max(0, quantity) * 10", "(quantity if quantity else 1) * 10")
    for raw in (patch(), "diff --git a/app.py b/app.py\nindex abcdef0..1234567 100644\n" + patch()):
        edits = parse_patch(raw, spec)
        assert len(edits) == 1
        assert apply_file_patch(text, edits[0]) == expected


@pytest.mark.parametrize("name", ["../app.py", "tests/test_app.py", ".env", ".git/config", "setup.py", "conftest.py", "app.js", "other.py", "app.py\tstamp", "app/../app.py"])
def test_external_patch_cannot_escape_owner_scope(name):
    with pytest.raises(ValueError):
        parse_patch(patch().replace("a/app.py", "a/" + name).replace("b/app.py", "b/" + name), specification())


@pytest.mark.parametrize("raw", ["", "garbage", "rename from app.py\n", "GIT binary patch\n", "new file mode 120000\n", "\x00", "+" * 200001],
                         ids=["empty", "garbage", "rename", "binary", "symlink_mode", "nul", "oversized"])
def test_unsupported_patches_fail(raw):
    with pytest.raises(ValueError):
        parse_patch(raw, specification())


@pytest.mark.parametrize("mutate", [
    lambda value: value.replace("@@ -4,2 +4,2 @@", "@@ -4,3 +4,2 @@"),
    lambda value: value.replace("@@ -4,2 +4,2 @@", "@@ -4,2 +7,2 @@"),
    lambda value: value.replace("max(0, quantity)", "max(1, quantity)"),
    lambda value: value + value,
    lambda value: value.replace("+++ b/app.py", "+++ b/other.py"),
    lambda value: value.replace("\n", "\r\n"),
])
def test_ambiguous_stale_or_malformed_patch_fails(mutate):
    with pytest.raises(ValueError):
        for edit in parse_patch(mutate(patch()), specification()):
            apply_file_patch((FIXTURE / "app.py").read_text(), edit)


def test_no_final_newline_round_trip():
    before, after = "def f():\n    return 1", "def f():\n    return 2"
    raw = "--- a/app.py\n+++ b/app.py\n@@ -1,2 +1,2 @@\n def f():\n-    return 1\n\\ No newline at end of file\n+    return 2\n\\ No newline at end of file\n"
    assert apply_file_patch(before, parse_patch(raw, specification())[0]) == after


def test_hunks_cannot_suppress_failing_tests():
    raw = "".join(difflib.unified_diff(["assert False\n"], ["assert True\n"], fromfile="a/test_app.py", tofile="b/test_app.py"))
    with pytest.raises(ValueError):
        parse_patch(raw, specification())


def test_generator_is_seeded_bounded_and_includes_boundary():
    spec = specification()
    cases = generated_cases(spec)
    assert cases == generated_cases(spec)
    assert len(cases) == 8 and len({row.case_id for row in cases}) == 8
    assert {0, 1, 9, 10} <= {row.args[0] for row in cases}
    assert all(0 <= row.args[0] <= 10 for row in cases)


@pytest.mark.parametrize("change", [
    {"integer_domains": [{"domain_id": "bad", "entrypoint": "app:fee", "minimum": 3, "maximum": 2}]},
    {"contracts": [{"case_id": "generated.x", "entrypoint": "app:fee"}]},
    {"source_paths": ["test_app.py"]}, {"test_paths": ["../tests"]}, {"unexpected": True},
    {"seed": True}, {"preservation_authorized": "true"},
])
def test_spec_rejects_ambiguous_or_unsafe_input(change):
    with pytest.raises(ValueError):
        PatchVerifySpec.model_validate({**specification().model_dump(), **change})


@pytest.mark.parametrize("shape", ["mapping", "list", "record"])
def test_prediction_adapter_binds_one_instance_without_trajectory(shape):
    row = {"instance_id": "owner__repo-1", "model_name_or_path": "example-model", "model_patch": patch()}
    payload = {row["instance_id"]: row} if shape == "mapping" else [row] if shape == "list" else row
    imported, origin = import_prediction(json.dumps(payload).encode(), row["instance_id"])
    assert imported == patch()
    assert origin["producer_authenticated"] is False
    assert "model_name_or_path" not in origin


def test_prediction_duplicates_missing_null_and_mismatched_identity_rejected():
    row = {"instance_id": "id", "model_patch": patch()}
    for payload in ([row, row], {"wrong": row}, {"id": {**row, "instance_id": "wrong"}}, {**row, "model_patch": None}):
        with pytest.raises(ValueError):
            import_prediction(json.dumps(payload).encode(), "id")
    with pytest.raises(ValueError):
        strict_json('{"a": 1, "a": 2}')
    with pytest.raises(ValueError):
        strict_json('{"a": NaN}')


def trials():
    cases = [specification().contracts[0]]
    tests = {"passed": True, "exit_code": 0, "collected": 2, "node_ids_sha256": "1" * 64, "collection_errors": 0}
    rows = [{"arm": arm, "repeat": repeat, "snapshot_sha256": "a" * 64 if arm == "original" else "b" * 64,
             "tests": copy.deepcopy(tests), "behavior": {cases[0].case_id: {"sha256": "3" * 64, "kind": "value", "type": None}}}
            for arm in ("original", "candidate") for repeat in range(2)]
    return rows, cases


def test_assessor_requires_complete_stable_owner_authorized_evidence():
    rows, cases = trials()
    assert assess(rows, cases, True)[0] == "checks_passed"
    assert assess(rows[:3], cases, True)[0] == "inconclusive"
    assert assess(rows, cases, False)[0] == "inconclusive"
    assert assess(rows, [], True)[0] == "inconclusive"
    rows[2]["behavior"]["ordinary"]["sha256"] = "4" * 64
    assert assess(rows, cases, True)[1] == "unstable_execution"
    rows[3]["behavior"]["ordinary"]["sha256"] = "4" * 64
    assert assess(rows, cases, True) == ("regression_found", "owner_preservation_violation", ["ordinary"])


@pytest.mark.parametrize("change,reason", [
    ("baseline_tests", "original_tests_not_all_passed"), ("different_tests", "test_population_changed"),
    ("candidate_tests", "existing_test_regression"), ("original_error", "original_behavior_does_not_match_owner_scope"),
    ("candidate_harness", "candidate_behavior_harness_error"),
])
def test_assessor_does_not_relabel_invalid_evidence(change, reason):
    rows, cases = trials()
    selected = rows[:2] if change in {"baseline_tests", "original_error"} else rows[2:]
    for row in selected:
        if change in {"baseline_tests", "candidate_tests"}:
            row["tests"]["passed"] = False
        elif change == "different_tests":
            row["tests"]["node_ids_sha256"] = "f" * 64
        else:
            row["behavior"]["ordinary"]["kind"] = "error" if change == "original_error" else "harness_error"
    assert assess(rows, cases, True)[1] == reason


def test_receipt_binding_rejects_replay_and_missing_cases():
    rows, cases = trials()
    receipt = {"binding": "bound", "mode": "behavior", "result": rows[0]["behavior"]}
    value, digest = _decode(json.dumps(receipt), "bound", "behavior", cases)
    assert value == receipt["result"] and digest == fingerprint(receipt)
    for update in ({"binding": "stale"}, {"result": {}}, {"mode": "tests"}):
        with pytest.raises(ValueError):
            _decode(json.dumps({**receipt, **update}), "bound", "behavior", cases)


def test_private_artifacts_refuse_public_or_namespace_root(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert private_directory(tmp_path / "data/patch-verify/run") == tmp_path / "data/patch-verify/run"
    for path in (tmp_path / "reports", tmp_path / "data/patch-verify", tmp_path / "data/patch-verify/../public"):
        with pytest.raises(ValueError):
            private_directory(path)


def test_installed_tool_artifacts_are_ignored_in_another_checkout(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    subprocess.run(["git", "init", "--quiet"], cwd=tmp_path, check=True, capture_output=True)
    output = reserve_private_directory(tmp_path / "data/patch-verify/external-tool-run")
    receipt = output / "report.json"
    receipt.write_text('{"private": true}')
    result = subprocess.run(["git", "check-ignore", "--quiet", "--", str(receipt)], cwd=tmp_path, capture_output=True)
    assert result.returncode == 0 and (output / ".gitignore").read_text() == "*\n"
    with pytest.raises(FileExistsError):
        reserve_private_directory(output)


def test_tree_fingerprint_binds_source_and_never_imports_it(tmp_path):
    source = tmp_path / "app.py"
    source.write_text("raise RuntimeError('must not import on host')\n")
    before = tree_fingerprint(tmp_path)
    source.write_text("raise RuntimeError('changed')\n")
    assert tree_fingerprint(tmp_path) != before
    with pytest.raises(ValueError):
        read_bounded(source, 1)


def test_verifier_implementation_fingerprint_binds_shared_credential_rules(monkeypatch):
    from code_agent.patch_verify import implementation_fingerprint

    before = implementation_fingerprint()
    original = Path.read_bytes

    def changed_rules(path):
        data = original(path)
        return data + b"\n# changed credential rule\n" if path.name == "secret_patterns.py" else data

    monkeypatch.setattr(Path, "read_bytes", changed_rules)
    assert implementation_fingerprint() != before


@pytest.mark.asyncio
async def test_invalid_patch_and_missing_consent_never_start_docker():
    def forbidden(*args):
        raise AssertionError("Docker must not be started")
    report = await verify_patch(FIXTURE, patch().replace("a/app.py", "a/../app.py"), specification(), sandbox_factory=forbidden, image_resolver=forbidden)
    assert report["status"] == "inconclusive" and not report["trials"]
    spec = specification().model_copy(update={"preservation_authorized": False})
    report = await verify_patch(FIXTURE, patch(), spec, sandbox_factory=forbidden, image_resolver=forbidden)
    assert report["status"] == "inconclusive" and report["source_unchanged"]


@pytest.mark.asyncio
async def test_cleanup_happens_after_failed_start():
    instances = []

    class FailedSandbox:
        def __init__(self, *args):
            instances.append(self)
            self.closed = False

        def start(self):
            raise RuntimeError("repository-controlled exception must not appear")

        def close(self):
            self.closed = True

    report = await verify_patch(FIXTURE, patch(), specification(), sandbox_factory=FailedSandbox, image_resolver=lambda _: "sha256:" + "a" * 64)
    assert len(instances) == 1 and instances[0].closed
    assert "repository-controlled" not in json.dumps(report)
    assert report["status"] == "inconclusive" and report["source_unchanged"]
