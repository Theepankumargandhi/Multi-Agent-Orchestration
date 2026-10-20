"""Build/install the SDK-only wheel and exercise it outside the source checkout."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

_EXTERNAL_CHECK = '''import hashlib, importlib.metadata, json, socket, sys
from pathlib import Path
target = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(target))
class BlockPlatform:
    def find_spec(self, fullname, *args):
        if fullname.split(".")[0] in {"agent", "code_agent", "client", "service", "langchain", "langchain_core", "langgraph", "chromadb", "openai", "groq", "dotenv", "opentelemetry"}:
            raise AssertionError("SDK tried to import the platform/provider stack")
sys.meta_path.insert(0, BlockPlatform())
class ForbiddenSocket(socket.socket):
    def __new__(cls, *args, **kwargs):
        raise AssertionError("SDK attempted network access")
socket.socket = ForbiddenSocket
import agentforge_sdk
from agentforge_sdk import DocumentRAG, DocumentPage
from agentforge_sdk import NumericClaim, NumericClaimCheck, check_numeric_claim, verify_numeric_claim
from agentforge_sdk import EvidenceSet
assert callable(EvidenceSet.without_sources) and isinstance(EvidenceSet.source_ids, property)
assert NumericClaim(claim_id="installed", label="Explicit numeric assertion", expected_value="25.0", expected_unit="%").expected_value == "25.0"
assert Path(agentforge_sdk.__file__).resolve().is_relative_to(target)
rag = DocumentRAG([DocumentPage(document_id="installed", source="report.pdf", page=2, version="r1", text="Atlas p95 latency is 180 ms.")])
answer = rag.answer("Atlas p95 latency", top_k=1)
assert answer.status == "evidence_found" and "180 ms" in answer.answer
assert answer.citations[0].page == 2 and rag.verify_answer(answer, question="Atlas p95 latency")
assert rag.answer("xylophonium").status == "abstained"
assert not rag.verify_answer(answer.model_copy(update={"answer": "made up"}))
assert not any(name.split(".")[0] in {"agent", "code_agent", "langchain_core", "langgraph", "dotenv", "openai", "groq"} for name in sys.modules)
assert not any(name.split(".")[0] in {"pdfplumber", "pypdfium2", "PIL"} for name in sys.modules)
assert not list(Path.cwd().iterdir())
metadata = importlib.metadata.distribution("agentforge-document-sdk")
assert metadata.version == "0.10.0"
assert all("extra ==" in row or row.lower().startswith("pydantic") for row in metadata.requires)
from agentforge_sdk import DocumentReasoningAgent, PlannerOutput, PlannerUsage, ReasoningPlan
class AuthoredPlanner:
    name, external_processing = "installed-authored-control", False
    calls = 0
    def plan(self, request):
        self.calls += 1
        selected = next(c.candidate_id for c in request.candidates if c.kind == "quote")
        return PlannerOutput(plan_json=ReasoningPlan(action="quote", candidate_ids=(selected,)).model_dump_json(),
                             usage=PlannerUsage(provider_requests=0, provider_cost_usd=0.0))
planner = AuthoredPlanner()
agent = DocumentReasoningAgent(rag, planner)
reasoning = agent.run("Atlas p95 latency")
assert reasoning.status == "answered" and agent.verify_result(reasoning, question="Atlas p95 latency")
assert planner.calls == 1 and reasoning.citations[0].page == 2
print(json.dumps({"installed_module_used": True, "outside_checkout": True, "network_blocked": True,
                  "platform_imports_blocked": True, "literal_answer": True, "citation_page": 2,
                  "receipt_verified": True, "abstention_verified": True, "tampering_rejected": True,
                  "cwd_files_created": 0, "sdk_version": metadata.version, "reasoning_agent_verified": True}))
'''

_EXTERNAL_BUNDLE_CHECK = '''import contextlib, io, json, socket, sys
from pathlib import Path
target, source = (Path(p).resolve() for p in sys.argv[1:])
sys.path.insert(0, str(target))
class BlockPlatform:
    def find_spec(self, fullname, *args):
        if fullname.split(".")[0] in {"agent", "code_agent", "service", "langchain_core", "langgraph", "openai", "groq", "dotenv", "pdfplumber", "pypdfium2", "pypdf", "PIL"}:
            raise AssertionError("offline replay imported a platform, provider or parser")
sys.meta_path.insert(0, BlockPlatform())
class ForbiddenSocket(socket.socket):
    def __new__(cls, *args, **kwargs):
        raise AssertionError("offline replay attempted network access")
socket.socket = ForbiddenSocket
import agentforge_sdk
from agentforge_sdk import load_claim_replay_bundle, verify_claim_replay_bundle, ClaimBundleError
from agentforge_sdk.bundle_cli import main
assert Path(agentforge_sdk.__file__).resolve().is_relative_to(target)
with source.open("rb") as stream:
    bundle = load_claim_replay_bundle(stream.read(3_000_001))
verified = verify_claim_replay_bundle(bundle, expected_fingerprint=bundle.fingerprint)
assert verified.replay_verified and verified.total == len(bundle.entries)
assert verified.supported + verified.refuted + verified.unresolved == verified.total
assert not verified.source_files_authenticated and not verified.extraction_reexecuted and not verified.release_authorized
output = io.StringIO()
with contextlib.redirect_stdout(output):
    status = main([str(source), "--expected-fingerprint", bundle.fingerprint, "--require-supported"])
assert status == (0 if verified.all_numeric_assertions_supported else 2)
assert json.loads(output.getvalue())["bundle_fingerprint"] == bundle.fingerprint
try:
    verify_claim_replay_bundle(bundle, expected_fingerprint="0" * 64)
except ClaimBundleError:
    pass
else:
    raise AssertionError("wrong pin accepted")
assert not list(Path.cwd().iterdir())
print(json.dumps({"installed_bundle_module_used": True, "network_blocked": True,
                  "platform_provider_and_parser_imports_blocked": True, "total": verified.total,
                  "supported": verified.supported, "refuted": verified.refuted, "unresolved": verified.unresolved,
                  "pinned_replay_verified": True, "wrong_pin_rejected": True, "cli_exit_code": status,
                  "cwd_files_created": 0}))
'''

_EXTERNAL_MULTIMODAL_CHECK = '''import json, socket, sys
from pathlib import Path
target, fixtures = (Path(p).resolve() for p in sys.argv[1:])
sys.path.insert(0, str(target))
class BlockPlatform:
    def find_spec(self, fullname, *args):
        if fullname.split(".")[0] in {"agent", "code_agent", "service", "langchain_core", "langgraph", "openai", "groq", "dotenv"}:
            raise AssertionError("multimodal SDK imported platform/provider code")
sys.meta_path.insert(0, BlockPlatform())
class ForbiddenSocket(socket.socket):
    def __new__(cls, *args, **kwargs):
        raise AssertionError("multimodal SDK attempted network access")
socket.socket = ForbiddenSocket
import agentforge_sdk
from agentforge_sdk import load_multimodal, MultimodalRAG, TesseractOCR
assert Path(agentforge_sdk.__file__).resolve().is_relative_to(target)
document = load_multimodal(fixtures / "native.pdf", document_id="installed")
rag = MultimodalRAG(document)
answer = rag.answer("Atlas p95 latency", top_k=1)
assert "180 ms" in answer.answer and rag.verify_answer(answer)
cells = {(r.row, r.column): r.region_id for r in document.pages[0].regions if r.kind == "table_cell"}
calc = rag.calculate("sum", [cells[1, 1], cells[1, 2]])
assert calc.value == "271.25" and rag.verify_calculation(calc)
scan = load_multimodal(fixtures / "scanned.pdf", document_id="scan", ocr=TesseractOCR())
ocr_rag = MultimodalRAG(scan)
ocr_answer = ocr_rag.answer("Recovery time seconds", top_k=1)
assert scan.ready and "42 seconds" in ocr_answer.answer and ocr_rag.verify_answer(ocr_answer)
assert not list(Path.cwd().iterdir())
print(json.dumps({"installed_multimodal_module_used": True, "network_blocked": True,
                  "platform_imports_blocked": True, "native_pdf_answer": True,
                  "decimal_cell_sum": True, "real_scanned_pdf_ocr": True, "cwd_files_created": 0}))
'''

_EXTERNAL_REVIEW_CHECK = '''import json, socket, sys
from pathlib import Path
target, evidence_dir = (Path(p).resolve() for p in sys.argv[1:])
sys.path.insert(0, str(target))
class BlockPlatform:
    def find_spec(self, fullname, *args):
        if fullname.split(".")[0] in {"agent", "code_agent", "service", "langchain_core", "langgraph", "openai", "groq", "dotenv"}:
            raise AssertionError("review SDK imported platform/provider code")
sys.meta_path.insert(0, BlockPlatform())
class ForbiddenSocket(socket.socket):
    def __new__(cls, *args, **kwargs):
        raise AssertionError("review SDK attempted network access")
socket.socket = ForbiddenSocket
import agentforge_sdk
from agentforge_sdk import ReviewAuthority, ReviewedEvidence, ReviewedRAG, ReviewError
assert Path(agentforge_sdk.__file__).resolve().is_relative_to(target)
bundle = ReviewedEvidence.model_validate_json((evidence_dir / "bundle.json").read_text())
authority = ReviewAuthority({"authored-reviewer": bytes(range(32))}, clock=lambda: 1000)
rag = ReviewedRAG(bundle, authority=authority, source_path=evidence_dir / "fixtures/review-table.png")
answer = rag.answer("Requests After", top_k=1)
assert [c.spatial.quote for c in answer.citations] == ["Requests", "After", "125"]
assert rag.verify_answer(answer, question="Requests After")
assert all(c.origin == "reviewer_supplied" for c in answer.citations)
assert rag.citation(answer.citations[-1].spatial.region_id) == answer.citations[-1]
cells = {(r.row, r.column): r.region_id for r in bundle.document.pages[0].regions if r.kind == "table_cell"}
calc = rag.calculate("percent_change", [cells[1,1], cells[1,2]])
assert calc.result.value == "25" and rag.verify_calculation(calc)
authority.revoke(bundle.approval)
assert not rag.verify_answer(answer) and not rag.verify_calculation(calc)
assert not list(Path.cwd().iterdir())
print(json.dumps({"installed_review_module_used": True, "network_blocked": True,
                  "reviewed_grid_lookup": True, "review_lineage_verified": True,
                  "reviewed_calculation": True, "revocation_enforced": True, "cwd_files_created": 0}))
'''

_EXTERNAL_CHANGE_CHECK = '''import json, socket, sys
from pathlib import Path
target, evidence_dir = (Path(p).resolve() for p in sys.argv[1:])
sys.path.insert(0, str(target))
class BlockPlatform:
    def find_spec(self, fullname, *args):
        if fullname.split(".")[0] in {"agent", "code_agent", "service", "langchain_core", "langgraph", "openai", "groq", "dotenv"}:
            raise AssertionError("change SDK imported platform/provider code")
sys.meta_path.insert(0, BlockPlatform())
class ForbiddenSocket(socket.socket):
    def __new__(cls, *args, **kwargs):
        raise AssertionError("change SDK attempted network access")
socket.socket = ForbiddenSocket
import agentforge_sdk
from agentforge_sdk import (load_multimodal, MultimodalRAG, DocumentRevision,
                           RevalidationBatch, RevalidationConfig, SavedAnswer, SavedCalculation,
                           EvidenceSet, EvidenceSource, FactQuery)
assert Path(agentforge_sdk.__file__).resolve().is_relative_to(target)
before = load_multimodal(evidence_dir / "fixtures/v1.pdf", document_id="atlas", source="atlas.pdf", version="v1")
after = load_multimodal(evidence_dir / "fixtures/v2.pdf", document_id="atlas", source="atlas.pdf", version="v2")
old, new = MultimodalRAG(before), MultimodalRAG(after)
regions = before.pages[0].regions
row = next(r.row for r in regions if r.kind == "table_cell" and r.column == 0 and r.text == "Requests")
ids = [next(r.region_id for r in regions if r.row == row and r.column == col) for col in (1,2)]
source = old.calculate("percent_change", ids)
revision = DocumentRevision(old, new)
replay = revision.recalculate(source)
assert source.value == "25" and replay.result.value == "50" and replay.value_changed
assert revision.verify_recalculation(replay, source) and new.verify_calculation(replay.result)
assert all(c.page == 2 for c in replay.result.citations)
assert not new.verify_calculation(source)
question = "Atlas p95 latency"
saved = [SavedAnswer(item_id="latency", question=question, receipt=old.answer(question, top_k=1)),
         SavedCalculation(item_id="requests", receipt=source)]
batch = RevalidationBatch(old, new, saved, config=RevalidationConfig(
    refresh_answers=True, replay_calculations=True, max_items_per_step=1, answer_top_k=1))
first = batch.advance()
assert not first.complete and first.answer_attempts == 1
final = batch.advance(batch.restore(first.model_dump_json()))
assert final.complete and batch.verify_checkpoint(final) and final.entries[1].replay.result.value == "50"
assert [q.item_id for q in batch.review_queue(final)] == ["requests"]
assert batch.advance(final) == final
same = MultimodalRAG(load_multimodal(evidence_dir / "fixtures/same-content.pdf", document_id="matching"))
scope = "atlas:2026-Q1:authored-report"
collection = EvidenceSet([EvidenceSource("old", scope, old), EvidenceSource("matching", scope, same)])
queries = [FactQuery(row_label="Requests", column_label=c, scope=scope) for c in ("Before", "After")]
facts = [collection.lookup(q) for q in queries]
assert all(f.status == "agreed" for f in facts) and facts[1].value == "125"
reconciled = collection.calculate("percent_change", facts)
assert reconciled.value == "25" and collection.verify_calculation(reconciled)
contradictory = EvidenceSet([EvidenceSource("old", scope, old), EvidenceSource("matching", scope, same), EvidenceSource("new", scope, new)])
conflict = contradictory.lookup(queries[1])
assert conflict.status == "conflict" and conflict.value is None
assert contradictory.calculate("sum", [conflict]).reason == "unresolved_fact"
assert not list(Path.cwd().iterdir())
print(json.dumps({"installed_revision_module_used": True, "network_blocked": True,
                  "page_and_row_alignment": True, "calculation_replayed": True,
                  "new_receipt_verified": True, "stale_receipt_rejected": True,
                  "batch_resume_verified": True, "changed_result_queued": True,
                  "cross_document_reconciliation": True, "conflict_blocks_calculation": True,
                  "reconciled_decimal_tool": True, "cwd_files_created": 0}))
'''


def evaluate(output: Path, *, multimodal_fixtures: Path | None = None, review_evidence: Path | None = None,
             change_evidence: Path | None = None, claim_bundle: Path | None = None):
    output = output.absolute()
    expected_root = (Path.cwd() / "data/evaluations").resolve()
    if output.resolve() == expected_root or not output.resolve().is_relative_to(expected_root):
        raise ValueError("packaging evidence must be below data/evaluations/")
    for path in (output, *output.parents):
        if path.is_symlink() or (path.exists() and getattr(path.lstat(), "st_file_attributes", 0) & 0x400):
            raise ValueError("packaging output links are unsupported")
    output.mkdir(parents=True, exist_ok=False)
    (output / ".gitignore").write_text("*\n", encoding="utf-8")
    # Build tools can exceed legacy Windows path limits under a deep checkout.
    # Keep disposable build/install paths short; retain only evidence and wheels.
    with tempfile.TemporaryDirectory(prefix="afv-sdk-") as temporary:
        return _evaluate_package(output, Path(temporary), multimodal_fixtures, review_evidence, change_evidence, claim_bundle)


def _evaluate_package(output: Path, workspace: Path, multimodal_fixtures: Path | None = None,
                      review_evidence: Path | None = None, change_evidence: Path | None = None,
                      claim_bundle: Path | None = None):
    staging = workspace / "staging"
    shutil.copytree(ROOT / "sdk", staging / "sdk", ignore=shutil.ignore_patterns("build", "dist", "*.egg-info", "__pycache__"))
    shutil.copytree(ROOT / "agentforge_sdk", staging / "agentforge_sdk", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    wheel_dir, target, application = output / "wheels", workspace / "installed", workspace / "external-app"
    application.mkdir()
    env = {**os.environ, "PIP_NO_INDEX": "1", "PIP_DISABLE_PIP_VERSION_CHECK": "1", "RAG_CHUNK_SIZE": "not-an-sdk-setting"}

    def run(name, command, cwd=None):
        result = subprocess.run(command, cwd=cwd, env=env, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=180)
        (output / f"{name}.log").write_text(result.stdout + result.stderr, encoding="utf-8")
        if result.returncode:
            raise RuntimeError(f"SDK packaging control failed at {name}; inspect its private log")
        return result

    run("build", [sys.executable, "-m", "pip", "wheel", "--no-deps", "--no-build-isolation", "--no-index", "--no-cache-dir",
                  str(staging / "sdk"), "--wheel-dir", str(wheel_dir)])
    wheels = list(wheel_dir.glob("*.whl"))
    if len(wheels) != 1:
        raise ValueError("expected exactly one SDK wheel")
    with zipfile.ZipFile(wheels[0]) as archive:
        files = archive.namelist()
        if (any(not (name.startswith("agentforge_sdk/") or name.startswith("agentforge_document_sdk-0.10.0.dist-info/")) for name in files)
                or not any(name.endswith("/licenses/LICENSE") for name in files)
                or "agentforge_sdk/py.typed" not in files):
            raise ValueError("SDK wheel contains unexpected files or lacks license/typing metadata")
    run("install", [sys.executable, "-m", "pip", "install", "--no-deps", "--no-index", "--no-compile", "--target", str(target), str(wheels[0])])
    check = run("external-app", [sys.executable, "-I", "-B", "-c", _EXTERNAL_CHECK, str(target)], cwd=application)
    report = {"evidence_kind": "actual_local_wheel_install_authored_application", "passed": True,
              "wheel_sha256": hashlib.sha256(wheels[0].read_bytes()).hexdigest(), "wheel_bytes": wheels[0].stat().st_size,
              "wheel_files": files, "checks": json.loads(check.stdout), "live_provider_calls": 0,
              "package_published": False, "dependencies_downloaded": False}
    if multimodal_fixtures is not None:
        result = run("external-multimodal-app", [sys.executable, "-I", "-B", "-c", _EXTERNAL_MULTIMODAL_CHECK,
                     str(target), str(multimodal_fixtures.resolve())], cwd=application)
        report["multimodal_checks"] = json.loads(result.stdout)
    if review_evidence is not None:
        result = run("external-review-app", [sys.executable, "-I", "-B", "-c", _EXTERNAL_REVIEW_CHECK,
                     str(target), str(review_evidence.resolve())], cwd=application)
        report["review_checks"] = json.loads(result.stdout)
    if change_evidence is not None:
        result = run("external-change-app", [sys.executable, "-I", "-B", "-c", _EXTERNAL_CHANGE_CHECK,
                     str(target), str(change_evidence.resolve())], cwd=application)
        report["change_checks"] = json.loads(result.stdout)
    if claim_bundle is not None:
        result = run("external-claim-bundle-app", [sys.executable, "-I", "-B", "-c", _EXTERNAL_BUNDLE_CHECK,
                     str(target), str(claim_bundle.resolve())], cwd=application)
        report["claim_bundle_checks"] = json.loads(result.stdout)
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--multimodal-fixtures", type=Path, help="Explicit authored fixtures from the multimodal control run")
    parser.add_argument("--review-evidence", type=Path, help="Explicit authored evidence from the review control run")
    parser.add_argument("--change-evidence", type=Path, help="Explicit authored PDF evidence from the change control run")
    parser.add_argument("--claim-bundle", type=Path, help="Explicit private numeric replay bundle from the API control")
    args = parser.parse_args()
    result = evaluate(args.output_dir, multimodal_fixtures=args.multimodal_fixtures,
                      review_evidence=args.review_evidence, change_evidence=args.change_evidence,
                      claim_bundle=args.claim_bundle)
    print(json.dumps({key: value for key, value in result.items() if key != "wheel_files"}, indent=2))


if __name__ == "__main__":
    main()
