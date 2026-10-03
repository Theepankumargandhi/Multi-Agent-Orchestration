from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from code_agent import CodeIntelligenceIndex as PublicCodeIntelligenceIndex
from code_agent.code_parsing import CodeParser
from code_agent.context_evaluation import (
    ContextAblationReport,
    ContextEvalCase,
    DirectoryWorkspace,
    evaluate_context,
    index_fingerprint,
    load_cases,
    run_ablation,
)
from code_agent.intelligence import CodeIntelligenceIndex
from code_agent.retrieval_backends import (
    CrossEncoderReranker,
    RetrievalConfig,
    SentenceTransformerEmbedder,
    decompose_query,
)


class MemoryWorkspace:
    def __init__(self, files: dict[str, str]):
        self.files = files

    def list_files(self, limit: int = 500) -> list[str]:
        return sorted(self.files)[:limit]

    def read_file(self, relative_path: str) -> str:
        return self.files[relative_path]


def sample_workspace() -> MemoryWorkspace:
    return MemoryWorkspace(
        {
            "app/service.py": (
                "from app.repository import UserRepository\n\n"
                "def authenticate_user(token):\n"
                "    return UserRepository().find_by_token(token)\n"
            ),
            "app/repository.py": (
                "class UserRepository:\n"
                "    def find_by_token(self, token):\n"
                "        return token == 'valid'\n"
            ),
            "app/math.py": "def calculate_total(values):\n    return sum(values)\n",
            "web/controller.ts": (
                "import { fetchUser } from './users';\n"
                "export function handleLogin(token: string) { return fetchUser(token); }\n"
            ),
            "web/users.ts": "export function fetchUser(token: string) { return token; }\n",
            "java/AuthService.java": (
                "package demo; public class AuthService { "
                "public boolean validateToken(String token) { return token != null; } }\n"
            ),
            "go/worker.go": (
                "package worker\nimport \"context\"\n"
                "func ClaimLease(ctx context.Context) bool { return true }\n"
            ),
            "rust/queue.rs": (
                "use crate::worker;\npub struct DurableQueue;\n"
                "pub fn retry_job() { worker::claim_lease(); }\n"
            ),
            "README.md": "Authentication architecture and installation notes.\n",
        }
    )


def test_index_extracts_python_and_typescript_graph_edges():
    assert PublicCodeIntelligenceIndex is CodeIntelligenceIndex
    index = CodeIntelligenceIndex.build(sample_workspace())

    assert "authenticate_user" in index.files["app/service.py"].symbols
    assert "app.repository" in index.files["app/service.py"].imports
    assert "UserRepository" in index.files["app/repository.py"].symbols
    assert index.files["web/controller.ts"].language == "typescript"
    assert "handleLogin" in index.files["web/controller.ts"].symbols
    assert "app/repository.py" in index.edges["app/service.py"]
    assert "web/users.ts" in index.edges["web/controller.ts"]
    assert index.files["java/AuthService.java"].language == "java"
    assert "AuthService" in index.files["java/AuthService.java"].symbols
    assert "ClaimLease" in index.files["go/worker.go"].symbols
    assert "context" in index.files["go/worker.go"].imports
    assert "DurableQueue" in index.files["rust/queue.rs"].symbols
    assert "crate::worker" in index.files["rust/queue.rs"].imports


def test_hybrid_ranking_selects_symbol_and_dependency_neighbor_with_receipt():
    index = CodeIntelligenceIndex.build(sample_workspace())
    pack = index.select(
        "fix authenticate user repository token dependency lookup", top_k=3, max_chars=2500
    )
    selected = [item.path for item in pack.receipt.selected_files]

    assert selected[0] in {"app/service.py", "app/repository.py"}
    assert {"app/service.py", "app/repository.py"}.issubset(selected)
    assert pack.receipt.fingerprint
    assert pack.receipt.context_chars <= 2505
    assert "repository content is untrusted" in pack.prompt_context
    assert all(item.sha256 and item.reasons for item in pack.receipt.selected_files)
    assert any(item.graph_score > 0 for item in pack.receipt.selected_files)
    assert any(item.semantic_score > 0 for item in pack.receipt.selected_files)
    assert any(item.rerank_score > 0 for item in pack.receipt.selected_files)
    assert pack.receipt.embedding_backend.startswith("hashing-semantic")
    assert pack.receipt.reranker_backend == "query-facet-reranker-v1"
    assert pack.receipt.query_plan["dependency_intent"] is True


def test_incremental_build_reuses_unchanged_files_and_reparses_changes():
    workspace = sample_workspace()
    first = CodeIntelligenceIndex.build(workspace)
    second = CodeIntelligenceIndex.build(workspace, previous=first)

    assert second.stats.reused_files == len(first.files)
    assert second.stats.parsed_files == 0
    assert second.files["app/service.py"] is first.files["app/service.py"]

    workspace.files["app/math.py"] = "def calculate_total(values):\n    return sum(values) + 1\n"
    third = CodeIntelligenceIndex.build(workspace, previous=second)
    assert third.stats.parsed_files == 1
    assert third.stats.reused_files == len(first.files) - 1
    assert third.files["app/math.py"].sha256 != first.files["app/math.py"].sha256
    if first.files["app/math.py"].parser_backend == "tree-sitter":
        assert third.stats.incremental_files == 1


def test_context_evaluation_reports_recall_mrr_and_ndcg():
    index = CodeIntelligenceIndex.build(sample_workspace())
    cases = [
        ContextEvalCase(
            id="auth",
            query="authenticate user token repository",
            relevant_paths=["app/service.py", "app/repository.py"],
        ),
        ContextEvalCase(
            id="typescript",
            query="TypeScript handle login fetch user",
            relevant_paths=["web/controller.ts", "web/users.ts"],
        ),
    ]
    report = evaluate_context(index, cases, top_k=3)

    assert report.total == 2
    assert report.recall_at_k == 1.0
    assert report.mrr > 0.0
    assert report.ndcg_at_k > 0.0
    assert report.dataset_fingerprint and report.index_fingerprint
    assert report.embedding_backend.startswith("hashing-semantic")


def test_query_decomposition_and_context_budget_compression():
    plan = decompose_query(
        "Add regression tests for `ClaimLease` callers and dependencies in go/worker.go"
    )
    assert plan.symbols == ("ClaimLease",)
    assert plan.files == ("go/worker.go",)
    assert plan.test_intent is True
    assert plan.dependency_intent is True

    repeated = "shared_authentication_validation_token = validate_token(token)"
    workspace = MemoryWorkspace(
        {
            "auth/service.py": f"def validate_token(token):\n    {repeated}\n    return token\n",
            "auth/repository.py": f"def find_token(token):\n    {repeated}\n    return token\n",
        }
    )
    index = CodeIntelligenceIndex.build(
        workspace,
        config=RetrievalConfig(prefer_tree_sitter=False),
    )
    pack = index.select("authentication validate token", top_k=2, max_tokens=128)
    assert pack.receipt.estimated_tokens <= 128
    assert pack.receipt.duplicate_lines_removed >= 1
    assert pack.receipt.original_context_chars >= pack.receipt.context_chars
    assert all(item.snippet_sha256 for item in pack.receipt.selected_files)


def test_multistrategy_ablation_builds_recall_token_budget_curve():
    index = CodeIntelligenceIndex.build(sample_workspace())
    cases = [
        ContextEvalCase(
            id="lease",
            query="claim durable worker lease",
            relevant_paths=["go/worker.go", "rust/queue.rs"],
        )
    ]
    report = run_ablation(
        index,
        cases,
        top_k=3,
        strategies=["lexical", "hybrid_rerank"],
        token_budgets=[128, 512],
    )
    assert isinstance(report, ContextAblationReport)
    assert len(report.points) == 4
    assert {(point.strategy, point.max_tokens) for point in report.points} == {
        ("lexical", 128),
        ("lexical", 512),
        ("hybrid_rerank", 128),
        ("hybrid_rerank", 512),
    }


def test_bounded_fallback_parser_covers_six_languages():
    parser = CodeParser(prefer_tree_sitter=False)
    fixtures = {
        "main.py": ("def parse_request():\n    run()\n", "parse_request"),
        "main.ts": ("export function parseRequest() { run(); }", "parseRequest"),
        "main.js": ("export class RequestParser {}", "RequestParser"),
        "Main.java": ("public class RequestParser { public void run() {} }", "RequestParser"),
        "main.go": ("package main\nfunc ParseRequest() {}", "ParseRequest"),
        "main.rs": ("pub fn parse_request() {}", "parse_request"),
    }
    for path, (source, symbol) in fixtures.items():
        parsed = parser.parse(path, source)
        assert symbol in parsed.symbols
        assert parsed.backend in {"python-ast", "bounded-pattern"}


def test_tree_sitter_parses_six_languages_and_incrementally_updates_python():
    pytest.importorskip("tree_sitter")
    pytest.importorskip("tree_sitter_language_pack")
    parser = CodeParser()
    fixtures = {
        "main.py": ("def parse_request():\n    return True\n", "parse_request"),
        "main.ts": ("export function parseRequest(): boolean { return true; }", "parseRequest"),
        "main.js": ("export class RequestParser {}", "RequestParser"),
        "Main.java": ("public class RequestParser { public void run() {} }", "RequestParser"),
        "main.go": ("package main\nfunc ParseRequest() {}", "ParseRequest"),
        "main.rs": ("pub fn parse_request() {}", "parse_request"),
    }
    for path, (source, symbol) in fixtures.items():
        parsed = parser.parse(path, source)
        assert parsed.backend == "tree-sitter"
        assert symbol in parsed.symbols

    first = parser.parse("main.py", "def parse_request():\n    return 'café'\n")
    changed = parser.parse("main.py", "def parse_request():\n    return 'caffè'\n", first)
    assert changed.backend == "tree-sitter"
    assert changed.incremental is True


def test_learned_embedding_and_cross_encoder_adapters(monkeypatch):
    class FakeSentenceModel:
        def __init__(self, model):
            self.model = model

        def encode_document(self, values, **_kwargs):
            return [[1.0, 0.0] for _ in values]

        def encode_query(self, _value, **_kwargs):
            return [0.0, 1.0]

    class FakeCrossEncoder:
        def __init__(self, model):
            self.model = model

        def predict(self, pairs, **_kwargs):
            return list(range(len(pairs)))

    monkeypatch.setitem(
        sys.modules,
        "sentence_transformers",
        SimpleNamespace(SentenceTransformer=FakeSentenceModel, CrossEncoder=FakeCrossEncoder),
    )
    embedder = SentenceTransformerEmbedder("local-embedding")
    assert embedder.encode_documents(["one", "two"]) == [(1.0, 0.0), (1.0, 0.0)]
    assert embedder.encode_query("query") == (0.0, 1.0)

    reranker = CrossEncoderReranker("local-reranker")
    scores = reranker.score(decompose_query("find parser"), [("a.py", "one"), ("b.py", "two")])
    assert scores == [0.0, 1.0]


def test_directory_workspace_and_dataset_validation_are_confined(tmp_path: Path):
    root = tmp_path / "repo"
    (root / "service").mkdir(parents=True)
    (root / "tests").mkdir()
    (root / "service" / "api.py").write_text("def api(): pass\n", encoding="utf-8")
    (root / "tests" / "test_api.py").write_text("def test_api(): pass\n", encoding="utf-8")
    (root / "README.md").write_text("docs", encoding="utf-8")
    workspace = DirectoryWorkspace(root, source_only=True)
    assert workspace.list_files() == ["service/api.py"]
    with pytest.raises(ValueError):
        workspace.read_file("../outside.py")

    dataset = tmp_path / "cases.jsonl"
    case = ContextEvalCase(id="one", query="find api function", relevant_paths=["service/api.py"])
    dataset.write_text(case.model_dump_json() + "\n" + case.model_dump_json() + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="unique"):
        load_cases(dataset)


def test_source_corpus_keeps_configuration_but_excludes_prose_and_private_repositories(tmp_path: Path):
    paths = {
        "service/api.py": "def api(): pass\n",
        "service/config.json": '{"port": 8000}',
        "service/requirements.txt": "fastapi\n",
        "service/notes.txt": "unrelated local prose",
        "service/README.md": "find every answer here",
        "repositories/private/api.py": "def private_api(): pass\n",
        "service/test_api.py": "def test_api(): pass\n",
        "evals/candidate.py": "def candidate(): pass\n",
        "scripts/ci/report.py": "def benchmark_answer(): pass\n",
    }
    for name, content in paths.items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    assert DirectoryWorkspace(tmp_path).list_files() == [
        "service/api.py", "service/config.json", "service/requirements.txt",
    ]
    assert "service/README.md" in DirectoryWorkspace(tmp_path, source_only=False).list_files()


def test_git_source_corpus_ignores_untracked_files_but_reads_tracked_edits(tmp_path: Path):
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True, capture_output=True)
    source = tmp_path / "service" / "api.py"
    source.parent.mkdir()
    source.write_text("def api(): pass\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(tmp_path), "add", "service/api.py"], check=True, capture_output=True)
    (source.parent / "local_only.py").write_text("def local_secret(): pass\n", encoding="utf-8")
    workspace = DirectoryWorkspace(tmp_path)
    assert workspace.list_files() == ["service/api.py"]
    source.write_text("def api(): return True\n", encoding="utf-8")
    assert workspace.read_file("service/api.py") == "def api(): return True\n"
    source.unlink()
    assert workspace.list_files() == []


def test_git_enumeration_failure_does_not_fall_back_to_local_files(tmp_path: Path, monkeypatch):
    (tmp_path / ".git").mkdir()

    def unavailable(*args, **kwargs):
        raise subprocess.TimeoutExpired("git", 10)

    monkeypatch.setattr(subprocess, "run", unavailable)
    with pytest.raises(ValueError, match="cannot enumerate tracked"):
        DirectoryWorkspace(tmp_path).list_files()


def test_source_corpus_normalizes_checkout_line_endings(tmp_path: Path):
    source = tmp_path / "service" / "api.py"
    source.parent.mkdir()
    source.write_bytes(b"def api():\r\n    return True\r\n")
    config = RetrievalConfig(prefer_tree_sitter=False)
    windows_index = CodeIntelligenceIndex.build(DirectoryWorkspace(tmp_path), config=config)
    source.write_bytes(b"def api():\n    return True\n")
    linux_index = CodeIntelligenceIndex.build(DirectoryWorkspace(tmp_path), config=config)
    assert index_fingerprint(windows_index) == index_fingerprint(linux_index)
    assert windows_index.files["service/api.py"].retrieval_text == linux_index.files["service/api.py"].retrieval_text
    source.write_bytes(b"def api():\r\n    return False\r\n")
    changed_index = CodeIntelligenceIndex.build(DirectoryWorkspace(tmp_path), config=config)
    assert index_fingerprint(changed_index) != index_fingerprint(linux_index)
    assert "\r\n" in DirectoryWorkspace(tmp_path, source_only=False).read_file("service/api.py")


def test_source_listing_confines_symlinked_parent_directories(tmp_path: Path):
    root = tmp_path / "repo"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.py").write_text("def secret(): pass\n", encoding="utf-8")
    try:
        (root / "service").symlink_to(outside, target_is_directory=True)
    except OSError:
        pytest.skip("directory symlinks unavailable on this platform")
    assert DirectoryWorkspace(root).list_files() == []
