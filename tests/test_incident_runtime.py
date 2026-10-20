"""Actual local-RAG operations and private capture; all generation is a labelled test double."""
import importlib

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableLambda

from agent import local_rag
from agent.incident_models import IncidentBundle, configuration_for
from agent.incident_pipeline import prepare_replay
from agent.incident_runtime import RuntimeCapture, active_capture, document_payload, runtime_context
from tests.test_incident_investigation import store


@pytest.fixture
def local_corpus(tmp_path, monkeypatch):
    path = str(tmp_path / "rag.db")
    monkeypatch.setattr(local_rag, "USE_CHROMA_RAG", False)
    monkeypatch.setattr(local_rag, "_redis_disabled", True)
    monkeypatch.setattr(local_rag, "_local_rag_cache", {})
    local_rag.add_local_document("Telescope", "Background information. " * 40 +
        "\nThe telescope request limit for authenticated users is 42.\nNot for anonymous users.", db_path=path)
    return path


def test_runtime_binding_includes_shared_sdk_retrieval_source(monkeypatch):
    from pathlib import Path

    from agent.incident_models import runtime_fingerprint

    before = runtime_fingerprint()
    original = Path.read_bytes

    def changed(path):
        if path.as_posix().endswith("agentforge_sdk/_retrieval.py"):
            return b"different shared retrieval implementation"
        return original(path)

    monkeypatch.setattr(Path, "read_bytes", changed)
    assert runtime_fingerprint() != before


def capture_sqlite(vault, path, *, family="runtime-family", owner="alice"):
    capture = RuntimeCapture(owner, family, "test-double-generation", vault)
    with capture.scope():
        notes, _ = local_rag.search_local_knowledge("telescope request limit authenticated", db_path=path, return_meta=True)
        user = "Question: telescope request limit authenticated\nLocal RAG evidence:\n" + notes
        capture.before_generation("Use evidence.", user, {"route": "rag"})
        capture.model_response("Use evidence.", user, AIMessage(content="Insufficient evidence."), "groq:test-double")
    receipt = capture.finish({"route": "rag"}, "Insufficient evidence.")
    assert receipt["status"] == "captured"
    return receipt["id"]


def test_sqlite_runtime_snapshot_preserves_baseline_and_reuses_real_bm25_and_packing(tmp_path, local_corpus):
    vault = store(tmp_path)
    identifier = capture_sqlite(vault, local_corpus)
    data = vault.get(identifier, "alice")
    bundle = IncidentBundle.model_validate(data["bundle"])
    assert bundle.origin == "runtime" and bundle.runtime_rag.backend == "sqlite_overlap"
    assert active_capture() is None
    baseline = prepare_replay(bundle, bundle.pipeline, 512)
    assert baseline.messages[1].content == bundle.runtime_rag.generation_input
    assert "42" not in baseline.context
    packed = prepare_replay(bundle, configuration_for(data, "packing"), 512)
    assert "42" in packed.context
    lexical = prepare_replay(bundle, configuration_for(data, "retrieval"), 512)
    assert lexical.config_fingerprint != baseline.config_fingerprint
    assert runtime_context(bundle, bundle.pipeline)[0] == bundle.runtime_rag.packed_context
    assert bundle.runtime_rag.input_tokens is None


def test_capture_cache_bounds_mixed_context_and_changed_answer_are_explicit_holds(tmp_path, local_corpus):
    vault = store(tmp_path)
    capture_sqlite(vault, local_corpus)
    cached = RuntimeCapture("alice", "cached", "test", vault)
    with cached.scope():
        local_rag.search_local_knowledge("telescope request limit authenticated", db_path=local_corpus)
    assert cached.finish({"route": "rag"}, "x")["reason"].startswith("cached_context")
    capture = RuntimeCapture("alice", "mixed", "test", vault)
    capture.before_generation("system", "input", {"route": "hybrid"})
    assert capture.finish({"route": "hybrid"}, "x")["reason"] == "mixed_or_adjudicated_context_not_supported"
    with pytest.raises(ValueError):
        document_payload("oversized " * 1000)


@pytest.mark.asyncio
async def test_real_rag_and_response_nodes_capture_actual_prompt_and_reported_usage(tmp_path, local_corpus, monkeypatch):
    graph = importlib.import_module("agent.research_assistant")
    monkeypatch.setattr(graph, "MODEL_GATEWAY_ENABLED", False)
    monkeypatch.setattr(graph, "GROUNDING_VERIFICATION_ENABLED", False)
    search = local_rag.search_local_knowledge
    monkeypatch.setattr(graph, "search_local_knowledge", lambda *args, **kwargs: search(*args, db_path=local_corpus, **kwargs))
    monkeypatch.setattr(graph, "_get_model", lambda _: RunnableLambda(lambda _: AIMessage(content="42 for authenticated users.",
        usage_metadata={"input_tokens": 23, "output_tokens": 5, "total_tokens": 28})))
    vault = store(tmp_path)
    capture = RuntimeCapture("alice", "node-integration", "labelled-test-double", vault)
    state = {"route": "rag", "query": "telescope request limit authenticated", "messages": [HumanMessage(content="local: telescope limit")]}
    with capture.scope():
        state.update(await graph.rag_agent(state, {"configurable": {"model": "llama-test-double"}}))
        response = await graph.response_agent(state, {"configurable": {"model": "llama-test-double"}})
    answer = response["messages"][-1].content
    receipt = capture.finish(state, answer)
    assert receipt["status"] == "captured"
    private = vault.get(receipt["id"], "alice")["bundle"]
    assert private["runtime_rag"]["input_tokens"] == 23 and private["runtime_rag"]["output_tokens"] == 5
    assert "Local RAG evidence:" in private["runtime_rag"]["generation_input"]
    assert private["model"] == "groq:llama-test-double"


def test_chroma_hybrid_capture_uses_actual_rankings_without_extra_vector_requests(tmp_path, monkeypatch):
    from langchain_core.documents import Document
    calls = []
    docs = [Document(page_content="Telescope request limit is 42.", metadata={"source": "manual", "file_name": "limits", "page": 1}),
            Document(page_content="Unrelated weather notes.", metadata={"source": "weather", "file_name": "weather", "page": 2})]
    class ChromaDouble:
        def similarity_search_with_relevance_scores(self, query, k):
            calls.append((query, k))
            return [(docs[0], .9), (docs[1], .2)]
        def get(self):
            return {"ids": ["first", "second"], "documents": [doc.page_content for doc in docs], "metadatas": [doc.metadata for doc in docs]}
    monkeypatch.setattr(local_rag, "USE_CHROMA_RAG", True)
    monkeypatch.setattr(local_rag, "RAG_ENABLE_LLM_RERANKER", False)
    monkeypatch.setattr(local_rag, "_get_chroma_store", lambda: ChromaDouble())
    monkeypatch.setattr(local_rag, "_chroma_has_documents", lambda _: True)
    vault = store(tmp_path)
    capture = RuntimeCapture("alice", "hybrid-control", "no-provider", vault)
    with capture.scope():
        notes = local_rag._search_chroma_hybrid_knowledge("telescope limit", limit=2)
        capture.before_generation("Evidence only.", "Evidence:\n" + notes, {"route": "rag"})
        capture.model_response("Evidence only.", "Evidence:\n" + notes, AIMessage(content="42"), "groq:test-double")
    receipt = capture.finish({"route": "rag"}, "42")
    assert receipt["status"] == "captured" and len(calls) == 1
    data = vault.get(receipt["id"], "alice")
    bundle = IncidentBundle.model_validate(data["bundle"])
    assert bundle.runtime_rag.vector_ranking[0].score == .9
    assert bundle.runtime_rag.reranker == "production_heuristic"
    assert runtime_context(bundle, bundle.pipeline)[0] == notes
    # Reconstruct RRF using frozen real vector scores; do not re-embed or query Chroma.
    changed = bundle.pipeline.model_copy(update={"retrieval": "hybrid_rrf"})
    runtime_context(bundle, changed)
    assert len(calls) == 1


@pytest.mark.asyncio
async def test_runtime_text_replay_uses_frozen_messages_temperature_caps_and_actual_usage(tmp_path, local_corpus, monkeypatch):
    from types import SimpleNamespace

    from agent.incident_models import InvestigationPlan
    from agent.incident_pipeline import RuntimeTextReplay
    from agent.incident_worker import replay_for
    vault = store(tmp_path)
    identifier = capture_sqlite(vault, local_corpus)
    data = vault.get(identifier, "alice")
    bundle = IncidentBundle.model_validate(data["bundle"])
    seen = []
    class TextDouble:
        def bind(self, **kwargs):
            seen.append(kwargs)
            return self
        async def ainvoke(self, messages):
            seen.append([message.content for message in messages])
            return AIMessage(content="42", usage_metadata={"input_tokens": 7, "output_tokens": 2, "total_tokens": 9})
    def factory(model, **kwargs):
        seen.append((model, kwargs))
        return SimpleNamespace(base=TextDouble())
    monkeypatch.setattr("code_agent.providers.build_provider", factory)
    monkeypatch.setenv("INCIDENT_LIVE_REPLAY_ENABLED", "true")
    monkeypatch.setenv("INCIDENT_WORKER_LIVE_ENABLED", "true")
    replay = replay_for(data, "live_provider")
    assert isinstance(replay, RuntimeTextReplay)
    prepared = prepare_replay(bundle, bundle.pipeline, 512)
    result = await replay.answer(prepared, bundle.pipeline)
    assert result == {"answer": "42", "citations": [], "input_tokens": 7, "output_tokens": 2}
    assert seen[0] == (bundle.model, {"max_completion_tokens": 512, "timeout_seconds": 20})
    assert seen[1] == {"temperature": .2} and seen[2] == [bundle.prompt, bundle.runtime_rag.generation_input]
    from tests.test_incident_investigation import bundle as manual_bundle
    with pytest.raises(ValueError):
        RuntimeTextReplay(manual_bundle(), InvestigationPlan())


def test_runtime_capture_payload_failure_changed_answer_and_missing_inputs_hold(tmp_path, local_corpus):
    vault = store(tmp_path)
    capture = RuntimeCapture("alice", "invalid", "test", vault)
    capture.retrieval({"documents": []})
    assert capture.finish({"route": "rag"}, "answer")["status"] == "held"
    assert capture.finish({"route": "rag"}, "answer")["reason"] == "capture_already_finalized"
    capture = RuntimeCapture("alice", "missing", "test", vault)
    assert capture.finish({"route": "rag"}, "answer")["reason"] == "no_supported_rag_generation"
    capture = RuntimeCapture("alice", "changed", "test", vault)
    with capture.scope():
        local_rag.search_local_knowledge("telescope request limit authenticated", db_path=local_corpus)
        capture.before_generation("system", "input", {"route": "rag"})
        capture.model_response("system", "input", AIMessage(content="original"), "groq:test")
    assert capture.finish({"route": "rag"}, "modified")["reason"] == "post_generation_answer_changed"
