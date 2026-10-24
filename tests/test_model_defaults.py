"""The default chat model must be selectable, so an unfunded provider key cannot break every chat."""
import sys
from types import ModuleType, SimpleNamespace

import pytest

from agent import local_rag
from schema import UserInput
from service import service


@pytest.fixture
def both_keys(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("GROQ_API_KEY", "gsk-test")
    monkeypatch.delenv("OPENAI_CHAT_MODEL", raising=False)
    monkeypatch.delenv("GROQ_CHAT_MODEL", raising=False)
    monkeypatch.delenv("DEFAULT_CHAT_MODEL", raising=False)


def test_explicit_default_wins_over_provider_order(both_keys, monkeypatch):
    assert service._default_chat_model() == "gpt-4o-mini"
    monkeypatch.setenv("DEFAULT_CHAT_MODEL", "openai/gpt-oss-120b")
    assert service._default_chat_model() == "openai/gpt-oss-120b"


def test_groq_only_setup_defaults_to_groq(both_keys, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY")
    assert service._default_chat_model() == "openai/gpt-oss-120b"


@pytest.mark.asyncio
async def test_capabilities_list_the_default_model_first(both_keys, monkeypatch):
    monkeypatch.setenv("DEFAULT_CHAT_MODEL", "openai/gpt-oss-120b")
    models = (await service.capabilities())["models"]
    assert [item["id"] for item in models][:2] == ["openai/gpt-oss-120b", "gpt-4o-mini"]


def test_requests_without_a_model_defer_to_the_server():
    assert UserInput(message="hello").model is None


def test_reranker_uses_groq_for_groq_model_ids(monkeypatch):
    built = []

    class FakeGroq:
        def __init__(self, model, temperature):
            built.append(model)

        def invoke(self, _messages):
            return SimpleNamespace(content="2,1")

    module = ModuleType("langchain_groq")
    module.ChatGroq = FakeGroq
    monkeypatch.setitem(sys.modules, "langchain_groq", module)
    monkeypatch.setattr(local_rag, "RAG_ENABLE_LLM_RERANKER", True)
    monkeypatch.setattr(local_rag, "RAG_RERANK_MODEL", "openai/gpt-oss-20b")
    monkeypatch.setenv("GROQ_API_KEY", "gsk-test")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    candidates = [{"file_name": "a.md", "page": 1, "snippet": "first"},
                  {"file_name": "b.md", "page": 1, "snippet": "second"}]

    ranked = local_rag._llm_rerank_candidates("query", candidates, top_k=2)

    assert built == ["openai/gpt-oss-20b"]
    assert [item["file_name"] for item in ranked] == ["b.md", "a.md"]
