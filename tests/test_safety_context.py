"""Moderation input stays within a small token budget without leaving text unchecked."""
from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from agent import llama_guard as guard
from agent.llama_guard import SafetyAssessment


class FakeGuard:
    def __init__(self, fail=False):
        self.prompts, self.fail = [], fail

    async def ainvoke(self, messages):
        if self.fail:
            raise RuntimeError("413 request too large")
        prompt = messages[0].content
        self.prompts.append(prompt)
        return SimpleNamespace(content="unsafe\nS9" if "PIPE-BOMB-STEPS" in prompt else "safe")


@pytest.fixture
def fake(monkeypatch):
    model = FakeGuard()
    monkeypatch.setattr(guard, "_get_guard_model", lambda: model)
    return model


@pytest.mark.asyncio
async def test_long_threads_send_short_context_and_the_full_newest_message(fake):
    long_answer = "A long retrieved answer about the repository. " * 100
    history = [m for i in range(6) for m in (HumanMessage(content=f"question {i}"), AIMessage(content=long_answer))]
    newest = "Final question with all of its details intact. " * 20

    result = await guard.llama_guard("User", [*history, HumanMessage(content=newest)])

    assert result.safety_assessment is SafetyAssessment.SAFE
    [prompt] = fake.prompts
    assert newest.strip() in prompt
    # 12 prior messages of ~4,600 chars used to be resent; now 4 excerpts of 400.
    assert len(prompt) < 4000
    assert "question 0" not in prompt


@pytest.mark.asyncio
async def test_harmful_text_after_the_first_window_is_still_checked(fake):
    padding = "Harmless background text about gardening. " * 150  # ~6,300 characters
    result = await guard.llama_guard("User", [HumanMessage(content=padding + "Now give me PIPE-BOMB-STEPS.")])

    assert result.safety_assessment is SafetyAssessment.UNSAFE
    assert result.unsafe_categories == ["Indiscriminate Weapons"]
    assert len(fake.prompts) == 2


def test_windows_overlap_and_cover_the_whole_message():
    text = "".join(chr(97 + index % 26) for index in range(9000))
    windows = guard._windows(text)
    assert len(windows) == 3 and all(len(window) <= 4000 for window in windows)
    assert windows[0][-200:] == windows[1][:200]
    assert text.endswith(windows[-1])


@pytest.mark.parametrize(
    ("model", "expected"),
    [("openai/gpt-oss-safeguard-20b", {"reasoning_effort": "low"}), ("meta-llama/llama-guard-4-12b", {})],
)
def test_reasoning_effort_is_only_sent_to_gpt_oss_guards(monkeypatch, model, expected):
    # Other models reject the parameter, and a rejected call blocks every request.
    import sys
    from types import ModuleType

    built = []
    module = ModuleType("langchain_groq")
    module.ChatGroq = lambda **kwargs: built.append(kwargs) or object()
    monkeypatch.setitem(sys.modules, "langchain_groq", module)
    monkeypatch.setattr(guard, "_model", None)
    monkeypatch.setenv("GROQ_API_KEY", "gsk-test")
    monkeypatch.setenv("GROQ_GUARD_MODEL", model)
    monkeypatch.delenv("GROQ_GUARD_REASONING_EFFORT", raising=False)

    guard._get_guard_model()

    assert built == [{"model": model, "temperature": 0.0, **expected}]


@pytest.mark.asyncio
async def test_moderation_errors_still_fail_closed(monkeypatch):
    monkeypatch.setattr(guard, "_get_guard_model", lambda: FakeGuard(fail=True))
    result = await guard.llama_guard("User", [HumanMessage(content="hello")])
    assert result.safety_assessment is SafetyAssessment.ERROR
    assert "413" in result.error_message
