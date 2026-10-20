"""Transport controls use a fake HTTPS opener; these tests never call Groq."""

import io
import json
import urllib.error

import pytest

from agentforge_sdk.groq_reasoning import GroqJSONClient, GroqReasoningPlanner, GroqRequestError
from agentforge_sdk.reasoning import PlannerRequest


def client(**kwargs):
    kwargs.setdefault("model", "authored-test-model")
    return GroqJSONClient("authored-test-key-" + "x" * 32, **kwargs)


def install_response(monkeypatch, payload):
    class Opener:
        def open(self, request, timeout):
            assert request.full_url == "https://api.groq.com/openai/v1/chat/completions"
            body = json.loads(request.data)
            assert body["response_format"] == {"type": "json_object"} and body["temperature"] == 0
            return io.BytesIO(json.dumps(payload).encode())
    monkeypatch.setattr("urllib.request.build_opener", lambda *a: Opener())


def test_real_usage_fields_are_preserved_and_cost_not_invented(monkeypatch):
    install_response(monkeypatch, {"choices": [{"message": {"content": '{"action":"abstain"}'}}],
                                    "usage": {"prompt_tokens": 123, "completion_tokens": 8}})
    transport = client(max_requests=1)
    planner = GroqReasoningPlanner(transport)
    result = planner.plan(PlannerRequest(question="Atlas", candidates=(), attempts_remaining=0))
    assert result.usage.input_tokens == 123 and result.usage.output_tokens == 8
    assert result.usage.provider_cost_usd is None and result.usage.provider_requests == 1
    assert transport.requests == 1 and transport.remaining == 0
    assert "authored-test-key" not in repr(transport)
    with pytest.raises(GroqRequestError, match="request_budget_exhausted"):
        planner.plan(PlannerRequest(question="Atlas", candidates=(), attempts_remaining=0))
    assert transport.requests == 1


@pytest.mark.parametrize("code,expected", [(401, "authentication_failed"), (403, "forbidden"), (429, "rate_limited"), (500, "http_500")])
def test_http_errors_count_once_without_retry_or_private_error_text(monkeypatch, code, expected):
    class Opener:
        def open(self, request, timeout):
            raise urllib.error.HTTPError(request.full_url, code, "PRIVATE", {}, io.BytesIO(b"PRIVATE"))
    monkeypatch.setattr("urllib.request.build_opener", lambda *a: Opener())
    transport = client()
    with pytest.raises(GroqRequestError, match=expected) as error:
        transport.complete("Return JSON", {})
    assert "PRIVATE" not in str(error.value) and transport.requests == 1


def test_malformed_usage_and_oversized_responses_fail_closed(monkeypatch):
    install_response(monkeypatch, {"choices": [{"message": {"content": '{}'}}], "usage": {"prompt_tokens": -1}})
    transport = client()
    with pytest.raises(GroqRequestError, match="invalid_or_unavailable_response"):
        transport.complete("Return JSON", {})
    assert transport.requests == 1


def test_configuration_and_prompt_limits_precede_network(monkeypatch):
    for kwargs in ({"max_requests": 21}, {"max_requests": True}, {"timeout_seconds": 0}, {"model": "bad?query"},
                   {"min_request_interval_seconds": 31}, {"min_request_interval_seconds": True}):
        with pytest.raises(ValueError):
            client(**kwargs)
    transport = client()
    with pytest.raises(ValueError):
        transport.complete("Return JSON", {"too_large": "x" * 100_001})
    assert transport.requests == 0


def test_optional_pacing_never_adds_requests_or_retries(monkeypatch):
    install_response(monkeypatch, {"choices": [{"message": {"content": '{}'}}]})
    ticks, sleeps = iter([0.0, 5.0, 15.0]), []
    monkeypatch.setattr("agentforge_sdk.groq_reasoning.time.monotonic", lambda: next(ticks))
    monkeypatch.setattr("agentforge_sdk.groq_reasoning.time.sleep", sleeps.append)
    transport = client(max_requests=2, min_request_interval_seconds=15, reasoning_effort="none")
    transport.complete("Return JSON", {})
    transport.complete("Return JSON", {})
    with pytest.raises(GroqRequestError, match="request_budget_exhausted"):
        transport.complete("Return JSON", {})
    assert sleeps == [10.0] and transport.requests == 2
