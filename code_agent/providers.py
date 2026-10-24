"""Explicit provider selection and validated structured calls, without SDK retries."""
from __future__ import annotations

import os

from pydantic import ValidationError

from code_agent.secret_patterns import require_secret_free_prompt


class StructuredOutputError(ValueError):
    """Content-free terminal schema failure; never retain provider output in errors."""


def resolve_model(identifier: str) -> tuple[str, str]:
    if not identifier or identifier != identifier.strip():
        raise ValueError("a nonempty model identifier without surrounding whitespace is required")
    if ":" in identifier:
        provider, model = identifier.split(":", 1)
        if provider not in {"openai", "groq"} or not model or ":" in model:
            raise ValueError("supported explicit identifiers are openai:<model> and groq:<model>")
        return provider, model
    return ("groq" if identifier.startswith(("llama-", "mixtral-", "gemma-")) else "openai"), identifier


class ValidatedStructuredModel:
    def __init__(self, base, schema):
        self.schema = schema
        # Function calling avoids assuming every provider supports strict JSON Schema.
        # The application still strictly validates the returned action and fields.
        self.runnable = base.with_structured_output(schema, method="function_calling", include_raw=True)

    async def ainvoke(self, messages, config=None):
        require_secret_free_prompt(messages)
        response = await self.runnable.ainvoke(messages, config=config)
        if not isinstance(response, dict) or response.get("parsing_error") is not None or response.get("parsed") is None:
            raise StructuredOutputError("provider did not return a valid structured result")
        calls = getattr(response.get("raw"), "tool_calls", None)
        if not calls or len(calls) != 1 or calls[0].get("name") != self.schema.__name__:
            raise StructuredOutputError("exactly one schema-matching tool result is required")
        # Validate original arguments, not an SDK-parsed object that may coerce values.
        value = calls[0].get("args")
        try:
            if isinstance(value, self.schema):
                value = value.model_dump()
            return self.schema.model_validate(value, strict=True)
        except (ValidationError, TypeError, ValueError):
            raise StructuredOutputError("provider structured result failed validation") from None


class ProviderAdapter:
    def __init__(self, base, provider: str, model: str):
        self.base, self.provider, self.model = base, provider, model

    def with_structured_output(self, schema, **_kwargs):
        return ValidatedStructuredModel(self.base, schema)


def build_provider(identifier: str, *, max_completion_tokens: int | None = None, timeout_seconds: float | None = None) -> ProviderAdapter:
    provider, model = resolve_model(identifier)
    cap = max_completion_tokens if max_completion_tokens is not None else int(os.getenv("CODE_AGENT_MAX_COMPLETION_TOKENS", "4096"))
    timeout = timeout_seconds if timeout_seconds is not None else float(os.getenv("CODE_AGENT_PROVIDER_TIMEOUT_SECONDS", "30"))
    if not 256 <= cap <= 8192 or not 1 <= timeout <= 120:
        raise ValueError("provider completion/timeout settings exceed allowed bounds")
    if provider == "groq":
        from langchain_groq import ChatGroq
        base = ChatGroq(model=model, temperature=0, max_tokens=cap, timeout=timeout, max_retries=0)
    else:
        from langchain_openai import ChatOpenAI
        base = ChatOpenAI(model=model, temperature=0, max_tokens=cap, timeout=timeout, max_retries=0)
    return ProviderAdapter(base, provider, model)
