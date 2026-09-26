"""Minimal OpenAI-compatible client for evaluating locally served adapters."""

from __future__ import annotations

import asyncio
import json
import os
from dataclasses import dataclass
from urllib.parse import urlparse
from urllib.request import Request, urlopen


@dataclass(frozen=True)
class InferenceResult:
    answer: str
    prompt_tokens: int
    completion_tokens: int
    model: str
    tool_calls: tuple[str, ...] = ()


def validate_inference_url(base_url: str, allow_remote: bool = False) -> str:
    parsed = urlparse(base_url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("inference base URL must be an absolute HTTP(S) URL")
    local_hosts = {"127.0.0.1", "localhost", "::1", "host.docker.internal"}
    if not allow_remote and parsed.hostname.casefold() not in local_hosts:
        raise ValueError("remote inference endpoints require allow_remote=true")
    return base_url.rstrip("/")


async def generate(
    *,
    base_url: str,
    model: str,
    prompt: str,
    api_key_env: str = "LOCAL_MODEL_API_KEY",
    timeout_seconds: float = 120,
    max_tokens: int = 512,
    temperature: float = 0.0,
    allow_remote: bool = False,
) -> InferenceResult:
    base_url = validate_inference_url(base_url, allow_remote)
    endpoint = base_url if base_url.endswith("/chat/completions") else f"{base_url}/chat/completions"
    payload = json.dumps(
        {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
    ).encode()
    headers = {"Content-Type": "application/json"}
    api_key = os.getenv(api_key_env, "").strip()
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = Request(endpoint, data=payload, headers=headers, method="POST")

    def execute() -> dict:
        with urlopen(request, timeout=max(1.0, min(timeout_seconds, 600.0))) as response:
            return json.loads(response.read().decode("utf-8"))

    response = await asyncio.to_thread(execute)
    choices = response.get("choices") or []
    if not choices:
        raise ValueError("inference response contains no choices")
    message = choices[0].get("message") or {}
    answer = str(message.get("content") or "").strip()
    tool_calls = tuple(
        str((item.get("function") or {}).get("name") or "").strip()
        for item in (message.get("tool_calls") or [])
        if str((item.get("function") or {}).get("name") or "").strip()
    )
    usage = response.get("usage") or {}
    return InferenceResult(
        answer=answer,
        prompt_tokens=int(usage.get("prompt_tokens") or 0),
        completion_tokens=int(usage.get("completion_tokens") or 0),
        model=str(response.get("model") or model),
        tool_calls=tool_calls,
    )
