"""Exercise request bounds before handlers receive attacker-controlled bytes."""

import asyncio
import base64
import json
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace
from unittest.mock import AsyncMock
from urllib.error import HTTPError

import pytest

from service.request_limits import RequestBodyLimitMiddleware


def test_oversized_trace_header_is_rejected_without_quadratic_parsing():
    # Keep the hostile input in an owned subprocess so a dependency regression
    # fails on a deadline rather than blocking the test runner's event loop.
    probe = (
        "import logging; logging.disable(logging.CRITICAL); "
        "from opentelemetry.trace import TraceState; "
        "assert list(TraceState.from_header(['vendor=value'])) == ['vendor']; "
        "assert not list(TraceState.from_header(['vendor=value' + ' ' * 1000000 + 'x']))"
    )
    subprocess.run([sys.executable, '-I', '-c', probe], check=True,
                   capture_output=True, timeout=5)


@pytest.mark.asyncio
@pytest.mark.parametrize("headers", [[], [(b"content-length", b"1")]])
async def test_chunked_or_false_length_is_rejected_before_handler(headers):
    downstream = AsyncMock()
    middleware = RequestBodyLimitMiddleware(downstream, lambda _: 8)
    receive = AsyncMock(side_effect=[
        {"type": "http.request", "body": b"123456", "more_body": True},
        {"type": "http.request", "body": b"789", "more_body": False},
    ])
    send = AsyncMock()
    await middleware({"type": "http", "method": "POST", "headers": headers}, receive, send)
    downstream.assert_not_awaited()
    assert send.await_args_list[0].args[0]["status"] == 413


@pytest.mark.asyncio
async def test_bounded_chunked_body_is_delivered_exactly_once():
    received = []

    async def app(scope, receive, send):
        received.append(await receive())

    receive = AsyncMock(side_effect=[
        {"type": "http.request", "body": b"12", "more_body": True},
        {"type": "http.request", "body": b"34", "more_body": False},
    ])
    await RequestBodyLimitMiddleware(app, lambda _: 4)({"type": "http", "method": "POST"}, receive, AsyncMock())
    assert received == [{"type": "http.request", "body": b"1234", "more_body": False}]


@pytest.mark.asyncio
async def test_slow_request_has_a_total_body_deadline():
    async def receive():
        await asyncio.sleep(1)
    app, send = AsyncMock(), AsyncMock()
    await RequestBodyLimitMiddleware(app, lambda _: 4, timeout_seconds=0.01)(
        {"type": "http", "method": "POST"}, receive, send)
    app.assert_not_awaited()
    assert send.await_args_list[0].args[0]["status"] == 408


@pytest.mark.parametrize("payload", [[], None, {"sub": "alice", "exp": "NaN"},
                                    {"sub": "../alice", "exp": 9999999999}])
def test_invalid_signed_token_shapes_fail_closed(payload):
    from service import service
    encoded = base64.urlsafe_b64encode(json.dumps(payload).encode()).decode().rstrip("=")
    assert service._verify_access_token(encoded + "." + service._sign_token(encoded)) is None


def test_non_ascii_token_cannot_crash_authentication():
    from service import service
    assert service._verify_access_token("payload.\u0100") is None


def test_unknown_paths_cannot_expand_rate_or_metrics_keys():
    from service import service
    assert len({service._rate_limit_path(f"/code/tasks/{i}") for i in range(200)}) == 1
    assert service._resolve_metric_path(SimpleNamespace(scope={}, url=SimpleNamespace(path="/PRIVATE-ID"))) == "unmatched"


def test_development_does_not_use_the_published_signing_key():
    from service import service
    if not service._CONFIGURED_USER_AUTH_SECRET:
        assert service.USER_AUTH_SECRET != "dev-insecure-user-auth-secret-change-me"
        assert len(service.USER_AUTH_SECRET) >= 32


@pytest.mark.asyncio
@pytest.mark.parametrize('secret', ['replace-with-at-least-32-random-characters',
                                   'dev-insecure-user-auth-secret-change-me', ' ' * 40])
@pytest.mark.parametrize('user_auth', [True, False])
async def test_production_refuses_public_or_blank_authentication_secrets(monkeypatch, secret, user_auth):
    from service import service

    monkeypatch.setattr(service, 'APP_ENV', 'production')
    monkeypatch.setattr(service, 'ENABLE_USER_AUTH', user_auth)
    monkeypatch.setattr(service, '_CONFIGURED_USER_AUTH_SECRET', secret)
    monkeypatch.setattr(service, 'USER_AUTH_SECRET', secret)
    monkeypatch.setenv('AUTH_SECRET', secret)
    checkpoint = AsyncMock(side_effect=AssertionError('unsafe startup reached persistence'))
    monkeypatch.setattr(service, '_open_checkpointer', checkpoint)
    with pytest.raises(RuntimeError, match='strong'):
        async with service.lifespan(service.app):
            pytest.fail('unsafe production authentication was accepted')
    checkpoint.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize('user', ['', 'ordinary-user'])
async def test_global_evaluation_reports_require_explicit_admin_grant(monkeypatch, user):
    from unittest.mock import Mock

    from fastapi import HTTPException

    from service import service
    monkeypatch.setattr(service, 'ENABLE_EVAL_API', True)
    monkeypatch.setenv('EVAL_ADMIN_USER_IDS', 'operator')
    store = Mock()
    monkeypatch.setattr(service, 'ExperimentStore', store)
    request = SimpleNamespace(state=SimpleNamespace(user_id=user))
    with pytest.raises(HTTPException) as denied:
        await service.list_evaluation_experiments(request)
    assert denied.value.status_code in {401, 403}
    with pytest.raises(HTTPException):
        await service.get_evaluation_experiment('00000000-0000-0000-0000-000000000000', request)
    store.assert_not_called()


@pytest.mark.asyncio
async def test_authorized_evaluation_listing_does_not_expose_host_paths(monkeypatch):
    from unittest.mock import Mock

    from service import service
    monkeypatch.setattr(service, 'ENABLE_EVAL_API', True)
    monkeypatch.setenv('EVAL_ADMIN_USER_IDS', 'operator')
    store = Mock()
    store.return_value.list.return_value = [{'experiment_id': 'authored', 'report_path': '/private/path'}]
    monkeypatch.setattr(service, 'ExperimentStore', store)
    result = await service.list_evaluation_experiments(SimpleNamespace(state=SimpleNamespace(user_id='operator')))
    assert result == {'experiments': [{'experiment_id': 'authored'}]}


@pytest.mark.asyncio
async def test_inference_redirect_cannot_forward_credentials(monkeypatch):
    from post_training.inference import generate
    hits = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            self.rfile.read(int(self.headers.get("Content-Length", "0")))
            hits.append(self.path)
            self.send_response(302)
            self.send_header("Location", "/credential-sink")
            self.end_headers()

        def do_GET(self):
            hits.append(self.path)
            self.send_response(200)
            self.end_headers()

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    monkeypatch.setenv("AUDIT_LOCAL_TOKEN", "synthetic-not-a-credential")
    try:
        with pytest.raises(HTTPError):
            await generate(base_url=f"http://127.0.0.1:{server.server_port}", model="fixture",
                           prompt="authored", api_key_env="AUDIT_LOCAL_TOKEN")
        assert hits == ["/chat/completions"]
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
