"""An unavailable optional cache must not introduce an unbounded socket wait."""
import sys
from types import SimpleNamespace

import pytest

from agent import graph_rag, local_rag


@pytest.mark.parametrize("module", [local_rag, graph_rag], ids=["local-rag", "graph-rag"])
def test_redis_setup_has_finite_deadlines_and_retains_failover(monkeypatch, module):
    calls = []

    def connect(url, **options):
        calls.append(options)
        assert options["socket_connect_timeout"] == 2.0
        assert options["socket_timeout"] == 2.0
        raise TimeoutError("authored cache failure")

    monkeypatch.setenv("CACHE_USE_REDIS", "true")
    monkeypatch.setenv("REDIS_URL", "redis://unused.invalid")
    monkeypatch.setattr(module, "_redis_client", None)
    monkeypatch.setattr(module, "_redis_disabled", False)
    monkeypatch.setitem(sys.modules, "redis", SimpleNamespace(Redis=SimpleNamespace(from_url=connect)))
    assert module._get_redis_client() is None
    assert module._redis_disabled
    assert module._get_redis_client() is None and len(calls) == 1
    assert calls == [{"decode_responses": True, "socket_connect_timeout": 2.0, "socket_timeout": 2.0}]
