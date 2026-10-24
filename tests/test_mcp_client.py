import asyncio
import sys

import pytest

from agent import mcp_client


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(mcp_client, "MCP_TOOL_SERVER_COMMAND", sys.executable)
    return mcp_client._PersistentMCPClient()


@pytest.mark.asyncio
async def test_real_tool_server_session_closes_from_a_different_task(client):
    # The service connects inside a request task and closes during shutdown.
    assert await asyncio.create_task(client.call("calculator", {"expression": "2**10"})) == "1024"
    await asyncio.create_task(client.close())
    assert client._session is None


@pytest.mark.asyncio
async def test_session_reconnects_after_close(client):
    assert await client.call("calculator", {"expression": "3*7"}) == "21"
    await client.close()
    assert await client.call("calculator", {"expression": "6*7"}) == "42"
    await client.close()


@pytest.mark.asyncio
async def test_missing_server_script_fails_without_spawning(client, monkeypatch):
    monkeypatch.setattr(mcp_client, "MCP_TOOL_SERVER_SCRIPT", "does-not-exist.py")
    with pytest.raises(RuntimeError, match="script not found"):
        await client.call("calculator", {"expression": "1+1"})
    assert client._session is None
