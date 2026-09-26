"""Persistent MCP stdio client used by graph tool nodes."""

import asyncio
import json
import os
import shlex
import subprocess
import sys
from contextlib import AsyncExitStack
from pathlib import Path
from typing import Any


def _default_sidecar_python() -> str:
    repo_root = Path(__file__).resolve().parent.parent
    candidates = [repo_root / ".venv" / "Scripts" / "python.exe", repo_root / ".venv" / "bin" / "python"]
    return str(next((path for path in candidates if path.exists()), Path(sys.executable)))


MCP_TOOLS_ENABLED = os.getenv("MCP_TOOLS_ENABLED", "false").strip().lower() in {"1", "true", "yes", "on"}
MCP_TOOL_SERVER_COMMAND = os.getenv("MCP_TOOL_SERVER_COMMAND", _default_sidecar_python()).strip()
MCP_TOOL_SERVER_SCRIPT = os.getenv(
    "MCP_TOOL_SERVER_SCRIPT", str(Path(__file__).resolve().parent / "mcp_tool_server.py")
).strip()
MCP_TOOL_SERVER_ARGS = os.getenv("MCP_TOOL_SERVER_ARGS", "").strip()
MCP_BRIDGE_COMMAND = os.getenv("MCP_BRIDGE_COMMAND", MCP_TOOL_SERVER_COMMAND).strip()
MCP_BRIDGE_SCRIPT = os.getenv(
    "MCP_BRIDGE_SCRIPT", str(Path(__file__).resolve().parent / "mcp_bridge_client.py")
).strip()
MCP_CALL_TIMEOUT_SECONDS = max(1, int(os.getenv("MCP_CALL_TIMEOUT_SECONDS", "20")))
MCP_TRANSPORT = os.getenv("MCP_TRANSPORT", "persistent").strip().lower()


def _content_to_text(result: Any) -> str:
    blocks = getattr(result, "content", None) or []
    parts = [str(getattr(block, "text", "")).strip() for block in blocks]
    return "\n".join(part for part in parts if part).strip()


class _PersistentMCPClient:
    def __init__(self) -> None:
        self._stack: AsyncExitStack | None = None
        self._session: Any = None
        self._lock: asyncio.Lock | None = None

    def _get_lock(self) -> asyncio.Lock:
        if self._lock is None:
            self._lock = asyncio.Lock()
        return self._lock

    async def _connect(self) -> None:
        if self._session is not None:
            return
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client

        script = Path(MCP_TOOL_SERVER_SCRIPT)
        if not script.exists():
            raise RuntimeError(f"MCP tool server script not found: {script}")
        args = [str(script), *shlex.split(MCP_TOOL_SERVER_ARGS, posix=os.name != "nt")]
        stack = AsyncExitStack()
        try:
            read, write = await stack.enter_async_context(
                stdio_client(StdioServerParameters(command=MCP_TOOL_SERVER_COMMAND, args=args))
            )
            session = await stack.enter_async_context(ClientSession(read, write))
            await session.initialize()
        except Exception:
            await stack.aclose()
            raise
        self._stack, self._session = stack, session

    async def call(self, tool_name: str, arguments: dict[str, Any]) -> str:
        async with self._get_lock():
            await self._connect()
            try:
                result = await asyncio.wait_for(
                    self._session.call_tool(tool_name, arguments=arguments), timeout=MCP_CALL_TIMEOUT_SECONDS
                )
            except Exception:
                await self.close()
                raise
            if getattr(result, "isError", False):
                raise RuntimeError(_content_to_text(result) or f"MCP tool failed: {tool_name}")
            return _content_to_text(result)

    async def close(self) -> None:
        stack, self._stack, self._session = self._stack, None, None
        if stack is not None:
            await stack.aclose()


_persistent_client = _PersistentMCPClient()


async def close_mcp_client() -> None:
    await _persistent_client.close()


async def _call_mcp_tool_via_bridge(
    tool_name: str, arguments: dict[str, Any]
) -> tuple[str | None, str | None]:
    bridge_path = Path(MCP_BRIDGE_SCRIPT)
    if not bridge_path.exists():
        return None, f"MCP bridge script not found: {bridge_path}"
    cmd = [
        MCP_BRIDGE_COMMAND, str(bridge_path), "--tool-name", tool_name,
        "--arguments-json", json.dumps(arguments, ensure_ascii=False),
        "--server-command", MCP_TOOL_SERVER_COMMAND, "--server-script", MCP_TOOL_SERVER_SCRIPT,
        "--server-args", MCP_TOOL_SERVER_ARGS,
    ]
    try:
        completed = await asyncio.to_thread(
            subprocess.run, cmd, capture_output=True, text=True,
            timeout=MCP_CALL_TIMEOUT_SECONDS, check=False,
        )
    except subprocess.TimeoutExpired:
        return None, f"MCP tool call timed out after {MCP_CALL_TIMEOUT_SECONDS}s ({tool_name})"
    except Exception as exc:
        return None, f"MCP bridge spawn failed: {exc}"
    stdout, stderr = (completed.stdout or "").strip(), (completed.stderr or "").strip()
    if completed.returncode != 0:
        return None, f"MCP bridge failed ({tool_name}): {stderr or stdout or completed.returncode}"
    for line in reversed(stdout.splitlines()):
        try:
            result = json.loads(line)
        except json.JSONDecodeError:
            continue
        if result.get("ok"):
            return str(result.get("result", "")).strip(), None
        return None, str(result.get("error") or f"Unknown MCP bridge error ({tool_name})")
    return None, f"MCP bridge returned non-JSON output ({tool_name})"


async def _call_mcp_tool(
    tool_name: str, arguments: dict[str, Any]
) -> tuple[str | None, str | None]:
    if not MCP_TOOLS_ENABLED:
        return None, "MCP tools are disabled."
    try:
        if MCP_TRANSPORT == "bridge":
            return await _call_mcp_tool_via_bridge(tool_name, arguments)
        return await _persistent_client.call(tool_name, arguments), None
    except Exception as exc:
        return None, f"MCP tool failed ({tool_name}): {exc}"


async def mcp_web_search(
    query: str, max_results: int = 5, recency_days: int | None = None,
    relevance_query: str | None = None,
) -> tuple[str | None, str | None]:
    return await _call_mcp_tool(
        "web_search",
        {"query": query, "max_results": int(max_results), "recency_days": recency_days,
         "relevance_query": relevance_query},
    )


async def mcp_calculator(expression: str) -> tuple[str | None, str | None]:
    return await _call_mcp_tool("calculator", {"expression": expression})
