import argparse
import asyncio
import json
import os
import shlex
import sys
from typing import Any


def _extract_tool_text(call_tool_result: Any) -> str:
    contents = getattr(call_tool_result, "content", None) or []
    lines: list[str] = []
    for item in contents:
        text = getattr(item, "text", None)
        if isinstance(text, str) and text.strip():
            lines.append(text.strip())
    if lines:
        return "\n".join(lines).strip()
    return str(call_tool_result or "").strip()


def _split_extra_args(extra_args: str) -> list[str]:
    if not extra_args:
        return []
    try:
        return shlex.split(extra_args, posix=(os.name != "nt"))
    except ValueError:
        return [extra_args]


async def _call_tool(
    tool_name: str,
    arguments: dict[str, Any],
    server_command: str,
    server_script: str,
    server_args: str,
) -> str:
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    args = [server_script]
    args.extend(_split_extra_args(server_args))
    server_params = StdioServerParameters(command=server_command, args=args)

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.call_tool(tool_name, arguments)
            return _extract_tool_text(result)


def _emit(payload: dict[str, Any], exit_code: int) -> int:
    print(json.dumps(payload, ensure_ascii=False))
    return exit_code


def main() -> int:
    parser = argparse.ArgumentParser(description="MCP bridge client")
    parser.add_argument("--tool-name", required=True)
    parser.add_argument("--arguments-json", required=True)
    parser.add_argument("--server-command", required=True)
    parser.add_argument("--server-script", required=True)
    parser.add_argument("--server-args", default="")
    args = parser.parse_args()

    try:
        arguments = json.loads(args.arguments_json)
        if not isinstance(arguments, dict):
            return _emit({"ok": False, "error": "arguments-json must decode to an object"}, 2)
    except json.JSONDecodeError as e:
        return _emit({"ok": False, "error": f"Invalid arguments-json: {e}"}, 2)

    try:
        result = asyncio.run(
            _call_tool(
                tool_name=args.tool_name,
                arguments=arguments,
                server_command=args.server_command,
                server_script=args.server_script,
                server_args=args.server_args,
            )
        )
        return _emit({"ok": True, "result": result}, 0)
    except Exception as e:
        return _emit({"ok": False, "error": str(e)}, 1)


if __name__ == "__main__":
    sys.exit(main())
