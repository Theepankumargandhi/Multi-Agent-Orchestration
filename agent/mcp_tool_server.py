import math
import re
import sys
import importlib.util
from pathlib import Path

import numexpr
from mcp.server.fastmcp import FastMCP

# Ensure imports work when this file is launched directly as a script.
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Load tools.py directly to avoid importing agent package side effects.
TOOLS_PATH = REPO_ROOT / "agent" / "tools.py"
_tools_spec = importlib.util.spec_from_file_location("agent_tools_standalone", TOOLS_PATH)
if _tools_spec is None or _tools_spec.loader is None:
    raise RuntimeError(f"Failed to load tools module from {TOOLS_PATH}")
_tools_module = importlib.util.module_from_spec(_tools_spec)
_tools_spec.loader.exec_module(_tools_module)
perform_web_search = getattr(_tools_module, "perform_web_search")


server = FastMCP(
    name="agent-service-toolkit-tools",
    instructions=(
        "MCP tool server for agent-service-toolkit. "
        "Provides web_search and calculator tools."
    ),
)


@server.tool(name="web_search")
def web_search(
    query: str,
    max_results: int = 5,
    recency_days: int | None = None,
    relevance_query: str | None = None,
) -> str:
    """
    Search the web and return normalized, source-linked result text.
    """
    result = perform_web_search(
        query=query,
        max_results=max_results,
        recency_days=recency_days,
        relevance_query=relevance_query,
        return_meta=False,
    )
    return str(result or "").strip()


@server.tool(name="calculator")
def calculator(expression: str) -> str:
    """
    Evaluate a numeric expression with numexpr.
    """
    local_dict = {"pi": math.pi, "e": math.e}
    output = str(
        numexpr.evaluate(
            expression.strip(),
            global_dict={},
            local_dict=local_dict,
        )
    )
    return re.sub(r"^\[|\]$", "", output).strip()


if __name__ == "__main__":
    server.run(transport="stdio")
