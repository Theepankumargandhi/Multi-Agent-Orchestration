"""Importing the service package must not start the application."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

_BOUNDARY = """
import importlib.abc
import sys

class BlockApplicationDependencies(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "service.service" or fullname.split(".")[0] in {
            "dotenv", "agent", "chromadb", "groq", "openai", "langchain",
            "langchain_core", "langchain_community", "langchain_groq",
            "langchain_openai", "langgraph", "langsmith", "mcp",
            "opentelemetry", "psycopg", "redis", "uvicorn",
            "pdfplumber", "pypdfium2", "pypdf",
        }:
            raise AssertionError("unexpected application dependency: " + fullname)

sys.meta_path.insert(0, BlockApplicationDependencies())
"""


def run_isolated(script: str) -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-I",
            "-B",
            "-c",
            "import sys; sys.path.insert(0, sys.argv[1]);\n" + script,
            str(ROOT),
        ],
        cwd=ROOT,
        env={**os.environ, "PYTHON_DOTENV_DISABLED": "1", "LANGSMITH_TRACING": "false"},
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_package_import_does_not_start_the_application():
    run_isolated(
        _BOUNDARY
        + """
import service
assert service.__all__ == ["app"]
assert "app" not in vars(service)
assert "service.service" not in sys.modules
"""
    )


def test_public_app_export_remains_compatible_and_is_cached():
    run_isolated("""
from types import ModuleType
import service
application = object()
module = ModuleType("service.service")
module.app = application
sys.modules[module.__name__] = module
from service import app
assert app is application
assert service.app is application
assert vars(service)["app"] is application
""")


def test_unknown_attributes_do_not_load_the_application():
    run_isolated(
        _BOUNDARY
        + """
import service
try:
    service.missing_attribute
except AttributeError as exc:
    assert "missing_attribute" in str(exc)
else:
    raise AssertionError("unknown attribute was accepted")
assert "service.service" not in sys.modules
"""
    )


def test_run_service_loads_env_before_importing_the_app():
    # Agent modules read settings at import; loading .env afterwards silently ignored
    # RAG_EMBEDDING_BACKEND, GROUNDING_VERIFICATION_ENABLED and the model names.
    import ast

    tree = ast.parse((ROOT / "run_service.py").read_text(encoding="utf-8"))
    load_line = next(
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and getattr(node.func, "id", "") == "load_dotenv"
    )
    app_import_line = next(
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module == "service"
    )
    assert load_line < app_import_line
