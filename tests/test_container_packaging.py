"""Catch missing internal modules in selectively copied runtime images."""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


def _source_modules() -> dict[str, Path]:
    result = {}
    for package in ("agent", "client", "evals", "schema", "service"):
        for path in (ROOT / package).rglob("*.py"):
            parts = path.relative_to(ROOT).with_suffix("").parts
            module = ".".join(parts[:-1] if parts[-1] == "__init__" else parts)
            result[module] = path
    return result


@pytest.mark.parametrize("name", ["service"])
def test_runtime_image_copies_its_internal_python_dependencies(name):
    """Static COPY/import closure; complements, but does not replace, image builds."""
    dockerfile = (ROOT / "docker" / f"Dockerfile.{name}").read_text(encoding="utf-8")
    sources = []
    for source in re.findall(r"^COPY\s+(\S+)\s+\S+\s*$", dockerfile, re.MULTILINE):
        path = ROOT / source
        if path.is_dir():
            sources.extend(path.rglob("*.py"))
        elif path.suffix == ".py":
            sources.append(path)
    available = {path.resolve() for path in sources}
    modules = _source_modules()
    missing = set()
    for source in sources:
        for node in ast.walk(ast.parse(source.read_bytes(), filename=str(source))):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                names = [node.module, *(f"{node.module}.{alias.name}" for alias in node.names)]
            else:
                continue
            for module in names:
                if module in modules and modules[module].resolve() not in available:
                    missing.add(f"{source.relative_to(ROOT).as_posix()} imports {module}")
    assert not missing, "Runtime image omits internal dependencies:\n" + "\n".join(sorted(missing))


def test_docker_context_excludes_private_inputs_and_generated_outputs():
    patterns = set((ROOT / ".dockerignore").read_text(encoding="utf-8").splitlines())
    assert {
        ".env",
        ".env.*",
        "*.key",
        "*.pem",
        "data",
        ".runtime_logs",
        "*.egg-info",
    } <= patterns


def test_development_compose_publishes_only_loopback_ports():
    import yaml

    services = yaml.safe_load((ROOT / "compose.yaml").read_text(encoding="utf-8"))["services"]
    published = [port for service in services.values() for port in service.get("ports", [])]
    assert published
    assert all(isinstance(port, str) and port.startswith("127.0.0.1:") for port in published)


def test_compose_keeps_data_in_a_named_volume():
    import yaml

    compose = yaml.safe_load((ROOT / "compose.yaml").read_text(encoding="utf-8"))
    assert compose["services"]["agent_service"]["volumes"] == ["citeright_data:/app/data"]
    assert "citeright_data" in compose["volumes"]
