"""Bounded, non-executing AST impact analysis against a pinned migration catalog."""
from __future__ import annotations

import ast
from pathlib import Path

from code_agent.upgrade_models import ImpactFinding, MigrationPlan, MigrationSpec, fingerprint

GUIDE_COMMIT = "5c842dfc9c245fb37aa1f5ec5b55c1aed10bd7e6"
GUIDE_SHA256 = "3a33c005259e6ede170df1904a168a4a64e8d8efc5b7fed360b65e5c000c05b7"
GUIDE = f"https://github.com/pydantic/pydantic/blob/{GUIDE_COMMIT}/docs/migration.md"
RULES = {
    "validators": ("semantic", "changes-to-validators", "Decorator field/config parameters become ValidationInfo; review validation order and exceptions."),
    "field-options": ("mechanical", "changes-to-pydanticfield", "Rename regex to pattern and review removed Field options; verify constraints."),
    "optional-default": ("semantic", "required-optional-and-nullable-fields", "Nullable annotations no longer imply an omitted default."),
    "union-coercion": ("semantic", "unions", "V2 preserves matching input types; review explicit union mode when preserving V1 coercion."),
    "config": ("semantic", "changes-to-config", "Review renamed configuration and removed settings."),
    "model-methods": ("mechanical", "changes-to-pydanticbasemodel", "Replace deprecated model methods; verify serialization."),
    "v1-namespace": ("unsupported", "continue-using-pydantic-v1-features", "Compatibility imports do not complete a native V2 migration."),
}


def source_files(root: Path, spec: MigrationSpec) -> list[Path]:
    root = root.resolve()
    result, size = set(), 0
    for name in spec.source_paths:
        selected = root / name
        if not selected.exists() or selected.is_symlink() or not selected.resolve().is_relative_to(root):
            raise ValueError("source scope is unavailable or escapes the repository")
        for path in ([selected] if selected.is_file() else selected.rglob("*.py")):
            if path.suffix != ".py":
                continue
            if path.is_symlink() or not path.resolve().is_relative_to(root):
                raise ValueError("source links are not supported")
            if path not in result:
                size += path.stat().st_size
                result.add(path)
            if len(result) > 500 or size > 2_000_000 or path.stat().st_size > 200000:
                raise ValueError("migration source scope exceeds the first-release analysis budget")
    return sorted(result)


def plan_migration(root: Path, spec: MigrationSpec) -> MigrationPlan:
    root = root.resolve()
    findings, reasons, hashes = [], [], {}
    for path in source_files(root, spec):
        name = path.relative_to(root).as_posix()
        data = path.read_bytes()
        import hashlib
        hashes[name] = hashlib.sha256(data).hexdigest()
        try:
            tree = ast.parse(data.decode("utf-8"), filename=name)
        except (SyntaxError, UnicodeError):
            reasons.append("unparseable_source:" + name)
            continue
        aliases = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                for imported in node.names:
                    aliases[imported.asname or imported.name] = node.module + "." + imported.name
            elif isinstance(node, ast.Import):
                for imported in node.names:
                    aliases[imported.asname or imported.name] = imported.name
        def symbol(node):
            if isinstance(node, ast.Name):
                return aliases.get(node.id, node.id)
            if isinstance(node, ast.Attribute):
                return symbol(node.value) + "." + node.attr
            return ""
        for node in ast.walk(tree):
            rule = None
            if isinstance(node, ast.Call):
                called = symbol(node.func)
                if called in {"pydantic.validator", "pydantic.root_validator"}:
                    rule = "validators"
                elif called == "pydantic.Field" and any(kw.arg in {"regex", "min_items", "max_items", "allow_mutation", "const", "unique_items"} for kw in node.keywords):
                    rule = "field-options"
                elif isinstance(node.func, ast.Attribute) and node.func.attr in {"dict", "json", "parse_obj", "parse_raw", "from_orm", "schema"}:
                    rule = "model-methods"
            elif isinstance(node, ast.AnnAssign) and node.value is None and isinstance(node.annotation, ast.Subscript) and symbol(node.annotation.value) == "typing.Optional":
                rule = "optional-default"
            elif isinstance(node, ast.Subscript) and symbol(node.value) == "typing.Union":
                rule = "union-coercion"
            elif isinstance(node, ast.ClassDef) and node.name == "Config":
                rule = "config"
            elif isinstance(node, ast.ImportFrom) and (node.module or "").startswith("pydantic.v1"):
                rule = "v1-namespace"
            if rule:
                severity, anchor, _ = RULES[rule]
                findings.append(ImpactFinding(path=name, line=node.lineno, symbol=getattr(node, "name", "") or ast.unparse(node)[:160],
                    rule_id=rule, severity=severity, source_url=GUIDE + "#" + anchor))
            if len(findings) > 1000:
                reasons.append("too_many_impact_findings")
                break
    if not hashes:
        reasons.append("no_python_sources_in_scope")
    if any(row.severity == "unsupported" for row in findings):
        reasons.append("existing_v1_compatibility_namespace_requires_owner_review")
    return MigrationPlan(spec_fingerprint=fingerprint(spec.model_dump()), source_fingerprint=fingerprint(hashes),
        catalog_fingerprint=fingerprint({"source_sha256": GUIDE_SHA256, "commit": GUIDE_COMMIT, "rules": RULES}),
        findings=sorted(findings, key=lambda row: (row.path, row.line, row.rule_id)), affected_files=sorted({row.path for row in findings}),
        status="held" if reasons else "planned", hold_reasons=sorted(set(reasons)))
