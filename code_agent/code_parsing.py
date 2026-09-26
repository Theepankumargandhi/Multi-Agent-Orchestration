"""Bounded multi-language parsing with optional Tree-sitter acceleration."""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field
from pathlib import PurePosixPath
from typing import Any

LANGUAGE_BY_SUFFIX = {
    ".py": "python",
    ".pyi": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".mjs": "javascript",
    ".cjs": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".java": "java",
    ".go": "go",
    ".rs": "rust",
    ".sql": "sql",
}

_SYMBOL_NODE_TYPES = {
    "python": {"function_definition", "class_definition"},
    "javascript": {
        "function_declaration", "class_declaration", "method_definition",
        "generator_function_declaration",
    },
    "typescript": {
        "function_declaration", "class_declaration", "method_definition",
        "generator_function_declaration", "interface_declaration", "type_alias_declaration",
        "enum_declaration",
    },
    "java": {
        "class_declaration", "interface_declaration", "enum_declaration",
        "record_declaration", "method_declaration", "constructor_declaration",
    },
    "go": {"function_declaration", "method_declaration", "type_spec"},
    "rust": {
        "function_item", "struct_item", "enum_item", "trait_item", "type_item",
        "union_item", "mod_item",
    },
}

_CALL_NODE_TYPES = {
    "python": {"call"},
    "javascript": {"call_expression", "new_expression"},
    "typescript": {"call_expression", "new_expression"},
    "java": {"method_invocation", "object_creation_expression"},
    "go": {"call_expression"},
    "rust": {"call_expression", "macro_invocation"},
}

_IMPORT_NODE_TYPES = {
    "python": {"import_statement", "import_from_statement"},
    "javascript": {"import_statement", "call_expression"},
    "typescript": {"import_statement", "call_expression"},
    "java": {"import_declaration"},
    "go": {"import_declaration", "import_spec"},
    "rust": {"use_declaration", "extern_crate_declaration"},
}

_SYMBOL_PATTERNS = {
    "javascript": re.compile(
        r"(?:export\s+)?(?:default\s+)?(?:async\s+)?"
        r"(?:function|class|interface|type|enum|const|let|var)\s+([A-Za-z_$][\w$]*)"
    ),
    "typescript": re.compile(
        r"(?:export\s+)?(?:default\s+)?(?:declare\s+)?(?:async\s+)?"
        r"(?:function|class|interface|type|enum|namespace|const|let|var)\s+([A-Za-z_$][\w$]*)"
    ),
    "java": re.compile(
        r"\b(?:class|interface|enum|record)\s+([A-Za-z_$][\w$]*)|"
        r"\b(?:public|protected|private|static|final|synchronized|native|abstract|\s)+"
        r"[\w<>\[\], ?]+\s+([A-Za-z_$][\w$]*)\s*\("
    ),
    "go": re.compile(r"\b(?:func\s+(?:\([^)]*\)\s*)?|type\s+)([A-Za-z_][\w]*)"),
    "rust": re.compile(
        r"\b(?:pub(?:\([^)]*\))?\s+)?(?:async\s+)?"
        r"(?:fn|struct|enum|trait|type|union|mod)\s+([A-Za-z_][\w]*)"
    ),
}

_IMPORT_PATTERNS = {
    "javascript": re.compile(
        r"(?:from\s+|require\s*\(\s*|import\s*\(\s*)['\"]([^'\"]+)['\"]"
    ),
    "typescript": re.compile(
        r"(?:from\s+|require\s*\(\s*|import\s*\(\s*)['\"]([^'\"]+)['\"]"
    ),
    "java": re.compile(r"\bimport\s+(?:static\s+)?([\w.]+)\s*;"),
    "go": re.compile(r"(?:\bimport\s+(?:[\w.]+\s+)?|^\s*(?:[\w.]+\s+)?)\"([^\"]+)\"", re.M),
    "rust": re.compile(r"\b(?:use|extern\s+crate)\s+([^;]+)"),
}

_CALL_PATTERN = re.compile(r"\b([A-Za-z_$][\w$]*)\s*(?:!\s*)?\(")
_CALL_STOPWORDS = {
    "if", "for", "while", "switch", "catch", "return", "sizeof", "typeof",
    "function", "fn", "func", "new", "match", "synchronized",
}


@dataclass(frozen=True)
class SyntaxSpan:
    start_line: int
    end_line: int
    kind: str
    name: str = ""


@dataclass
class ParsedCode:
    language: str
    backend: str
    symbols: list[str] = field(default_factory=list)
    imports: list[str] = field(default_factory=list)
    calls: list[str] = field(default_factory=list)
    spans: list[SyntaxSpan] = field(default_factory=list)
    tree: Any = field(default=None, repr=False, compare=False)
    source_bytes: bytes = field(default=b"", repr=False, compare=False)
    incremental: bool = False


def language_for_path(path: str) -> str:
    return LANGUAGE_BY_SUFFIX.get(PurePosixPath(path).suffix.lower(), "text")


class CodeParser:
    """Use Tree-sitter when installed, with safe built-in fallbacks."""

    def __init__(self, *, prefer_tree_sitter: bool = True, max_nodes: int = 100_000):
        self.prefer_tree_sitter = prefer_tree_sitter
        self.max_nodes = max(1_000, max_nodes)
        self._parsers: dict[str, Any] = {}
        self._tree_sitter_error = ""

    @property
    def tree_sitter_available(self) -> bool:
        return self.prefer_tree_sitter and self._tree_sitter_imports() is not None

    @property
    def tree_sitter_error(self) -> str:
        return self._tree_sitter_error

    def parse(self, path: str, content: str, previous: ParsedCode | None = None) -> ParsedCode:
        language = language_for_path(path)
        if language not in _SYMBOL_NODE_TYPES:
            return _fallback_parse(language, content)
        if self.prefer_tree_sitter:
            try:
                parsed = self._parse_tree_sitter(language, content, previous)
                if parsed is not None:
                    return parsed
            except (ImportError, RuntimeError, TypeError, ValueError, OSError) as exc:
                self._tree_sitter_error = f"{type(exc).__name__}: {exc}"
        return _fallback_parse(language, content)

    def _tree_sitter_imports(self):
        try:
            from tree_sitter import Parser
            from tree_sitter_language_pack import get_language
        except ImportError as exc:
            self._tree_sitter_error = f"{type(exc).__name__}: {exc}"
            return None
        return Parser, get_language

    def _parser(self, language: str):
        if language in self._parsers:
            return self._parsers[language]
        imports = self._tree_sitter_imports()
        if imports is None:
            return None
        Parser, get_language = imports
        parser = Parser(get_language(language))
        self._parsers[language] = parser
        return parser

    def _parse_tree_sitter(
        self,
        language: str,
        content: str,
        previous: ParsedCode | None,
    ) -> ParsedCode | None:
        parser = self._parser(language)
        if parser is None:
            return None
        source = content.encode("utf-8")
        old_tree = None
        incremental = False
        if previous and previous.language == language and previous.tree is not None:
            old_tree = previous.tree.copy() if hasattr(previous.tree, "copy") else None
            if old_tree is not None:
                old_bytes = previous.source_bytes
                if old_bytes:
                    _edit_tree(old_tree, old_bytes, source)
                    incremental = True
                else:
                    old_tree = None
        tree = parser.parse(source, old_tree) if old_tree is not None else parser.parse(source)
        if tree is None:
            return None
        symbols, imports, calls, spans = _extract_tree(
            tree.root_node, source, language, self.max_nodes
        )
        if not imports:
            imports = _extract_imports(language, content)
        result = ParsedCode(
            language=language,
            backend="tree-sitter",
            symbols=_unique(symbols, 200),
            imports=_unique(imports, 200),
            calls=_unique(calls, 500),
            spans=spans[:500],
            tree=tree,
            source_bytes=source,
            incremental=incremental,
        )
        return result


def _extract_tree(root: Any, source: bytes, language: str, max_nodes: int):
    symbols: list[str] = []
    imports: list[str] = []
    calls: list[str] = []
    spans: list[SyntaxSpan] = []
    stack = [root]
    visited = 0
    while stack and visited < max_nodes:
        node = stack.pop()
        visited += 1
        node_type = getattr(node, "type", "")
        if node_type in _SYMBOL_NODE_TYPES[language]:
            name_node = node.child_by_field_name("name")
            name = _node_text(name_node, source) if name_node is not None else ""
            if name:
                symbols.append(name)
            start = getattr(node, "start_point", (0, 0))[0] + 1
            end = getattr(node, "end_point", (start - 1, 0))[0] + 1
            spans.append(SyntaxSpan(start, end, node_type, name))
        if node_type in _CALL_NODE_TYPES[language]:
            call_node = None
            for field_name in ("function", "name", "type"):
                call_node = node.child_by_field_name(field_name)
                if call_node is not None:
                    break
            value = _node_text(call_node, source).split(".")[-1] if call_node is not None else ""
            value = re.sub(r"[^A-Za-z0-9_$]", "", value)
            if value and value not in _CALL_STOPWORDS:
                calls.append(value)
        if node_type in _IMPORT_NODE_TYPES[language]:
            imports.extend(
                _imports_from_statement(language, _node_text(node, source))
            )
        stack.extend(reversed(getattr(node, "children", [])))
    return symbols, imports, calls, spans


def _node_text(node: Any, source: bytes) -> str:
    return source[node.start_byte:node.end_byte].decode("utf-8", errors="replace")


def _edit_tree(tree: Any, old: bytes, new: bytes) -> None:
    old_text = old.decode("utf-8")
    new_text = new.decode("utf-8")
    prefix = 0
    common = min(len(old_text), len(new_text))
    while prefix < common and old_text[prefix] == new_text[prefix]:
        prefix += 1
    suffix = 0
    while (
        suffix < common - prefix
        and old_text[-1 - suffix] == new_text[-1 - suffix]
    ):
        suffix += 1
    start_byte = len(old_text[:prefix].encode("utf-8"))
    old_end = len(old_text[: len(old_text) - suffix].encode("utf-8"))
    new_end = len(new_text[: len(new_text) - suffix].encode("utf-8"))
    tree.edit(
        start_byte=start_byte,
        old_end_byte=old_end,
        new_end_byte=new_end,
        start_point=_point(old, start_byte),
        old_end_point=_point(old, old_end),
        new_end_point=_point(new, new_end),
    )


def _point(value: bytes, offset: int) -> tuple[int, int]:
    before = value[:offset]
    row = before.count(b"\n")
    last = before.rfind(b"\n")
    return row, len(before) if last < 0 else len(before) - last - 1


def _fallback_parse(language: str, content: str) -> ParsedCode:
    symbols: list[str] = []
    imports: list[str] = []
    calls: list[str] = []
    spans: list[SyntaxSpan] = []
    if language == "python":
        try:
            tree = ast.parse(content)
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    symbols.append(node.name)
                    spans.append(
                        SyntaxSpan(node.lineno, getattr(node, "end_lineno", node.lineno), type(node).__name__, node.name)
                    )
                elif isinstance(node, ast.Import):
                    imports.extend(alias.name for alias in node.names)
                elif isinstance(node, ast.ImportFrom):
                    imports.append(("." * node.level) + (node.module or ""))
                elif isinstance(node, ast.Call):
                    if isinstance(node.func, ast.Name):
                        calls.append(node.func.id)
                    elif isinstance(node.func, ast.Attribute):
                        calls.append(node.func.attr)
        except (SyntaxError, ValueError, MemoryError, RecursionError):
            pass
    else:
        pattern = _SYMBOL_PATTERNS.get(language)
        if pattern:
            for match in pattern.finditer(content):
                name = next((group for group in match.groups() if group), "")
                if name:
                    symbols.append(name)
                    line = content.count("\n", 0, match.start()) + 1
                    spans.append(SyntaxSpan(line, line, "pattern_symbol", name))
        imports = _extract_imports(language, content)
        calls = [item for item in _CALL_PATTERN.findall(content) if item not in _CALL_STOPWORDS]
    return ParsedCode(
        language=language,
        backend=(
            "python-ast"
            if language == "python"
            else "bounded-pattern"
            if language in _SYMBOL_PATTERNS
            else "lexical-only"
        ),
        symbols=_unique(symbols, 200),
        imports=_unique(imports, 200),
        calls=_unique(calls, 500),
        spans=spans[:500],
    )


def _extract_imports(language: str, content: str) -> list[str]:
    if language == "python":
        return _imports_from_statement(language, content)
    pattern = _IMPORT_PATTERNS.get(language)
    return pattern.findall(content) if pattern else []


def _imports_from_statement(language: str, content: str) -> list[str]:
    if language == "python":
        matches = re.findall(
            r"(?m)^\s*(?:from\s+([.\w]+)\s+import|import\s+([.\w]+))",
            content,
        )
        return [left or right for left, right in matches if left or right]
    pattern = _IMPORT_PATTERNS.get(language)
    return pattern.findall(content) if pattern else []


def _unique(values: list[str], limit: int) -> list[str]:
    return list(dict.fromkeys(value.strip() for value in values if value.strip()))[:limit]
