"""Optional native grammars cannot turn runtime retrieval into a downloader."""
from types import SimpleNamespace

from code_agent.code_parsing import CodeParser


def test_missing_cached_grammar_uses_fallback_without_downloading(monkeypatch):
    calls = []

    def forbidden(language):
        calls.append(language)
        raise AssertionError("runtime must not fetch a native grammar")

    pack = SimpleNamespace(downloaded_languages=lambda: [], get_language=forbidden)
    parser = CodeParser()
    monkeypatch.setattr(parser, "_tree_sitter_imports", lambda: (forbidden, pack))
    result = parser.parse("example.py", "def local_symbol(): return 1\n")
    assert result.backend == "python-ast" and result.symbols == ["local_symbol"]
    assert parser.tree_sitter_error == "grammar_not_cached:python"
    assert calls == []


def test_native_loader_error_is_content_free_cached_and_nonfatal(monkeypatch):
    calls = []

    class NativePackError(Exception):
        pass

    def broken(language):
        calls.append(language)
        raise NativePackError("PRIVATE-PATH-OR-URL")

    pack = SimpleNamespace(downloaded_languages=lambda: ["python"], get_language=broken)
    parser = CodeParser()
    monkeypatch.setattr(parser, "_tree_sitter_imports", lambda: (lambda value: value, pack))
    for _ in range(2):
        assert parser.parse("example.py", "def local_symbol(): return 1\n").backend == "python-ast"
    assert parser.tree_sitter_error == "NativePackError"
    assert calls == ["python"]
