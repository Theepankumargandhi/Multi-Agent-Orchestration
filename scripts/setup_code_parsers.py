"""Explicitly provision native grammars before offline code retrieval/testing.

This setup command may download parser binaries. Runtime indexing never calls it.
"""
from __future__ import annotations


def main() -> None:
    import tree_sitter_language_pack as pack

    languages = ["python", "typescript", "javascript", "java", "go", "rust"]
    download = getattr(pack, "download", None)
    if download is not None:
        download(languages)
    # Fail setup if an installed or downloaded grammar cannot actually load.
    for language in languages:
        pack.get_language(language)
    print("Verified local grammars: " + ", ".join(languages))


if __name__ == "__main__":
    main()
