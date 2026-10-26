"""Ingest this repository's Markdown documentation into ChromaDB for RAG.

With RAG_EMBEDDING_BACKEND=local the embeddings run on this machine for free; with
"openai" they use OPENAI_API_KEY. Re-running replaces changed sections and removes deleted ones.
"""

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[2]
# The service resolves relative store paths such as chroma_db from the repository
# root, and agent.local_rag reads its settings at import, so load .env first.
os.chdir(REPO_ROOT)
load_dotenv()
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from agent.local_rag import ingest_markdown_to_chroma  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest README.md and docs/ into ChromaDB for RAG.")
    parser.add_argument(
        "paths",
        nargs="*",
        help="Markdown files or directories relative to the repository (default: README.md docs).",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Delete existing ChromaDB data, including ingested PDFs, before ingesting.",
    )
    args = parser.parse_args()

    files = []
    for target in [Path(item) for item in args.paths or ["README.md", "docs"]]:
        files.extend(sorted(target.rglob("*.md")) if target.is_dir() else [target])
    summary = ingest_markdown_to_chroma([path.as_posix() for path in files], reset=args.reset)
    status = "OK" if summary.get("ok") else "FAILED"
    print(f"[{status}] {summary.get('message', '')}")
    for key in ("file_count", "section_count", "chunk_count", "stale_chunks_removed", "persist_dir", "collection"):
        if key in summary:
            print(f"{key}: {summary[key]}")
    if not summary.get("ok"):
        sys.exit(1)


if __name__ == "__main__":
    main()
