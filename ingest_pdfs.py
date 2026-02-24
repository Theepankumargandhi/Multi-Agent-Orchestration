import argparse
from dotenv import load_dotenv

from agent.local_rag import ingest_pdfs_to_chroma


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest local PDFs into ChromaDB for RAG.")
    parser.add_argument(
        "--pdf-dir",
        default="rag_docs",
        help="Directory containing PDF files (searched recursively).",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Delete existing ChromaDB data before ingesting.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1000,
        help="Chunk size in characters.",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=150,
        help="Chunk overlap in characters.",
    )
    args = parser.parse_args()

    load_dotenv()
    summary = ingest_pdfs_to_chroma(
        pdf_dir=args.pdf_dir,
        reset=args.reset,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )
    status = "OK" if summary.get("ok") else "FAILED"
    print(f"[{status}] {summary.get('message', '')}")
    for key in ("pdf_count", "page_count", "chunk_count", "persist_dir", "collection"):
        if key in summary:
            print(f"{key}: {summary[key]}")


if __name__ == "__main__":
    main()
