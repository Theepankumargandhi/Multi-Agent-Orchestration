"""Explicit local document ingestion. No .env, remote model, upload or persistence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from agentforge_sdk import MultimodalRAG, TesseractOCR, load_multimodal


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path)
    parser.add_argument("--question", required=True)
    parser.add_argument("--ocr", action="store_true", help="Use caller-installed local Tesseract (English)")
    args = parser.parse_args()
    document = load_multimodal(args.path, document_id="local-document", ocr=TesseractOCR() if args.ocr else None)
    rag = MultimodalRAG(document)
    answer = rag.answer(args.question)
    # This explicitly requested stdout includes document excerpts. Do not publish private output.
    print(json.dumps({"pages": [{"page": p.page, "status": p.status, "reason": p.reason} for p in document.pages],
                      "answer": answer.model_dump(mode="json"), "receipt_verified": rag.verify_answer(answer, question=args.question)}, indent=2))


if __name__ == "__main__":
    main()
