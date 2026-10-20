"""Explicit, bounded, read-only adapters for existing text/PDF/SQLite/Chroma evidence."""

from __future__ import annotations

import hashlib
import io
import sqlite3
from contextlib import closing
from pathlib import Path

from agentforge_sdk.models import DocumentInputError, DocumentPage


def _file_bytes(path: Path, limit: int) -> bytes:
    path = Path(path).absolute()
    for part in (path, *path.parents):
        if part.is_symlink() or (part.exists() and getattr(part.lstat(), "st_file_attributes", 0) & 0x400):
            raise DocumentInputError("document links/reparse points are not supported")
    if not path.is_file() or not 0 < path.stat().st_size <= limit:
        raise DocumentInputError("document is absent or exceeds its byte budget")
    # The file may grow after stat(); bound the read itself, not just the preflight.
    with path.open("rb") as stream:
        data = stream.read(limit + 1)
    if len(data) > limit:
        raise DocumentInputError("document changed beyond its byte budget")
    return data


def load_text(
    path: Path, *, document_id: str, source: str | None = None, version="unversioned"
) -> tuple[DocumentPage, ...]:
    """One explicit UTF-8 text/Markdown file; no recursion or implicit page number."""
    path = Path(path)
    if path.suffix.lower() not in {".txt", ".md"}:
        raise DocumentInputError("text loader accepts explicit .txt or .md files only")
    try:
        text = _file_bytes(path, 200_000).decode("utf-8")
        return (
            DocumentPage(
                document_id=document_id,
                text=text,
                source=source or path.name,
                title=path.name,
                version=version,
            ),
        )
    except (OSError, ValueError, UnicodeError):
        raise DocumentInputError("text document could not be loaded within the declared profile") from None


def load_pdf(
    path: Path, *, document_id: str, source: str | None = None, version="unversioned", max_pages=200
) -> tuple[DocumentPage, ...]:
    """Extract embedded PDF text only. This is not OCR, geometry recovery or a parser sandbox."""
    if type(max_pages) is not int or not 1 <= max_pages <= 1000:
        raise ValueError("PDF page limit must be between 1 and 1000")
    path = Path(path)
    if path.suffix.lower() != ".pdf":
        raise DocumentInputError("PDF loader requires an explicit .pdf file")
    try:
        from pypdf import PdfReader
    except ImportError:
        raise DocumentInputError("PDF extraction requires the optional sdk[pdf] dependencies") from None
    try:
        raw = _file_bytes(path, 10_000_000)
        if not raw.startswith(b"%PDF-"):
            raise ValueError("PDF header is absent")
        reader = PdfReader(io.BytesIO(raw), strict=True)
        if reader.is_encrypted or not 1 <= len(reader.pages) <= max_pages:
            raise ValueError("encrypted or over-budget PDF")
        pages, total = [], 0
        for number, page in enumerate(reader.pages, start=1):
            text = (page.extract_text() or "").strip()
            if not text:
                # Do not silently discard image-only/blank pages and claim full-document coverage.
                raise ValueError("a page has no embedded text")
            total += len(text.encode())
            if total > 5_000_000:
                raise ValueError("PDF extracted text budget exceeded")
            pages.append(
                DocumentPage(
                    document_id=document_id,
                    text=text,
                    source=source or path.name,
                    title=path.name,
                    version=version,
                    page=number,
                )
            )
        return tuple(pages)
    except Exception:
        raise DocumentInputError(
            "PDF extraction failed or contains unsupported/empty pages; no OCR was attempted"
        ) from None


def load_sqlite(path: Path, *, max_documents=200) -> tuple[DocumentPage, ...]:
    """Read the existing local_rag docs table; never seed or migrate the database."""
    if type(max_documents) is not int or not 1 <= max_documents <= 1000:
        raise ValueError("SQLite document limit must be between 1 and 1000")
    path = Path(path)
    try:
        # Validate bounds/links without loading a potentially large database into memory.
        if path.stat().st_size > 50_000_000:
            raise DocumentInputError("SQLite database exceeds the supported byte budget")
        for item in (path.absolute(), *path.absolute().parents):
            if item.is_symlink() or (
                item.exists() and getattr(item.lstat(), "st_file_attributes", 0) & 0x400
            ):
                raise DocumentInputError("SQLite links/reparse points are unsupported")
        with closing(sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)) as connection:
            connection.execute("PRAGMA query_only=ON")
            connection.execute("PRAGMA trusted_schema=OFF")
            rows = connection.execute(
                "SELECT id, title, content, source FROM docs ORDER BY id LIMIT ?", (max_documents + 1,)
            ).fetchall()
        if len(rows) > max_documents or sum(len(str(row[2]).encode()) for row in rows) > 5_000_000:
            raise ValueError("SQLite corpus budget exceeded")
        return tuple(
            DocumentPage(document_id=f"sqlite:{row[0]}", title=row[1], text=row[2], source=row[3])
            for row in rows
        )
    except (OSError, sqlite3.Error, ValueError):
        raise DocumentInputError(
            "local-RAG database could not be read in the supported read-only profile"
        ) from None


def load_chroma_records(payload: dict) -> tuple[DocumentPage, ...]:
    """Adapt an already-fetched Chroma get() payload. No store/client or network call."""
    try:
        if not isinstance(payload, dict):
            raise ValueError("expected a data-only Chroma payload")
        ids, texts, metadata = payload.get("ids"), payload.get("documents"), payload.get("metadatas")
        if (
            not all(isinstance(value, list) for value in (ids, texts, metadata))
            or not len(ids) == len(texts) == len(metadata)
            or len(ids) > 1000
            or any(not isinstance(value, str) or not value or len(value) > 1000 for value in ids)
            or len(ids) != len(set(ids))
        ):
            raise ValueError("incomplete or ambiguous Chroma records")
        pages = []
        for identifier, text, meta in zip(ids, texts, metadata, strict=True):
            if not isinstance(meta, dict):
                raise ValueError("Chroma metadata is missing")
            page = meta.get("page")
            # Existing AgentForge ingestion stores one-based page metadata.
            if page is not None and (type(page) is not int or page < 1):
                raise ValueError("one-based page metadata is required")
            pages.append(
                DocumentPage(
                    document_id="chroma:" + hashlib.sha256(identifier.encode()).hexdigest()[:32],
                    text=text,
                    source=meta.get("source", "chroma-record"),
                    title=meta.get("file_name", ""),
                    version=meta.get("file_hash", "unversioned"),
                    page=page,
                )
            )
        if sum(len(page.text.encode()) for page in pages) > 5_000_000:
            raise ValueError("Chroma corpus byte budget exceeded")
        return tuple(pages)
    except (ValueError, TypeError, AttributeError):
        raise DocumentInputError(
            "Chroma records are outside the bounded AgentForge metadata profile"
        ) from None
