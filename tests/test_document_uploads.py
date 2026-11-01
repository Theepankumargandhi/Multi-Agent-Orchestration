"""Files uploaded in the app are indexed per user and only ever searched for their owner."""
import warnings

import pytest
from langchain_core.documents import Document

from agent import local_rag


def _matches(meta, where):
    if not where:
        return True
    if "$and" in where:
        return all(_matches(meta, part) for part in where["$and"])
    return all(meta.get(key) == value for key, value in where.items())


class FakeStore:
    """Chroma's get/add/delete/search with metadata filters; word overlap stands in for vectors."""

    def __init__(self):
        self.rows = {}

    def get(self, where=None, include=None, **_kwargs):
        rows = [(key, text, meta) for key, (text, meta) in self.rows.items() if _matches(meta, where)]
        return {
            "ids": [key for key, _, _ in rows],
            "documents": [text for _, text, _ in rows],
            "metadatas": [meta for _, _, meta in rows],
        }

    def add_texts(self, texts, metadatas, ids):
        self.rows.update({key: (text, meta) for key, text, meta in zip(ids, texts, metadatas)})

    def delete(self, ids):
        for key in ids:
            self.rows.pop(key, None)

    def similarity_search_with_relevance_scores(self, query, k, filter=None):
        words = set(query.lower().split())
        hits = [
            (Document(page_content=text, metadata=meta), len(words & set(text.lower().split())) / len(words))
            for text, meta in self.rows.values()
            if _matches(meta, filter)
        ]
        return sorted(hits, key=lambda hit: -hit[1])[:k]


@pytest.fixture
def stores(monkeypatch):
    shared, uploads = FakeStore(), FakeStore()
    shared.add_texts(
        ["The service streams answers over server-sent events."],
        [{"source": "README.md", "file_name": "README.md", "page": 1}],
        ["shared-1"],
    )
    monkeypatch.setattr(local_rag, "USE_CHROMA_RAG", True)
    monkeypatch.setattr(local_rag, "_get_chroma_store", lambda: shared)
    monkeypatch.setattr(local_rag, "_get_uploads_store", lambda: uploads)
    monkeypatch.setattr(local_rag, "_local_rag_cache", {})
    return shared, uploads


def _pdf(*pages: str) -> bytes:
    """A minimal valid PDF with one line of Helvetica text per page."""
    count = len(pages)
    kids = " ".join(f"{3 + 2 * i} 0 R" for i in range(count))
    font = 3 + 2 * count
    objects = [b"<< /Type /Catalog /Pages 2 0 R >>", f"<< /Type /Pages /Kids [{kids}] /Count {count} >>".encode()]
    for i, text in enumerate(pages):
        stream = f"BT /F1 12 Tf 72 720 Td ({text}) Tj ET".encode()
        objects.append(
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents {4 + 2 * i} 0 R "
            f"/Resources << /Font << /F1 {font} 0 R >> >> >>".encode()
        )
        objects.append(b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"\nendstream")
    objects.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for number, body in enumerate(objects, 1):
        offsets.append(len(out))
        out += b"%d 0 obj\n" % number + body + b"\nendobj\n"
    xref = len(out)
    out += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objects) + 1)
    out += b"".join(b"%010d 00000 n \n" % offset for offset in offsets)
    out += b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (len(objects) + 1, xref)
    return bytes(out)


def test_pdf_pages_become_citable_pages(stores):
    _, uploads = stores
    receipt = local_rag.ingest_upload("alice", "Q3 Report, final.PDF", _pdf("Revenue grew in March", "Costs fell"))

    # The name is safe to cite as "source=<name>, page=<n>".
    assert receipt["file_name"] == "Q3_Report_final.pdf"
    assert receipt["pages"] == 2 and receipt["chunk_count"] == 2 and not receipt["replaced"]
    rows = sorted(uploads.rows.values(), key=lambda row: row[1]["page"])
    assert [(text, meta["page"], meta["owner"]) for text, meta in rows] == [
        ("Revenue grew in March", 1, "alice"),
        ("Costs fell", 2, "alice"),
    ]


def test_markdown_sections_and_text_files(stores):
    _, uploads = stores
    local_rag.ingest_upload("alice", "notes.md", b"# Plan\nShip uploads.\n\n## Risks\nLarge files.\n")
    local_rag.ingest_upload("alice", "todo.txt", "Buy café beans".encode())

    sections = {(meta["file_name"], meta["page"], meta["section"]) for _, meta in uploads.rows.values()}
    assert sections == {("notes.md", 1, "Plan"), ("notes.md", 2, "Risks"), ("todo.txt", 1, "todo.txt")}


def test_uploading_the_same_name_replaces_the_old_version(stores):
    _, uploads = stores
    local_rag.ingest_upload("alice", "notes.md", b"# Old\nfirst version\n")
    receipt = local_rag.ingest_upload("alice", "notes.md", b"# New\nsecond version\n")

    assert receipt["replaced"]
    assert [text for text, _ in uploads.rows.values()] == ["# New\nsecond version"]
    assert [doc["document_id"] for doc in local_rag.list_uploads("alice")] == [receipt["document_id"]]


@pytest.mark.parametrize(
    ("name", "data", "message"),
    [
        ("virus.exe", b"MZ", "Upload a PDF, Markdown or text file."),
        ("scan.pdf", _pdf(""), "No text was found"),
        ("broken.pdf", b"%PDF-1.4 not really", "could not be read"),
        ("blank.md", b"   \n", "No text was found"),
    ],
)
def test_unusable_files_are_rejected_with_a_reason(stores, name, data, message):
    with pytest.raises(local_rag.UploadError, match=message):
        local_rag.ingest_upload("alice", name, data)
    assert stores[1].rows == {}


def test_very_long_files_are_rejected(stores, monkeypatch):
    monkeypatch.setattr(local_rag, "RAG_UPLOAD_MAX_CHUNKS", 2)
    with pytest.raises(local_rag.UploadError, match="too long"):
        local_rag.ingest_upload("alice", "long.md", b"# A\none\n\n# B\ntwo\n\n# C\nthree\n")


def test_listing_and_deleting_only_touch_the_owners_files(stores):
    _, uploads = stores
    mine = local_rag.ingest_upload("alice", "mine.md", b"# Mine\nalice text\n")
    theirs = local_rag.ingest_upload("bob", "theirs.md", b"# Theirs\nbob text\n")

    assert [doc["file_name"] for doc in local_rag.list_uploads("alice")] == ["mine.md"]
    assert local_rag.delete_upload("alice", theirs["document_id"]) == 0
    assert local_rag.delete_upload("alice", mine["document_id"]) == 1
    assert local_rag.list_uploads("alice") == []
    assert [meta["owner"] for _, meta in uploads.rows.values()] == ["bob"]


def test_search_reads_the_shared_docs_and_only_the_askers_uploads(stores):
    local_rag.ingest_upload("alice", "budget.md", b"# Budget\nThe marketing budget is 40000 dollars.\n")

    alice = local_rag.search_local_knowledge("what is the marketing budget", owner="alice")
    bob = local_rag.search_local_knowledge("what is the marketing budget", owner="bob")

    assert "source=budget.md, page=1" in alice and "40000" in alice
    assert "budget.md" not in bob and "40000" not in bob
    # Cached results are kept per owner, so asking again as bob still finds nothing of alice's.
    assert "40000" not in local_rag.search_local_knowledge("what is the marketing budget", owner="bob")
    assert "source=README.md" in local_rag.search_local_knowledge("how are answers streamed", owner="bob")


def test_score_warnings_never_print_uploaded_text(stores, recwarn):
    _, uploads = stores
    search = uploads.similarity_search_with_relevance_scores

    def noisy_search(query, k, filter=None):
        hits = search(query, k, filter)
        warnings.warn(f"Relevance scores must be between 0 and 1, got {hits}")
        return hits

    uploads.similarity_search_with_relevance_scores = noisy_search
    local_rag.ingest_upload("alice", "salaries.md", b"# Pay\nThe CEO salary is 900000.\n")

    assert "salaries.md" in local_rag.search_local_knowledge("CEO salary", owner="alice")
    assert not [warning for warning in recwarn if "900000" in str(warning.message)]


def test_search_with_only_uploads_and_no_shared_docs(stores):
    shared, _ = stores
    shared.rows.clear()
    assert "No documents found yet" in local_rag.search_local_knowledge("anything", owner="alice")

    local_rag.ingest_upload("alice", "budget.md", b"# Budget\nThe marketing budget is 40000 dollars.\n")
    assert "source=budget.md" in local_rag.search_local_knowledge("marketing budget", owner="alice")
