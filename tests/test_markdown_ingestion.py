"""Repository Markdown ingestion stores cite-able sections without a provider call."""
import pytest

from agent import local_rag


class FakeStore:
    def __init__(self):
        self.rows = {}

    def get(self, where=None, **_kwargs):
        rows = [(key, meta) for key, (_, meta) in self.rows.items() if not where or meta["source"] == where["source"]]
        return {"ids": [key for key, _ in rows], "metadatas": [meta for _, meta in rows]}

    def add_texts(self, texts, metadatas, ids):
        self.rows.update({key: (text, meta) for key, text, meta in zip(ids, texts, metadatas)})

    def delete(self, ids):
        for key in ids:
            self.rows.pop(key, None)


@pytest.fixture
def store(monkeypatch, tmp_path):
    fake = FakeStore()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(local_rag, "_get_chroma_store", lambda: fake)
    monkeypatch.setattr(local_rag, "_reset_local_rag_cache", lambda: None)
    return fake


def test_each_heading_section_is_a_citable_page(store, tmp_path):
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "guide.md").write_text(
        "# Guide\nIntro text.\n\n## Install\nRun the installer.\n\n## Usage\nCall the API.\n", encoding="utf-8"
    )
    (tmp_path / "docs" / "notes.txt").write_text("not markdown", encoding="utf-8")

    summary = local_rag.ingest_markdown_to_chroma(["docs/guide.md", "docs/notes.txt"])

    assert summary["ok"] and summary["file_count"] == 1 and summary["section_count"] == 3
    sections = {meta["section"]: meta["page"] for _, meta in store.rows.values()}
    assert sections == {"Guide": 1, "Install": 2, "Usage": 3}
    assert {meta["source"] for _, meta in store.rows.values()} == {"docs/guide.md"}


def test_reingesting_removes_deleted_sections(store, tmp_path):
    doc = tmp_path / "README.md"
    doc.write_text("# A\nfirst\n\n## B\nsecond\n", encoding="utf-8")
    local_rag.ingest_markdown_to_chroma(["README.md"])
    doc.write_text("# A\nfirst\n", encoding="utf-8")

    summary = local_rag.ingest_markdown_to_chroma(["README.md"])

    # Chunk IDs include the file hash, so every chunk of the old version is replaced.
    assert summary["stale_chunks_removed"] == 2
    assert [meta["section"] for _, meta in store.rows.values()] == ["A"]


def test_no_markdown_files_is_reported_without_touching_the_store(store):
    summary = local_rag.ingest_markdown_to_chroma(["missing.md"])
    assert not summary["ok"] and store.rows == {}


def test_reingesting_a_folder_removes_deleted_files(store, tmp_path):
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "kept.md").write_text("# Kept\nstill here\n", encoding="utf-8")
    (tmp_path / "docs" / "gone.md").write_text("# Gone\nremoved later\n", encoding="utf-8")
    (tmp_path / "notes.md").write_text("# Notes\noutside the folder\n", encoding="utf-8")
    local_rag.ingest_markdown_to_chroma(["docs/kept.md", "docs/gone.md", "notes.md"])
    (tmp_path / "docs" / "gone.md").unlink()

    summary = local_rag.ingest_markdown_to_chroma(["docs/kept.md"], prune_roots=["docs"])

    assert summary["deleted_files_removed"] == 1
    # Files outside the scanned folder are left alone.
    assert {meta["source"] for _, meta in store.rows.values()} == {"docs/kept.md", "notes.md"}


def test_warm_up_loads_the_index_and_never_raises(monkeypatch, tmp_path):
    searched = []
    monkeypatch.setattr(local_rag, "CHROMA_PERSIST_DIR", str(tmp_path / "missing"))
    assert local_rag.warm_up() is False  # never creates an index
    monkeypatch.setattr(local_rag, "CHROMA_PERSIST_DIR", str(tmp_path))

    class WarmStore:
        def similarity_search(self, query, k):
            searched.append((query, k))

    monkeypatch.setattr(local_rag, "USE_CHROMA_RAG", True)
    monkeypatch.setattr(local_rag, "_get_chroma_store", WarmStore)
    assert local_rag.warm_up() and searched == [("warm up", 1)]

    def broken():
        raise RuntimeError("index unavailable")

    monkeypatch.setattr(local_rag, "_get_chroma_store", broken)
    assert local_rag.warm_up() is False
    monkeypatch.setattr(local_rag, "USE_CHROMA_RAG", False)
    assert local_rag.warm_up() is False
