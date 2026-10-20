"""Private SQLite document library and immutable, owner-attested result reviews."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from contextlib import closing
from pathlib import Path
from uuid import uuid4

from agentforge_sdk.models import fingerprint
from agentforge_sdk.multimodal import MultimodalRAG
from agentforge_sdk.spatial import MultimodalDocument


class LibraryError(Exception):
    def __init__(self, status: int, code: str):
        self.status, self.code = status, code


class DocumentLibrary:
    def __init__(self, path: Path, runtime):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path, self.runtime = path, runtime
        with closing(self.connect()) as db, db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS documents (
                    id TEXT PRIMARY KEY, owner TEXT NOT NULL, name TEXT NOT NULL,
                    version TEXT NOT NULL, digest TEXT NOT NULL, ocr INTEGER NOT NULL,
                    created REAL NOT NULL, status TEXT NOT NULL, reason TEXT NOT NULL,
                    original BLOB NOT NULL, manifest TEXT, previews TEXT,
                    UNIQUE(owner, digest, name, ocr));
                CREATE INDEX IF NOT EXISTS document_owner ON documents(owner, created DESC);
                CREATE TABLE IF NOT EXISTS document_results (
                    id TEXT PRIMARY KEY, document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
                    owner TEXT NOT NULL, created REAL NOT NULL, payload TEXT NOT NULL,
                    UNIQUE(document_id, payload));
                CREATE TABLE IF NOT EXISTS document_reviews (
                    result_id TEXT PRIMARY KEY REFERENCES document_results(id) ON DELETE CASCADE,
                    owner TEXT NOT NULL, created REAL NOT NULL, digest TEXT NOT NULL,
                    decision TEXT NOT NULL, note TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS document_lineage (
                    document_id TEXT PRIMARY KEY REFERENCES documents(id) ON DELETE CASCADE,
                    family_id TEXT NOT NULL,
                    parent_id TEXT REFERENCES documents(id) ON DELETE RESTRICT);
                CREATE INDEX IF NOT EXISTS document_family ON document_lineage(family_id);
                INSERT OR IGNORE INTO document_lineage(document_id,family_id,parent_id)
                    SELECT id,id,NULL FROM documents;
                CREATE TABLE IF NOT EXISTS document_comparisons (
                    id TEXT PRIMARY KEY, owner TEXT NOT NULL,
                    before_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
                    after_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
                    created REAL NOT NULL, binding TEXT NOT NULL, checkpoint TEXT,
                    links TEXT NOT NULL, checkpoint_fingerprint TEXT NOT NULL,
                    processed INTEGER NOT NULL, total INTEGER NOT NULL, complete INTEGER NOT NULL,
                    UNIQUE(owner,before_id,after_id,binding));
                CREATE INDEX IF NOT EXISTS comparison_owner ON document_comparisons(owner,after_id,created DESC);
                CREATE TABLE IF NOT EXISTS evidence_sets (
                    id TEXT PRIMARY KEY, owner TEXT NOT NULL, created REAL NOT NULL,
                    binding TEXT NOT NULL, digest TEXT NOT NULL, collection_fingerprint TEXT NOT NULL,
                    UNIQUE(owner,digest));
                CREATE TABLE IF NOT EXISTS evidence_set_sources (
                    set_id TEXT NOT NULL REFERENCES evidence_sets(id) ON DELETE CASCADE,
                    document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE RESTRICT,
                    PRIMARY KEY(set_id,document_id));
                CREATE INDEX IF NOT EXISTS evidence_source_document ON evidence_set_sources(document_id);
                CREATE TABLE IF NOT EXISTS evidence_set_results (
                    id TEXT PRIMARY KEY, set_id TEXT NOT NULL REFERENCES evidence_sets(id) ON DELETE CASCADE,
                    owner TEXT NOT NULL, created REAL NOT NULL, payload TEXT NOT NULL,
                    UNIQUE(set_id,payload));
                CREATE TABLE IF NOT EXISTS evidence_set_reviews (
                    result_id TEXT PRIMARY KEY REFERENCES evidence_set_results(id) ON DELETE CASCADE,
                    owner TEXT NOT NULL, created REAL NOT NULL, digest TEXT NOT NULL,
                    decision TEXT NOT NULL, note TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS evidence_claim_audits (
                    id TEXT PRIMARY KEY, set_id TEXT NOT NULL REFERENCES evidence_sets(id) ON DELETE CASCADE,
                    owner TEXT NOT NULL, created REAL NOT NULL, binding TEXT NOT NULL,
                    digest TEXT NOT NULL, checks TEXT NOT NULL, checkpoint_fingerprint TEXT NOT NULL,
                    UNIQUE(set_id,owner,digest));
                CREATE INDEX IF NOT EXISTS claim_audit_owner ON evidence_claim_audits(owner,set_id,created DESC);
                CREATE TABLE IF NOT EXISTS evidence_claim_replays (
                    id TEXT PRIMARY KEY, source_audit_id TEXT NOT NULL REFERENCES evidence_claim_audits(id) ON DELETE CASCADE,
                    target_set_id TEXT NOT NULL REFERENCES evidence_sets(id) ON DELETE CASCADE,
                    owner TEXT NOT NULL, created REAL NOT NULL, binding TEXT NOT NULL,
                    digest TEXT NOT NULL, entries TEXT NOT NULL, checkpoint_fingerprint TEXT NOT NULL,
                    UNIQUE(owner,digest));
                CREATE INDEX IF NOT EXISTS claim_replay_owner ON evidence_claim_replays(owner,source_audit_id,created DESC);
                CREATE TABLE IF NOT EXISTS evidence_claim_stress (
                    id TEXT PRIMARY KEY, source_audit_id TEXT NOT NULL REFERENCES evidence_claim_audits(id) ON DELETE CASCADE,
                    owner TEXT NOT NULL, created REAL NOT NULL, binding TEXT NOT NULL,
                    digest TEXT NOT NULL, scenarios TEXT NOT NULL, checkpoint_fingerprint TEXT NOT NULL,
                    UNIQUE(owner,digest));
                CREATE INDEX IF NOT EXISTS claim_stress_owner ON evidence_claim_stress(owner,source_audit_id,created DESC);
                CREATE TABLE IF NOT EXISTS document_runtime_lease (
                    slot INTEGER PRIMARY KEY CHECK(slot=1), token TEXT NOT NULL, created REAL NOT NULL);
            """)

    def connect(self):
        db = sqlite3.connect(self.path, timeout=5)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA foreign_keys=ON")
        db.execute("PRAGMA secure_delete=ON")
        return db

    @staticmethod
    def summary(row):
        result = {
            key: row[key]
            for key in (
                "id",
                "name",
                "version",
                "digest",
                "ocr",
                "created",
                "status",
                "reason",
                "family_id",
                "parent_id",
            )
        }
        result["manifest_fingerprint"] = (
            json.loads(row["manifest"])["fingerprint"]
            if "manifest" in row.keys() and row["manifest"]
            else None
        )
        return result

    @staticmethod
    def row(db, owner, document_id, *, previews=False):
        DocumentLibrary.expire(db)
        columns = ",".join(
            "d." + c
            for c in (
                "id",
                "owner",
                "name",
                "version",
                "digest",
                "ocr",
                "created",
                "status",
                "reason",
                "manifest",
            )
        )
        columns += ",v.family_id,v.parent_id" + (",d.previews" if previews else "")
        row = db.execute(
            f"SELECT {columns} FROM documents d JOIN document_lineage v ON v.document_id=d.id WHERE d.id=? AND d.owner=?",
            (document_id, owner),
        ).fetchone()
        if row is None:
            raise LibraryError(404, "document_unavailable")
        return row

    @staticmethod
    def expire(db):
        # Runtime maximum is <150s including cleanup. A crashed request must not hold the slot forever.
        db.execute(
            "UPDATE documents SET status='failed',reason='extraction_interrupted' WHERE status='processing' AND created<?",
            (time.time() - 300,),
        )
        db.execute("DELETE FROM document_runtime_lease WHERE created<?", (time.time() - 300,))

    @staticmethod
    def runtime_busy(db):
        return bool(
            db.execute("SELECT 1 FROM documents WHERE status='processing' LIMIT 1").fetchone()
            or db.execute("SELECT 1 FROM document_runtime_lease WHERE slot=1").fetchone()
        )

    def list(self, owner):
        with closing(self.connect()) as db, db:
            self.expire(db)
            return [
                self.summary(r)
                for r in db.execute(
                    "SELECT d.id,d.name,d.version,d.digest,d.ocr,d.created,d.status,d.reason,d.manifest,v.family_id,v.parent_id FROM documents d JOIN document_lineage v ON v.document_id=d.id WHERE d.owner=? ORDER BY d.created DESC LIMIT 10",
                    (owner,),
                )
            ]

    def ingest(self, owner, raw, *, name, suffix, ocr, parent_id=None):
        digest = hashlib.sha256(raw).hexdigest()
        document_id, version = uuid4().hex, "sha256:" + digest
        with closing(self.connect()) as db, db:
            db.execute("BEGIN IMMEDIATE")
            self.expire(db)
            family_id = document_id
            if parent_id is not None:
                parent = self.row(db, owner, parent_id)
                if not parent["manifest"]:
                    raise LibraryError(409, "parent_version_not_extracted")
                family_id = parent["family_id"]
                if MultimodalDocument.model_validate_json(parent["manifest"]).document_id != family_id:
                    raise LibraryError(409, "document_lineage_invalid")
            existing = db.execute(
                "SELECT d.id,d.name,d.version,d.digest,d.ocr,d.created,d.status,d.reason,v.family_id,v.parent_id FROM documents d JOIN document_lineage v ON v.document_id=d.id WHERE d.owner=? AND d.digest=? AND d.name=? AND d.ocr=?",
                (owner, digest, name, ocr),
            ).fetchone()
            if existing:
                if existing["parent_id"] != parent_id:
                    raise LibraryError(409, "existing_upload_has_different_lineage")
                return self.summary(existing)
            # Bound aggregate storage and concurrency across API threads sharing this local DB.
            counts = db.execute(
                "SELECT COUNT(*), SUM(owner=?), SUM(status=?) FROM documents", (owner, "processing")
            ).fetchone()
            if counts[0] >= 100 or (counts[1] or 0) >= 10:
                raise LibraryError(429, "document_library_full")
            if self.runtime_busy(db):
                raise LibraryError(429, "document_processor_busy")
            db.execute(
                "INSERT INTO documents VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    document_id,
                    owner,
                    name,
                    version,
                    digest,
                    ocr,
                    time.time(),
                    "processing",
                    "extraction_running",
                    raw,
                    None,
                    None,
                ),
            )
            db.execute("INSERT INTO document_lineage VALUES (?,?,?)", (document_id, family_id, parent_id))
        try:
            result = self.runtime.extract(
                raw, name=name, suffix=suffix, document_id=family_id, version=version, ocr=ocr
            )
            manifest = MultimodalDocument.model_validate_json(json.dumps(result["manifest"]))
            status, reason = ("ready", "extracted") if manifest.ready else ("held", "incomplete_extraction")
            with closing(self.connect()) as db, db:
                db.execute(
                    "UPDATE documents SET status=?,reason=?,manifest=?,previews=? WHERE id=? AND owner=?",
                    (
                        status,
                        reason,
                        manifest.model_dump_json(),
                        json.dumps(result["previews"]),
                        document_id,
                        owner,
                    ),
                )
        except Exception:
            with closing(self.connect()) as db, db:
                db.execute(
                    "UPDATE documents SET status=?,reason=? WHERE id=? AND owner=?",
                    ("failed", "isolated_extraction_failed", document_id, owner),
                )
        return self.detail(owner, document_id)["document"]

    def detail(self, owner, document_id):
        with closing(self.connect()) as db, db:
            row = self.row(db, owner, document_id)
            manifest = json.loads(row["manifest"]) if row["manifest"] else None
            results = []
            for result in db.execute(
                "SELECT * FROM document_results WHERE document_id=? AND owner=? ORDER BY created DESC LIMIT 20",
                (document_id, owner),
            ):
                review = db.execute(
                    "SELECT owner,created,digest,decision,note FROM document_reviews WHERE result_id=? AND owner=?",
                    (result["id"], owner),
                ).fetchone()
                results.append(
                    {
                        "id": result["id"],
                        "created": result["created"],
                        "payload": json.loads(result["payload"]),
                        "review": dict(review) if review else None,
                    }
                )
            versions = [
                self.summary(r)
                for r in db.execute(
                    "SELECT d.id,d.name,d.version,d.digest,d.ocr,d.created,d.status,d.reason,v.family_id,v.parent_id FROM documents d JOIN document_lineage v ON v.document_id=d.id WHERE d.owner=? AND v.family_id=? ORDER BY d.created DESC LIMIT 10",
                    (owner, row["family_id"]),
                )
            ]
            return {
                "document": self.summary(row),
                "manifest": manifest,
                "results": results,
                "versions": versions,
                "evidence_set_count": db.execute(
                    "SELECT COUNT(*) FROM evidence_set_sources WHERE document_id=?", (document_id,)
                ).fetchone()[0],
            }

    def preview(self, owner, document_id, page):
        with closing(self.connect()) as db, db:
            row = self.row(db, owner, document_id, previews=True)
            previews = json.loads(row["previews"]) if row["previews"] else []
            if not 1 <= page <= len(previews):
                raise LibraryError(404, "preview_unavailable")
            return {
                "image": "data:image/png;base64," + previews[page - 1],
                "digest": row["digest"],
                "page": page,
            }

    def query(self, owner, document_id, request):
        with closing(self.connect()) as db, db:
            db.execute("BEGIN IMMEDIATE")
            row = self.row(db, owner, document_id)
            if not row["manifest"]:
                raise LibraryError(409, "document_not_extracted")
            manifest = MultimodalDocument.model_validate_json(row["manifest"])
            if request["document_fingerprint"] != manifest.fingerprint:
                raise LibraryError(409, "document_version_changed")
            rag = MultimodalRAG(manifest)
            if request["kind"] == "question":
                receipt = rag.answer(request["question"], top_k=3)
                verified = rag.verify_answer(receipt, question=request["question"])
            else:
                receipt = rag.calculate(request["operation"], request["region_ids"])
                verified = rag.verify_calculation(receipt)
            if not verified:
                raise LibraryError(409, "receipt_verification_failed")
            return self.save_result(db, owner, document_id, request, receipt)

    @staticmethod
    def save_result(db, owner, document_id, request, receipt):
        """Persist only a receipt already verified by the caller's target SDK reader."""
        content = {
            "request": request,
            "receipt": receipt.model_dump(mode="json"),
            "lineage_verified": True,
            "model_calls": 0,
            "mode": "local_extractive",
        }
        content["digest"] = fingerprint(content)
        payload = json.dumps(content, sort_keys=True)
        if len(payload.encode()) > 200_000:
            raise LibraryError(413, "result_budget_exceeded")
        existing = db.execute(
            "SELECT id FROM document_results WHERE document_id=? AND owner=? AND payload=?",
            (document_id, owner, payload),
        ).fetchone()
        if existing:
            return {"id": existing["id"]}
        if (
            db.execute(
                "SELECT COUNT(*) FROM document_results WHERE document_id=?", (document_id,)
            ).fetchone()[0]
            >= 20
        ):
            raise LibraryError(429, "document_result_limit")
        result_id = uuid4().hex
        db.execute(
            "INSERT INTO document_results VALUES (?,?,?,?,?)",
            (result_id, document_id, owner, time.time(), payload),
        )
        return {"id": result_id}

    def review(self, owner, document_id, result_id, *, digest, decision, note):
        with closing(self.connect()) as db, db:
            db.execute("BEGIN IMMEDIATE")
            self.row(db, owner, document_id)
            result = db.execute(
                "SELECT payload FROM document_results WHERE id=? AND document_id=? AND owner=?",
                (result_id, document_id, owner),
            ).fetchone()
            if result is None:
                raise LibraryError(404, "result_unavailable")
            if json.loads(result["payload"])["digest"] != digest:
                raise LibraryError(409, "result_binding_changed")
            existing = db.execute("SELECT * FROM document_reviews WHERE result_id=?", (result_id,)).fetchone()
            if existing:
                if (existing["digest"], existing["decision"], existing["note"]) != (digest, decision, note):
                    raise LibraryError(409, "review_is_immutable")
                return {"recorded": True}
            db.execute(
                "INSERT INTO document_reviews VALUES (?,?,?,?,?,?)",
                (result_id, owner, time.time(), digest, decision, note),
            )
            return {"recorded": True}

    def delete(self, owner, document_id):
        with closing(self.connect()) as db, db:
            row = self.row(db, owner, document_id)
            if row["status"] == "processing":
                raise LibraryError(409, "processing_document_cannot_be_deleted")
            if db.execute(
                "SELECT 1 FROM evidence_set_sources WHERE document_id=? LIMIT 1", (document_id,)
            ).fetchone():
                raise LibraryError(409, "document_used_by_evidence_set")
            if db.execute(
                "SELECT 1 FROM document_lineage WHERE parent_id=? LIMIT 1", (document_id,)
            ).fetchone():
                raise LibraryError(409, "delete_child_versions_first")
            db.execute("DELETE FROM documents WHERE id=? AND owner=?", (document_id, owner))
