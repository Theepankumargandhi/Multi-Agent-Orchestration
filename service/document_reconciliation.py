"""Private, snapshot-bound numeric evidence sets. Conflicts never acquire a winning value."""

from __future__ import annotations

import json
import time
from contextlib import closing
from uuid import uuid4

from agentforge_sdk.models import fingerprint
from agentforge_sdk.multimodal import MultimodalRAG
from agentforge_sdk.reconciliation import (
    EvidenceSet,
    EvidenceSource,
    FactCalculation,
    FactQuery,
    FactReconciliation,
    ReconciliationConfig,
)
from agentforge_sdk.spatial import MultimodalDocument
from service.document_store import DocumentLibrary, LibraryError


class DocumentReconciliation:
    def __init__(self, library: DocumentLibrary):
        self.library = library

    @staticmethod
    def _config():
        # Browser callers cannot weaken coverage or count one artifact as corroboration.
        return ReconciliationConfig(
            min_distinct_artifacts=2,
            require_complete_coverage=True,
            max_total_regions=2000,
            max_total_text_chars=500_000,
        )

    def _evidence(self, db, owner, binding):
        scope = " | ".join(binding["scope"][field] for field in ("entity", "period", "basis"))
        sources, headers = [], []
        for item in binding["sources"]:
            row = self.library.row(db, owner, item["document_id"])
            if not row["manifest"]:
                raise LibraryError(409, "evidence_source_not_extracted")
            document = MultimodalDocument.model_validate_json(row["manifest"])
            if document.fingerprint != item["fingerprint"]:
                raise LibraryError(409, "evidence_source_changed")
            sources.append(EvidenceSource(row["id"], scope, MultimodalRAG(document)))
            headers.append(self.library.summary(row))
        return EvidenceSet(sources, config=self._config()), scope, headers

    def _load(self, db, owner, set_id):
        row = db.execute("SELECT * FROM evidence_sets WHERE id=? AND owner=?", (set_id, owner)).fetchone()
        if row is None:
            raise LibraryError(404, "evidence_set_unavailable")
        binding = json.loads(row["binding"])
        if fingerprint(binding) != row["digest"]:
            raise LibraryError(409, "evidence_set_binding_changed")
        evidence, scope, sources = self._evidence(db, owner, binding)
        if evidence.fingerprint != row["collection_fingerprint"]:
            raise LibraryError(409, "evidence_set_context_changed")
        return row, binding, evidence, scope, sources

    def create(self, owner, request):
        with closing(self.library.connect()) as db, db:
            db.execute("BEGIN IMMEDIATE")
            binding = {
                "name": request["name"],
                "scope": request["scope"],
                "scope_confirmed": True,
                "sources": sorted(request["sources"], key=lambda s: s["document_id"]),
            }
            evidence, _, _ = self._evidence(db, owner, binding)
            digest = fingerprint(binding)
            existing = db.execute(
                "SELECT id FROM evidence_sets WHERE owner=? AND digest=?", (owner, digest)
            ).fetchone()
            if existing:
                return {"id": existing["id"]}
            counts = db.execute("SELECT COUNT(*),SUM(owner=?) FROM evidence_sets", (owner,)).fetchone()
            if counts[0] >= 100 or (counts[1] or 0) >= 20:
                raise LibraryError(429, "evidence_set_limit")
            set_id = uuid4().hex
            db.execute(
                "INSERT INTO evidence_sets VALUES (?,?,?,?,?,?)",
                (
                    set_id,
                    owner,
                    time.time(),
                    json.dumps(binding, sort_keys=True),
                    digest,
                    evidence.fingerprint,
                ),
            )
            db.executemany(
                "INSERT INTO evidence_set_sources VALUES (?,?)",
                [(set_id, s["document_id"]) for s in binding["sources"]],
            )
            return {"id": set_id}

    def list(self, owner):
        with closing(self.library.connect()) as db:
            rows = db.execute(
                "SELECT id,created,binding FROM evidence_sets WHERE owner=? ORDER BY created DESC LIMIT 20",
                (owner,),
            )
            return [
                {"id": r["id"], "created": r["created"], "name": json.loads(r["binding"])["name"]}
                for r in rows
            ]

    @staticmethod
    def _result(db, owner, set_id, result_id):
        row = db.execute(
            "SELECT * FROM evidence_set_results WHERE id=? AND set_id=? AND owner=?",
            (result_id, set_id, owner),
        ).fetchone()
        if row is None:
            raise LibraryError(404, "evidence_result_unavailable")
        payload = json.loads(row["payload"])
        if payload["digest"] != fingerprint({k: v for k, v in payload.items() if k != "digest"}):
            raise LibraryError(409, "evidence_result_changed")
        return row, payload

    def _verify(self, db, owner, set_id, payload, evidence):
        if payload["collection_fingerprint"] != evidence.fingerprint:
            raise LibraryError(409, "evidence_result_context_changed")
        if payload["kind"] == "fact":
            receipt = FactReconciliation.model_validate_json(json.dumps(payload["receipt"]))
            valid = evidence.verify_fact(receipt)
        else:
            receipt = FactCalculation.model_validate_json(json.dumps(payload["receipt"]))
            facts = []
            for result_id in payload["request"]["fact_ids"]:
                _, parent = self._result(db, owner, set_id, result_id)
                if parent["kind"] != "fact":
                    raise LibraryError(409, "expected_saved_fact")
                facts.append(self._verify(db, owner, set_id, parent, evidence))
            valid = evidence.verify_calculation(receipt, facts=facts)
        if not valid:
            raise LibraryError(409, "evidence_receipt_unverifiable")
        return receipt

    @staticmethod
    def _save(db, owner, set_id, kind, request, receipt, evidence):
        payload = {
            "kind": kind,
            "request": request,
            "receipt": receipt.model_dump(mode="json"),
            "collection_fingerprint": evidence.fingerprint,
            "lineage_verified": True,
            "model_calls": 0,
        }
        payload["digest"] = fingerprint(payload)
        serialized = json.dumps(payload, sort_keys=True)
        if len(serialized.encode()) > 250_000:
            raise LibraryError(413, "evidence_result_budget")
        existing = db.execute(
            "SELECT id FROM evidence_set_results WHERE set_id=? AND owner=? AND payload=?",
            (set_id, owner, serialized),
        ).fetchone()
        if existing:
            return {"id": existing["id"]}
        if (
            db.execute("SELECT COUNT(*) FROM evidence_set_results WHERE set_id=?", (set_id,)).fetchone()[0]
            >= 20
        ):
            raise LibraryError(429, "evidence_result_limit")
        result_id = uuid4().hex
        db.execute(
            "INSERT INTO evidence_set_results VALUES (?,?,?,?,?)",
            (result_id, set_id, owner, time.time(), serialized),
        )
        return {"id": result_id}

    def lookup(self, owner, set_id, request):
        with closing(self.library.connect()) as db, db:
            db.execute("BEGIN IMMEDIATE")
            _, _, evidence, scope, _ = self._load(db, owner, set_id)
            if request["collection_fingerprint"] != evidence.fingerprint:
                raise LibraryError(409, "evidence_set_context_changed")
            query = FactQuery(
                row_label=request["row_label"], column_label=request["column_label"], scope=scope
            )
            receipt = evidence.lookup(query)
            if not evidence.verify_fact(receipt):
                raise LibraryError(409, "evidence_receipt_unverifiable")
            return self._save(db, owner, set_id, "fact", request, receipt, evidence)

    def calculate(self, owner, set_id, request):
        with closing(self.library.connect()) as db, db:
            db.execute("BEGIN IMMEDIATE")
            _, _, evidence, _, _ = self._load(db, owner, set_id)
            if request["collection_fingerprint"] != evidence.fingerprint:
                raise LibraryError(409, "evidence_set_context_changed")
            facts = []
            for result_id in request["fact_ids"]:
                _, payload = self._result(db, owner, set_id, result_id)
                if payload["kind"] != "fact":
                    raise LibraryError(422, "select_saved_facts_only")
                facts.append(self._verify(db, owner, set_id, payload, evidence))
            receipt = evidence.calculate(request["operation"], facts)
            if not evidence.verify_calculation(receipt, facts=facts):
                raise LibraryError(409, "evidence_calculation_unverifiable")
            return self._save(db, owner, set_id, "calculation", request, receipt, evidence)

    def detail(self, owner, set_id):
        with closing(self.library.connect()) as db, db:
            row, binding, evidence, scope, sources = self._load(db, owner, set_id)
            catalog = evidence.query_catalog()
            results = []
            for saved in db.execute(
                "SELECT * FROM evidence_set_results WHERE set_id=? AND owner=? ORDER BY created DESC LIMIT 20",
                (set_id, owner),
            ):
                _, payload = self._result(db, owner, set_id, saved["id"])
                receipt = self._verify(db, owner, set_id, payload, evidence)
                review_row = db.execute(
                    "SELECT owner,created,digest,decision,note FROM evidence_set_reviews WHERE result_id=? AND owner=?",
                    (saved["id"], owner),
                ).fetchone()
                review = dict(review_row) if review_row else None
                if review and review["digest"] != payload["digest"]:
                    raise LibraryError(409, "evidence_review_binding_changed")
                reason = (
                    "unresolved_evidence"
                    if receipt.status not in {"agreed", "calculated"}
                    else "unreviewed"
                    if not review
                    else "owner_flagged"
                    if review["decision"] != "inspected"
                    else None
                )
                results.append(
                    {"id": saved["id"], "payload": payload, "review": review, "attention_reason": reason}
                )
            return {
                "id": set_id,
                "name": binding["name"],
                "created": row["created"],
                "scope": binding["scope"],
                "scope_label": scope,
                "collection_fingerprint": evidence.fingerprint,
                "sources": sources,
                "policy": self._config().model_dump(mode="json"),
                "catalog": [q.model_dump(mode="json") for q in catalog[:200]],
                "catalog_truncated": len(catalog) > 200,
                "results": results,
                "model_calls": 0,
            }

    def review(self, owner, set_id, result_id, request):
        with closing(self.library.connect()) as db, db:
            db.execute("BEGIN IMMEDIATE")
            _, _, evidence, _, _ = self._load(db, owner, set_id)
            _, payload = self._result(db, owner, set_id, result_id)
            self._verify(db, owner, set_id, payload, evidence)
            if request["digest"] != payload["digest"]:
                raise LibraryError(409, "evidence_result_changed")
            existing = db.execute(
                "SELECT * FROM evidence_set_reviews WHERE result_id=? AND owner=?", (result_id, owner)
            ).fetchone()
            if existing:
                if (existing["digest"], existing["decision"], existing["note"]) != (
                    request["digest"],
                    request["decision"],
                    request["note"],
                ):
                    raise LibraryError(409, "evidence_review_is_immutable")
                return {"recorded": True}
            db.execute(
                "INSERT INTO evidence_set_reviews VALUES (?,?,?,?,?,?)",
                (result_id, owner, time.time(), request["digest"], request["decision"], request["note"]),
            )
            return {"recorded": True}

    def delete(self, owner, set_id):
        with closing(self.library.connect()) as db, db:
            db.execute("BEGIN IMMEDIATE")
            self._load(db, owner, set_id)
            db.execute("DELETE FROM evidence_sets WHERE id=? AND owner=?", (set_id, owner))
