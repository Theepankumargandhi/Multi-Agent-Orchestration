"""Server-owned, resumable local revalidation. Never import approvals into a new version."""

from __future__ import annotations

import json
import time
from contextlib import closing
from uuid import uuid4

from agentforge_sdk.document_changes import ChangeConfig, compare_documents
from agentforge_sdk.models import fingerprint
from agentforge_sdk.multimodal import MultimodalRAG
from agentforge_sdk.revalidation import RevalidationBatch, RevalidationConfig, SavedAnswer, SavedCalculation
from agentforge_sdk.spatial import Calculation, MultimodalDocument, SpatialAnswer
from service.document_store import DocumentLibrary, LibraryError


class DocumentRevalidation:
    def __init__(self, library: DocumentLibrary):
        self.library = library

    @staticmethod
    def _manifest(row):
        if not row["manifest"]:
            raise LibraryError(409, "version_not_extracted")
        manifest = MultimodalDocument.model_validate_json(row["manifest"])
        if manifest.document_id != row["family_id"]:
            raise LibraryError(409, "document_lineage_invalid")
        return manifest

    def _pair(self, db, owner, before_id, after_id):
        before = self.library.row(db, owner, before_id)
        after = self.library.row(db, owner, after_id)
        if before_id == after_id or before["family_id"] != after["family_id"]:
            raise LibraryError(409, "choose_versions_from_one_lineage")
        ancestor = after["parent_id"]
        for _ in range(10):
            if ancestor is None or ancestor == before_id:
                break
            ancestor = self.library.row(db, owner, ancestor)["parent_id"]
        if ancestor != before_id:
            raise LibraryError(409, "source_must_be_an_ancestor_version")
        return before, after

    @staticmethod
    def _source(db, owner, document_id, result_id, digest=None):
        row = db.execute(
            "SELECT payload FROM document_results WHERE id=? AND document_id=? AND owner=?",
            (result_id, document_id, owner),
        ).fetchone()
        if row is None:
            raise LibraryError(404, "source_result_unavailable")
        payload = json.loads(row["payload"])
        actual = fingerprint({k: v for k, v in payload.items() if k != "digest"})
        if payload["digest"] != actual or (digest is not None and actual != digest):
            raise LibraryError(409, "source_result_changed")
        receipt_json = json.dumps(payload["receipt"])
        if payload["request"]["kind"] == "question":
            item = SavedAnswer(
                item_id=result_id,
                question=payload["request"]["question"],
                receipt=SpatialAnswer.model_validate_json(receipt_json),
            )
        else:
            item = SavedCalculation(item_id=result_id, receipt=Calculation.model_validate_json(receipt_json))
        return payload, item

    def _build(self, db, owner, before_id, after_id, binding):
        before, after = self._pair(db, owner, before_id, after_id)
        left, right = self._manifest(before), self._manifest(after)
        if (
            left.fingerprint != binding["before_fingerprint"]
            or right.fingerprint != binding["after_fingerprint"]
        ):
            raise LibraryError(409, "comparison_version_changed")
        sources = [self._source(db, owner, before_id, r["id"], r["digest"]) for r in binding["results"]]
        if sum(len(item.model_dump_json().encode()) for _, item in sources) > 2_000_000:
            raise LibraryError(413, "comparison_saved_input_budget")
        config = RevalidationConfig.model_validate_json(json.dumps(binding["config"]))
        change_config = ChangeConfig(max_spatial_comparisons=20_000)
        batch = RevalidationBatch(
            MultimodalRAG(left),
            MultimodalRAG(right),
            [item for _, item in sources],
            config=config,
            change_config=change_config,
        )
        if binding.get("batch_fingerprint", batch.fingerprint) != batch.fingerprint:
            raise LibraryError(409, "comparison_binding_changed")
        return batch, sources, left, right, change_config

    @staticmethod
    def _record(db, owner, after_id, comparison_id):
        row = db.execute(
            "SELECT * FROM document_comparisons WHERE id=? AND owner=? AND after_id=?",
            (comparison_id, owner, after_id),
        ).fetchone()
        if row is None:
            raise LibraryError(404, "comparison_unavailable")
        return row

    def create(self, owner, after_id, request):
        with closing(self.library.connect()) as db, db:
            db.execute("BEGIN IMMEDIATE")
            before, after = self._pair(db, owner, request["before_id"], after_id)
            left, right = self._manifest(before), self._manifest(after)
            if (
                left.fingerprint != request["before_fingerprint"]
                or right.fingerprint != request["after_fingerprint"]
            ):
                raise LibraryError(409, "comparison_version_changed")
            results = []
            for result_id in request["result_ids"]:
                payload, _ = self._source(db, owner, before["id"], result_id)
                results.append({"id": result_id, "digest": payload["digest"]})
            config = RevalidationConfig(
                refresh_answers=request["refresh_local"],
                replay_calculations=request["refresh_local"],
                max_items_per_step=2,
                max_answer_attempts=request["max_answer_attempts"],
                answer_top_k=3,
            )
            binding = {
                "before_fingerprint": left.fingerprint,
                "after_fingerprint": right.fingerprint,
                "results": results,
                "config": config.model_dump(mode="json"),
            }
            batch, _, _, _, _ = self._build(db, owner, before["id"], after_id, binding)
            binding["batch_fingerprint"] = batch.fingerprint
            serialized = json.dumps(binding, sort_keys=True)
            existing = db.execute(
                "SELECT id FROM document_comparisons WHERE owner=? AND before_id=? AND after_id=? AND binding=?",
                (owner, before["id"], after_id, serialized),
            ).fetchone()
            if existing:
                return {"id": existing["id"]}
            counts = db.execute("SELECT COUNT(*),SUM(owner=?) FROM document_comparisons", (owner,)).fetchone()
            if counts[0] >= 100 or (counts[1] or 0) >= 20:
                raise LibraryError(429, "comparison_library_full")
            comparison_id = uuid4().hex
            db.execute(
                "INSERT INTO document_comparisons VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    comparison_id,
                    owner,
                    before["id"],
                    after_id,
                    time.time(),
                    serialized,
                    None,
                    "{}",
                    "initial",
                    0,
                    len(results),
                    0,
                ),
            )
            return {"id": comparison_id}

    def list(self, owner, after_id):
        with closing(self.library.connect()) as db, db:
            self.library.row(db, owner, after_id)
            return [
                dict(r)
                for r in db.execute(
                    "SELECT id,before_id,after_id,created,processed,total,complete FROM document_comparisons WHERE owner=? AND after_id=? ORDER BY created DESC LIMIT 20",
                    (owner, after_id),
                )
            ]

    def advance(self, owner, after_id, comparison_id, expected):
        with closing(self.library.connect()) as db, db:
            # Serialize advancement and fresh-result insertion in one local transaction.
            db.execute("BEGIN IMMEDIATE")
            row = self._record(db, owner, after_id, comparison_id)
            if expected != row["checkpoint_fingerprint"]:
                raise LibraryError(409, "comparison_step_changed_refresh_first")
            binding = json.loads(row["binding"])
            batch, sources, _, target, _ = self._build(db, owner, row["before_id"], after_id, binding)
            old = batch.restore(row["checkpoint"]) if row["checkpoint"] else None
            if old is not None and old.complete:
                return {"id": comparison_id, "complete": True}
            checkpoint = batch.advance(old)
            serialized = checkpoint.model_dump_json()
            if len(serialized.encode()) > 4_000_000:
                raise LibraryError(413, "comparison_checkpoint_budget")
            links = json.loads(row["links"])
            for entry in checkpoint.entries[len(old.entries) if old else 0 :]:
                receipt = entry.answer or (entry.replay.result if entry.replay else None)
                if receipt is None:
                    continue
                request = next(
                    payload["request"] for payload, item in sources if item.item_id == entry.item_id
                ).copy()
                request["document_fingerprint"] = target.fingerprint
                reader = MultimodalRAG(target)
                if entry.answer is not None:
                    verified = reader.verify_answer(receipt, question=request["question"])
                else:
                    request["region_ids"] = list(receipt.region_ids)
                    verified = reader.verify_calculation(receipt)
                if not verified:
                    raise LibraryError(409, "fresh_receipt_unverifiable")
                links[entry.item_id] = self.library.save_result(db, owner, after_id, request, receipt)["id"]
            db.execute(
                "UPDATE document_comparisons SET checkpoint=?,links=?,checkpoint_fingerprint=?,processed=?,complete=? WHERE id=? AND owner=?",
                (
                    serialized,
                    json.dumps(links),
                    checkpoint.fingerprint,
                    len(checkpoint.entries),
                    int(checkpoint.complete),
                    comparison_id,
                    owner,
                ),
            )
            return {"id": comparison_id, "complete": checkpoint.complete}

    def detail(self, owner, after_id, comparison_id):
        with closing(self.library.connect()) as db, db:
            row = self._record(db, owner, after_id, comparison_id)
            binding = json.loads(row["binding"])
            batch, sources, before, after, change_config = self._build(
                db, owner, row["before_id"], after_id, binding
            )
            checkpoint = batch.restore(row["checkpoint"]) if row["checkpoint"] else None
            if (
                row["checkpoint_fingerprint"] != (checkpoint.fingerprint if checkpoint else "initial")
                or row["processed"] != (len(checkpoint.entries) if checkpoint else 0)
                or bool(row["complete"]) != bool(checkpoint and checkpoint.complete)
            ):
                raise LibraryError(409, "comparison_checkpoint_changed")
            queue = (
                {item.item_id: item.model_dump(mode="json") for item in batch.review_queue(checkpoint)}
                if checkpoint
                else {}
            )
            links = json.loads(row["links"])
            items = []
            for payload, source in sources:
                target_result_id = links.get(source.item_id)
                review = None
                if target_result_id:
                    result = db.execute(
                        "SELECT payload FROM document_results WHERE id=? AND document_id=? AND owner=?",
                        (target_result_id, after_id, owner),
                    ).fetchone()
                    entry = (
                        next((e for e in checkpoint.entries if e.item_id == source.item_id), None)
                        if checkpoint
                        else None
                    )
                    receipt = (
                        entry.answer or (entry.replay.result if entry.replay else None) if entry else None
                    )
                    result_payload = json.loads(result["payload"]) if result else None
                    if (
                        result is None
                        or receipt is None
                        or result_payload["receipt"] != receipt.model_dump(mode="json")
                        or result_payload["request"]["document_fingerprint"] != after.fingerprint
                        or result_payload["digest"]
                        != fingerprint({k: v for k, v in result_payload.items() if k != "digest"})
                    ):
                        raise LibraryError(409, "comparison_fresh_result_changed")
                    review_row = db.execute(
                        "SELECT owner,decision,digest FROM document_reviews WHERE result_id=? AND owner=?",
                        (target_result_id, owner),
                    ).fetchone()
                    review = dict(review_row) if review_row else None
                    if review and review.pop("digest") != result_payload["digest"]:
                        raise LibraryError(409, "comparison_review_binding_changed")
                    if review and review["decision"] == "supported":
                        queue.pop(source.item_id, None)
                    else:
                        queue.setdefault(
                            source.item_id,
                            {
                                "item_id": source.item_id,
                                "category": "new_version_review",
                                "reason": "fresh_version_requires_owner_review",
                            },
                        )
                title = payload["request"].get("question") or payload["request"]["operation"]
                old = payload["receipt"]
                items.append(
                    {
                        "source_result_id": source.item_id,
                        "target_result_id": target_result_id,
                        "title": title,
                        "old_status": old["status"],
                        "old_value": old.get("value"),
                        "old_unit": old.get("unit"),
                        "review": review,
                    }
                )
            return {
                "id": comparison_id,
                "before_id": row["before_id"],
                "after_id": after_id,
                "config": binding["config"],
                "checkpoint_fingerprint": row["checkpoint_fingerprint"],
                "processed": row["processed"],
                "total": row["total"],
                "complete": bool(row["complete"]),
                "changes": compare_documents(before, after, config=change_config).model_dump(mode="json"),
                "entries": [entry.model_dump(mode="json") for entry in checkpoint.entries]
                if checkpoint
                else [],
                "items": items,
                "review_queue": list(queue.values()),
                "model_calls": 0,
                "mode": "local_extractive",
            }
