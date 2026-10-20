"""Replay fixed numeric assertions against an explicitly selected comparable evidence set."""

import json
import time
from contextlib import closing
from uuid import uuid4

from agentforge_sdk.claim_audit import (
    NumericClaim,
    NumericClaimCheck,
    check_numeric_claim,
    verify_numeric_claim,
)
from agentforge_sdk.models import fingerprint
from agentforge_sdk.reconciliation import FactQuery, FactReconciliation
from service.document_claim_audits import DocumentClaimAudits
from service.document_store import LibraryError


def compare_checks(before, after):
    """Internal: callers first reverify both checks and their identical assertions."""
    if before.claim != after.claim:
        raise LibraryError(409, "replay_claim_changed")
    if before.status == "supported" and after.status != "supported":
        transition = "support_lost"
    elif before.status != "supported" and after.status == "supported":
        transition = "new_support"
    elif before.status != after.status:
        transition = "became_unresolved" if after.status == "unresolved" else "new_refutation"
    else:
        transition = "unchanged_verdict"
    return {
        "transition": transition,
        "observed_value_changed": None
        if before.observed_value is None or after.observed_value is None
        else (before.observed_value, before.observed_unit) != (after.observed_value, after.observed_unit),
    }


class DocumentClaimReplay:
    step_size = 2
    max_checkpoint_bytes = 150_000
    max_report_bytes = 3_000_000

    def __init__(self, library):
        self.library = library
        self.audits = DocumentClaimAudits(library)
        self.sets = self.audits.sets

    @staticmethod
    def _checkpoint(digest, entries):
        return fingerprint({"replay_binding": digest, "entries": entries})

    @staticmethod
    def _signature(receipt, parents):
        facts = [receipt] if isinstance(receipt, FactReconciliation) else parents
        return {
            "kind": "fact" if isinstance(receipt, FactReconciliation) else "calculation",
            "operation": getattr(receipt, "operation", None),
            "axes": [(f.query.row_label, f.query.column_label) for f in facts],
        }

    def _source(self, db, owner, set_id, audit_id):
        state = self.audits._load(db, owner, set_id, audit_id)
        if len(state[2]) != len(state[1]["claims"]):
            raise LibraryError(409, "complete_source_audit_first")
        return state

    @staticmethod
    def _plan(inputs):
        # Unique literal queries plus calculations bound worst-case new saved results.
        queries, calculations = set(), set()
        for payload, receipt, parents in inputs.values():
            facts = [receipt] if isinstance(receipt, FactReconciliation) else parents
            keys = [
                fingerprint({"row_label": f.query.row_label, "column_label": f.query.column_label})
                for f in facts
            ]
            queries.update(keys)
            if payload["kind"] == "calculation":
                calculations.add(fingerprint({"operation": receipt.operation, "facts": keys}))
        return len(queries) + len(calculations)

    def create(self, owner, source_set_id, audit_id, request):
        with closing(self.library.connect()) as db, db:
            db.execute("BEGIN IMMEDIATE")
            source, _, _, inputs, _, _, _ = self._source(db, owner, source_set_id, audit_id)
            _, old_binding, _, _, _ = self.sets._load(db, owner, source_set_id)
            _, new_binding, target, _, _ = self.sets._load(db, owner, request["target_set_id"])
            if (
                source["checkpoint_fingerprint"] != request["source_checkpoint_fingerprint"]
                or target.fingerprint != request["target_collection_fingerprint"]
            ):
                raise LibraryError(409, "replay_context_changed")
            if request["target_set_id"] == source_set_id:
                raise LibraryError(422, "choose_a_different_evidence_set")
            if old_binding["scope"] != new_binding["scope"]:
                raise LibraryError(409, "replay_requires_identical_declared_scope")
            planned = self._plan(inputs)
            if planned > 20:
                raise LibraryError(422, "replay_exceeds_target_result_capacity")
            binding = {
                **request,
                "source_set_id": source_set_id,
                "source_audit_id": audit_id,
                "source_binding_digest": source["digest"],
                "planned_result_upper_bound": planned,
            }
            digest = fingerprint(binding)
            existing = db.execute(
                "SELECT id FROM evidence_claim_replays WHERE owner=? AND digest=?", (owner, digest)
            ).fetchone()
            if existing:
                return {"id": existing["id"]}
            counts = db.execute(
                "SELECT COUNT(*),SUM(owner=?) FROM evidence_claim_replays", (owner,)
            ).fetchone()
            if counts[0] >= 100 or (counts[1] or 0) >= 20:
                raise LibraryError(429, "claim_replay_limit")
            replay_id = uuid4().hex
            db.execute(
                "INSERT INTO evidence_claim_replays VALUES (?,?,?,?,?,?,?,?,?)",
                (
                    replay_id,
                    audit_id,
                    request["target_set_id"],
                    owner,
                    time.time(),
                    json.dumps(binding, sort_keys=True),
                    digest,
                    "[]",
                    self._checkpoint(digest, []),
                ),
            )
            return {"id": replay_id}

    def list(self, owner, source_set_id, audit_id):
        with closing(self.library.connect()) as db, db:
            self._source(db, owner, source_set_id, audit_id)
            return [
                {
                    "id": row["id"],
                    "name": json.loads(row["binding"])["name"],
                    "target_set_id": row["target_set_id"],
                }
                for row in db.execute(
                    "SELECT id,binding,target_set_id FROM evidence_claim_replays WHERE owner=? AND source_audit_id=? ORDER BY created DESC LIMIT 20",
                    (owner, audit_id),
                )
            ]

    def _load(self, db, owner, source_set_id, audit_id, replay_id):
        row = db.execute(
            "SELECT * FROM evidence_claim_replays WHERE id=? AND owner=? AND source_audit_id=?",
            (replay_id, owner, audit_id),
        ).fetchone()
        if row is None:
            raise LibraryError(404, "claim_replay_unavailable")
        binding, entries = json.loads(row["binding"]), json.loads(row["entries"])
        if binding["source_set_id"] != source_set_id or binding["source_audit_id"] != audit_id:
            raise LibraryError(404, "claim_replay_unavailable")
        if (
            fingerprint(binding) != row["digest"]
            or self._checkpoint(row["digest"], entries) != row["checkpoint_fingerprint"]
        ):
            raise LibraryError(409, "claim_replay_checkpoint_changed")
        source = self._source(db, owner, source_set_id, audit_id)
        if (
            source[0]["digest"] != binding["source_binding_digest"]
            or source[0]["checkpoint_fingerprint"] != binding["source_checkpoint_fingerprint"]
            or len(entries) > len(source[1]["claims"])
        ):
            raise LibraryError(409, "claim_replay_source_changed")
        _, _, target, scope, sources = self.sets._load(db, owner, row["target_set_id"])
        if target.fingerprint != binding["target_collection_fingerprint"]:
            raise LibraryError(409, "claim_replay_target_changed")
        current = {}
        for index, entry in enumerate(entries):
            result_id = entry["result_id"]
            if result_id not in current:
                current[result_id] = self.audits._input(db, owner, row["target_set_id"], result_id, target)
            payload, receipt, parents = current[result_id]
            try:
                check = NumericClaimCheck.model_validate_json(json.dumps(entry["check"]))
            except ValueError:
                raise LibraryError(409, "claim_replay_check_unverifiable") from None
            spec = NumericClaim.model_validate(source[1]["claims"][index]["claim"])
            _, old_receipt, old_parents = source[3][source[1]["claims"][index]["result_id"]]
            if (
                entry["digest"] != payload["digest"]
                or check.claim != spec
                or self._signature(old_receipt, old_parents) != self._signature(receipt, parents)
                or not verify_numeric_claim(target, check, receipt, facts=parents)
            ):
                raise LibraryError(409, "claim_replay_check_unverifiable")
        return row, binding, entries, source, target, scope, sources, current

    def _lookup(self, db, owner, set_id, evidence, scope, old_fact, cache):
        query = FactQuery(
            row_label=old_fact.query.row_label, column_label=old_fact.query.column_label, scope=scope
        )
        key = fingerprint(query.model_dump(mode="json"))
        if key not in cache:
            receipt = evidence.lookup(query)
            if not evidence.verify_fact(receipt):
                raise LibraryError(409, "replay_fact_unverifiable")
            request = {
                "collection_fingerprint": evidence.fingerprint,
                "row_label": query.row_label,
                "column_label": query.column_label,
            }
            saved = self.sets._save(db, owner, set_id, "fact", request, receipt, evidence)
            cache[key] = (saved["id"], receipt)
        return cache[key]

    def advance(self, owner, source_set_id, audit_id, replay_id, expected):
        with closing(self.library.connect()) as db, db:
            db.execute("BEGIN IMMEDIATE")
            row, _, entries, source, target, scope, _, _ = self._load(
                db, owner, source_set_id, audit_id, replay_id
            )
            if expected != row["checkpoint_fingerprint"]:
                raise LibraryError(409, "claim_replay_step_is_stale")
            if len(entries) == len(source[1]["claims"]):
                raise LibraryError(409, "claim_replay_already_complete")
            cache = {}
            for item in source[1]["claims"][len(entries) : len(entries) + self.step_size]:
                old_payload, old_receipt, old_parents = source[3][item["result_id"]]
                facts = [old_receipt] if isinstance(old_receipt, FactReconciliation) else old_parents
                fresh = [
                    self._lookup(db, owner, row["target_set_id"], target, scope, fact, cache)
                    for fact in facts
                ]
                if old_payload["kind"] == "fact":
                    result_id = fresh[0][0]
                else:
                    parents = [f[1] for f in fresh]
                    receipt = target.calculate(old_receipt.operation, parents)
                    if not target.verify_calculation(receipt, facts=parents):
                        raise LibraryError(409, "replay_calculation_unverifiable")
                    request = {
                        "collection_fingerprint": target.fingerprint,
                        "operation": old_receipt.operation,
                        "fact_ids": [f[0] for f in fresh],
                    }
                    result_id = self.sets._save(
                        db, owner, row["target_set_id"], "calculation", request, receipt, target
                    )["id"]
                payload, receipt, parents = self.audits._input(
                    db, owner, row["target_set_id"], result_id, target
                )
                check = check_numeric_claim(
                    target, NumericClaim.model_validate(item["claim"]), receipt, facts=parents
                )
                if not verify_numeric_claim(target, check, receipt, facts=parents):
                    raise LibraryError(409, "claim_replay_check_unverifiable")
                entries.append(
                    {
                        "result_id": result_id,
                        "digest": payload["digest"],
                        "check": check.model_dump(mode="json"),
                    }
                )
            serialized = json.dumps(entries, sort_keys=True)
            if len(serialized.encode()) > self.max_checkpoint_bytes:
                raise LibraryError(413, "claim_replay_checkpoint_budget")
            db.execute(
                "UPDATE evidence_claim_replays SET entries=?,checkpoint_fingerprint=? WHERE id=? AND owner=?",
                (serialized, self._checkpoint(row["digest"], entries), replay_id, owner),
            )
            return {"id": replay_id}

    def detail(self, owner, source_set_id, audit_id, replay_id):
        with closing(self.library.connect()) as db, db:
            row, binding, entries, source, _, _, sources, current = self._load(
                db, owner, source_set_id, audit_id, replay_id
            )
            items = []
            for index, spec in enumerate(source[1]["claims"]):
                before = NumericClaimCheck.model_validate_json(json.dumps(source[2][index]))
                after, observations, result_id = None, [], None
                if index < len(entries):
                    entry = entries[index]
                    after = NumericClaimCheck.model_validate_json(json.dumps(entry["check"]))
                    result_id = entry["result_id"]
                    _, receipt, parents = current[result_id]
                    facts = [receipt] if isinstance(receipt, FactReconciliation) else parents
                    observations = [o.model_dump(mode="json") for f in facts for o in f.observations]
                items.append(
                    {
                        "claim": spec["claim"],
                        "before": before.model_dump(mode="json"),
                        "after": after.model_dump(mode="json") if after else None,
                        "comparison": compare_checks(before, after) if after else None,
                        "target_result_id": result_id,
                        "observations": observations,
                    }
                )
            complete = len(entries) == len(items)
            report = {
                "id": replay_id,
                "name": binding["name"],
                "target_set_id": row["target_set_id"],
                "target_sources": sources,
                "checkpoint_fingerprint": row["checkpoint_fingerprint"],
                "source_checkpoint_fingerprint": binding["source_checkpoint_fingerprint"],
                "processed": len(entries),
                "total": len(items),
                "complete": complete,
                "items": items,
                "support_lost": sum(
                    i["comparison"] is not None and i["comparison"]["transition"] == "support_lost"
                    for i in items
                ),
                "all_claims_supported": complete
                and all(e["check"]["status"] == "supported" for e in entries),
                "planned_result_upper_bound": binding["planned_result_upper_bound"],
                "model_calls": 0,
            }
            if len(json.dumps(report).encode()) > self.max_report_bytes:
                raise LibraryError(413, "claim_replay_report_budget")
            return report

    def delete(self, owner, source_set_id, audit_id, replay_id):
        with closing(self.library.connect()) as db, db:
            row = db.execute(
                "SELECT r.id FROM evidence_claim_replays r JOIN evidence_claim_audits a ON a.id=r.source_audit_id WHERE r.id=? AND r.owner=? AND a.owner=? AND a.id=? AND a.set_id=?",
                (replay_id, owner, owner, audit_id, source_set_id),
            ).fetchone()
            if row is None:
                raise LibraryError(404, "claim_replay_unavailable")
            db.execute("DELETE FROM evidence_claim_replays WHERE id=? AND owner=?", (replay_id, owner))
