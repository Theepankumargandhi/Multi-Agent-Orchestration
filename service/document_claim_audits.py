"""Bounded, resumable numeric claim audits over owner-selected frozen evidence."""

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
from agentforge_sdk.claim_bundle import (
    ClaimBundleError,
    ClaimReplayEntry,
    ClaimReplayRecipe,
    build_claim_replay_bundle,
)
from agentforge_sdk.models import fingerprint
from agentforge_sdk.reconciliation import FactReconciliation
from service.document_reconciliation import DocumentReconciliation
from service.document_store import LibraryError


class DocumentClaimAudits:
    step_size = 2
    max_report_bytes = 3_000_000
    max_checkpoint_bytes = 100_000

    def __init__(self, library):
        self.library = library
        self.sets = DocumentReconciliation(library)

    def _input(self, db, owner, set_id, result_id, evidence):
        _, payload = self.sets._result(db, owner, set_id, result_id)
        receipt = self.sets._verify(db, owner, set_id, payload, evidence)
        parents = []
        if payload["kind"] == "calculation":
            for parent_id in payload["request"]["fact_ids"]:
                _, parent = self.sets._result(db, owner, set_id, parent_id)
                parents.append(self.sets._verify(db, owner, set_id, parent, evidence))
        return payload, receipt, parents

    @staticmethod
    def _checkpoint(digest, checks):
        return fingerprint({"binding_digest": digest, "checks": checks})

    def create(self, owner, set_id, request):
        with closing(self.library.connect()) as db, db:
            db.execute("BEGIN IMMEDIATE")
            _, _, evidence, _, _ = self.sets._load(db, owner, set_id)
            if request["collection_fingerprint"] != evidence.fingerprint:
                raise LibraryError(409, "evidence_set_context_changed")
            inputs = {}
            for claim in request["claims"]:
                result_id = claim["result_id"]
                if result_id not in inputs:
                    inputs[result_id] = self._input(db, owner, set_id, result_id, evidence)[0]["digest"]
            binding = {**request, "inputs": inputs}
            digest = fingerprint(binding)
            existing = db.execute(
                "SELECT id FROM evidence_claim_audits WHERE owner=? AND set_id=? AND digest=?",
                (owner, set_id, digest),
            ).fetchone()
            if existing:
                return {"id": existing["id"]}
            count = db.execute("SELECT COUNT(*),SUM(owner=?) FROM evidence_claim_audits", (owner,)).fetchone()
            if count[0] >= 100 or (count[1] or 0) >= 20:
                raise LibraryError(429, "claim_audit_limit")
            audit_id = uuid4().hex
            db.execute(
                "INSERT INTO evidence_claim_audits VALUES (?,?,?,?,?,?,?,?)",
                (
                    audit_id,
                    set_id,
                    owner,
                    time.time(),
                    json.dumps(binding, sort_keys=True),
                    digest,
                    "[]",
                    self._checkpoint(digest, []),
                ),
            )
            return {"id": audit_id}

    def list(self, owner, set_id):
        with closing(self.library.connect()) as db, db:
            self.sets._load(db, owner, set_id)
            return [
                {"id": row["id"], "name": json.loads(row["binding"])["name"], "created": row["created"]}
                for row in db.execute(
                    "SELECT id,created,binding FROM evidence_claim_audits WHERE owner=? AND set_id=? ORDER BY created DESC LIMIT 20",
                    (owner, set_id),
                )
            ]

    def _load(self, db, owner, set_id, audit_id):
        row = db.execute(
            "SELECT * FROM evidence_claim_audits WHERE id=? AND set_id=? AND owner=?",
            (audit_id, set_id, owner),
        ).fetchone()
        if row is None:
            raise LibraryError(404, "claim_audit_unavailable")
        binding, checks = json.loads(row["binding"]), json.loads(row["checks"])
        if (
            fingerprint(binding) != row["digest"]
            or len(checks) > len(binding["claims"])
            or self._checkpoint(row["digest"], checks) != row["checkpoint_fingerprint"]
        ):
            raise LibraryError(409, "claim_audit_checkpoint_changed")
        _, _, evidence, scope, sources = self.sets._load(db, owner, set_id)
        if binding["collection_fingerprint"] != evidence.fingerprint:
            raise LibraryError(409, "evidence_set_context_changed")
        inputs = {}
        for result_id, expected_digest in binding["inputs"].items():
            item = self._input(db, owner, set_id, result_id, evidence)
            if item[0]["digest"] != expected_digest:
                raise LibraryError(409, "claim_audit_input_changed")
            inputs[result_id] = item
        for index, serialized in enumerate(checks):
            check = NumericClaimCheck.model_validate_json(json.dumps(serialized))
            item = binding["claims"][index]
            _, receipt, parents = inputs[item["result_id"]]
            if check.claim != NumericClaim.model_validate(item["claim"]) or not verify_numeric_claim(
                evidence, check, receipt, facts=parents
            ):
                raise LibraryError(409, "claim_audit_check_unverifiable")
        return row, binding, checks, inputs, evidence, scope, sources

    def advance(self, owner, set_id, audit_id, expected):
        with closing(self.library.connect()) as db, db:
            db.execute("BEGIN IMMEDIATE")
            row, binding, checks, inputs, evidence, _, _ = self._load(db, owner, set_id, audit_id)
            if row["checkpoint_fingerprint"] != expected:
                raise LibraryError(409, "claim_audit_step_is_stale")
            if len(checks) == len(binding["claims"]):
                raise LibraryError(409, "claim_audit_already_complete")
            start = len(checks)
            for item in binding["claims"][start : start + self.step_size]:
                _, receipt, parents = inputs[item["result_id"]]
                check = check_numeric_claim(
                    evidence, NumericClaim.model_validate(item["claim"]), receipt, facts=parents
                )
                if not verify_numeric_claim(evidence, check, receipt, facts=parents):
                    raise LibraryError(409, "claim_audit_check_unverifiable")
                checks.append(check.model_dump(mode="json"))
            serialized = json.dumps(checks, sort_keys=True)
            if len(serialized.encode()) > self.max_checkpoint_bytes:
                raise LibraryError(413, "claim_audit_checkpoint_budget")
            db.execute(
                "UPDATE evidence_claim_audits SET checks=?,checkpoint_fingerprint=? WHERE id=? AND owner=?",
                (serialized, self._checkpoint(row["digest"], checks), audit_id, owner),
            )
            return {"id": audit_id}

    def detail(self, owner, set_id, audit_id, *, export=False):
        with closing(self.library.connect()) as db, db:
            row, binding, checks, inputs, evidence, scope, sources = self._load(db, owner, set_id, audit_id)
            counts = {
                status: sum(c["status"] == status for c in checks)
                for status in ("supported", "refuted", "unresolved")
            }
            complete = len(checks) == len(binding["claims"])
            items = []
            for index, item in enumerate(binding["claims"]):
                _, receipt, parents = inputs[item["result_id"]]
                facts = [receipt] if isinstance(receipt, FactReconciliation) else parents
                items.append(
                    {
                        **item,
                        "check": checks[index] if index < len(checks) else None,
                        "evidence_status": receipt.status,
                        "observations": [
                            o.model_dump(mode="json") for fact in facts for o in fact.observations
                        ],
                    }
                )
            report = {
                "protocol": "agentforge-workbench-claim-audit-v1",
                "id": audit_id,
                "name": binding["name"],
                "created": row["created"],
                "set_id": set_id,
                "scope_label": scope,
                "sources": sources,
                "collection_fingerprint": evidence.fingerprint,
                "binding_digest": row["digest"],
                "checkpoint_fingerprint": row["checkpoint_fingerprint"],
                "items": items,
                "processed": len(checks),
                "total": len(items),
                "complete": complete,
                "counts": counts,
                "all_claims_supported": complete and counts["supported"] == len(items),
                "model_calls": 0,
                "policy": "exact_numeric_only; no semantic entailment, rounding tolerance or approval",
            }
            if export:
                # Export only explicitly requested evidence, never originals, credentials or other sets.
                report["evidence"] = {key: value[0] for key, value in inputs.items()}
                report["integrity_notice"] = (
                    "Unkeyed digests detect changes; not a signature, standalone source proof or release authorization."
                )
                report["report_fingerprint"] = fingerprint(report)
            if len(json.dumps(report).encode()) > self.max_report_bytes:
                raise LibraryError(413, "claim_audit_report_budget")
            return report

    def delete(self, owner, set_id, audit_id):
        with closing(self.library.connect()) as db, db:
            # Deletion needs ownership, not a valid checkpoint; a damaged report must be removable.
            row = db.execute(
                "SELECT id FROM evidence_claim_audits WHERE id=? AND set_id=? AND owner=?",
                (audit_id, set_id, owner),
            ).fetchone()
            if row is None:
                raise LibraryError(404, "claim_audit_unavailable")
            db.execute("DELETE FROM evidence_claim_audits WHERE id=? AND owner=?", (audit_id, owner))

    def replay_bundle(self, owner, set_id, audit_id, expected):
        """Explicit private export, not another saved artifact or source authenticity proof."""
        with closing(self.library.connect()) as db, db:
            # One read snapshot: neither source results nor the audit may change mid-export.
            db.execute("BEGIN")
            row, binding, checks, inputs, evidence, _, _ = self._load(db, owner, set_id, audit_id)
            if expected != row["checkpoint_fingerprint"]:
                raise LibraryError(409, "claim_bundle_checkpoint_changed")
            if len(checks) != len(binding["claims"]):
                raise LibraryError(409, "complete_source_audit_first")
            try:
                return self._bundle(binding, checks, inputs, evidence).model_dump(mode="json")
            except ClaimBundleError:
                raise LibraryError(409, "claim_bundle_not_replayable_within_budget") from None

    @staticmethod
    def _bundle(binding, checks, inputs, evidence):
        entries = []
        for item, check in zip(binding["claims"], checks, strict=True):
            _, receipt, parents = inputs[item["result_id"]]
            fact = isinstance(receipt, FactReconciliation)
            entries.append(
                ClaimReplayEntry(
                    recipe=ClaimReplayRecipe(
                        claim=NumericClaim.model_validate(item["claim"]),
                        queries=tuple(f.query for f in ([receipt] if fact else parents)),
                        operation=None if fact else receipt.operation,
                    ),
                    check=NumericClaimCheck.model_validate(check),
                )
            )
        return build_claim_replay_bundle(evidence, entries)
