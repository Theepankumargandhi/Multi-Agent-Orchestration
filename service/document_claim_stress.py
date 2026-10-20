"""Diagnostic source-omission stress tests; never rewrite evidence or resolve conflicts."""

import json
import sqlite3
import time
from contextlib import closing
from dataclasses import dataclass
from uuid import uuid4

from agentforge_sdk.claim_audit import NumericClaim, check_numeric_claim, verify_numeric_claim
from agentforge_sdk.models import fingerprint
from agentforge_sdk.reconciliation import EvidenceSet, FactReconciliation
from service.document_claim_audits import DocumentClaimAudits
from service.document_store import LibraryError


@dataclass(frozen=True)
class _AuditContext:
    row: sqlite3.Row
    binding: dict
    checks: list
    inputs: dict
    evidence: EvidenceSet
    sources: list


def omission_effect(baseline, variant):
    """Classify a controlled input change, not which source is right."""
    if baseline["status"] == "supported" and variant["status"] != "supported":
        return "support_lost"
    if baseline["status"] != "supported" and variant["status"] == "supported":
        return "support_appeared_after_omission"
    if (
        baseline["observed_value"] is not None
        and variant["observed_value"] is not None
        and (baseline["observed_value"], baseline["observed_unit"])
        != (variant["observed_value"], variant["observed_unit"])
    ):
        return "observed_value_changed"
    return "verdict_changed" if baseline["status"] != variant["status"] else "unchanged_verdict"


class DocumentClaimStress:
    max_checkpoint_bytes = 150_000
    max_report_bytes = 3_000_000

    def __init__(self, library):
        self.library = library
        self.audits = DocumentClaimAudits(library)

    def _source(self, db, owner, set_id, audit_id):
        row, binding, checks, inputs, evidence, _, sources = self.audits._load(db, owner, set_id, audit_id)
        if len(checks) != len(binding["claims"]):
            raise LibraryError(409, "complete_source_audit_first")
        if not 2 <= len(evidence.source_ids) <= 5:
            raise LibraryError(409, "stress_source_count_outside_bounds")
        return _AuditContext(row, binding, checks, inputs, evidence, sources)

    @staticmethod
    def _checkpoint(digest, scenarios):
        return fingerprint({"stress_binding": digest, "scenarios": scenarios})

    def create(self, owner, set_id, audit_id, request):
        with closing(self.library.connect()) as db, db:
            db.execute("BEGIN IMMEDIATE")
            source = self._source(db, owner, set_id, audit_id)
            if request["source_checkpoint_fingerprint"] != source.row["checkpoint_fingerprint"]:
                raise LibraryError(409, "stress_source_checkpoint_changed")
            binding = {
                **request,
                "set_id": set_id,
                "source_audit_id": audit_id,
                "source_binding_digest": source.row["digest"],
                "collection_fingerprint": source.evidence.fingerprint,
                "omission_order": list(source.evidence.source_ids),
                "mode": "leave_one_source_out",
            }
            digest = fingerprint(binding)
            existing = db.execute(
                "SELECT id FROM evidence_claim_stress WHERE owner=? AND digest=?", (owner, digest)
            ).fetchone()
            if existing:
                return {"id": existing["id"]}
            counts = db.execute(
                "SELECT COUNT(*),SUM(owner=?) FROM evidence_claim_stress", (owner,)
            ).fetchone()
            if counts[0] >= 100 or (counts[1] or 0) >= 20:
                raise LibraryError(429, "claim_stress_limit")
            run_id = uuid4().hex
            db.execute(
                "INSERT INTO evidence_claim_stress VALUES (?,?,?,?,?,?,?,?)",
                (
                    run_id,
                    audit_id,
                    owner,
                    time.time(),
                    json.dumps(binding, sort_keys=True),
                    digest,
                    "[]",
                    self._checkpoint(digest, []),
                ),
            )
            return {"id": run_id}

    def list(self, owner, set_id, audit_id):
        with closing(self.library.connect()) as db, db:
            self._source(db, owner, set_id, audit_id)
            return [
                {"id": row["id"], "name": json.loads(row["binding"])["name"]}
                for row in db.execute(
                    "SELECT id,binding FROM evidence_claim_stress WHERE source_audit_id=? AND owner=? ORDER BY created DESC LIMIT 20",
                    (audit_id, owner),
                )
            ]

    @staticmethod
    def _evaluate(source, omitted):
        view = source.evidence.without_sources([omitted])
        cache, checks, observations = {}, [], []

        def lookup(original):
            key = fingerprint(original.query.model_dump(mode="json"))
            if key not in cache:
                fresh = view.lookup(original.query)
                if not view.verify_fact(fresh):
                    raise LibraryError(409, "stress_fact_unverifiable")
                cache[key] = fresh
            return cache[key]

        for item in source.binding["claims"]:
            _, original, parents = source.inputs[item["result_id"]]
            facts = (
                [lookup(original)]
                if isinstance(original, FactReconciliation)
                else [lookup(f) for f in parents]
            )
            if isinstance(original, FactReconciliation):
                receipt = facts[0]
            else:
                receipt = view.calculate(original.operation, facts)
                if not view.verify_calculation(receipt, facts=facts):
                    raise LibraryError(409, "stress_calculation_unverifiable")
            check = check_numeric_claim(
                view, NumericClaim.model_validate(item["claim"]), receipt, facts=facts
            )
            if not verify_numeric_claim(view, check, receipt, facts=facts):
                raise LibraryError(409, "stress_check_unverifiable")
            checks.append(check.model_dump(mode="json"))
            observations.append([o.model_dump(mode="json") for fact in facts for o in fact.observations])
        return {
            "omitted_source_id": omitted,
            "collection_fingerprint": view.fingerprint,
            "checks": checks,
        }, observations

    def _load(self, db, owner, set_id, audit_id, run_id):
        row = db.execute(
            "SELECT * FROM evidence_claim_stress WHERE id=? AND owner=? AND source_audit_id=?",
            (run_id, owner, audit_id),
        ).fetchone()
        if row is None:
            raise LibraryError(404, "claim_stress_unavailable")
        binding, scenarios = json.loads(row["binding"]), json.loads(row["scenarios"])
        if binding["set_id"] != set_id:
            raise LibraryError(404, "claim_stress_unavailable")
        if (
            fingerprint(binding) != row["digest"]
            or self._checkpoint(row["digest"], scenarios) != row["checkpoint_fingerprint"]
        ):
            raise LibraryError(409, "claim_stress_checkpoint_changed")
        source = self._source(db, owner, set_id, audit_id)
        if (
            source.row["digest"] != binding["source_binding_digest"]
            or source.row["checkpoint_fingerprint"] != binding["source_checkpoint_fingerprint"]
            or source.evidence.fingerprint != binding["collection_fingerprint"]
            or list(source.evidence.source_ids) != binding["omission_order"]
            or len(scenarios) > len(binding["omission_order"])
        ):
            raise LibraryError(409, "claim_stress_source_changed")
        evaluated = []
        for index, scenario in enumerate(scenarios):
            fresh, observations = self._evaluate(source, binding["omission_order"][index])
            if fresh != scenario:
                raise LibraryError(409, "claim_stress_scenario_unverifiable")
            evaluated.append((fresh, observations))
        return row, binding, source, scenarios, evaluated

    def advance(self, owner, set_id, audit_id, run_id, expected):
        with closing(self.library.connect()) as db, db:
            db.execute("BEGIN IMMEDIATE")
            row, binding, source, scenarios, _ = self._load(db, owner, set_id, audit_id, run_id)
            if expected != row["checkpoint_fingerprint"]:
                raise LibraryError(409, "claim_stress_step_is_stale")
            if len(scenarios) == len(binding["omission_order"]):
                raise LibraryError(409, "claim_stress_already_complete")
            scenario, _ = self._evaluate(source, binding["omission_order"][len(scenarios)])
            scenarios.append(scenario)
            serialized = json.dumps(scenarios, sort_keys=True)
            if len(serialized.encode()) > self.max_checkpoint_bytes:
                raise LibraryError(413, "claim_stress_checkpoint_budget")
            db.execute(
                "UPDATE evidence_claim_stress SET scenarios=?,checkpoint_fingerprint=? WHERE id=? AND owner=?",
                (serialized, self._checkpoint(row["digest"], scenarios), run_id, owner),
            )
            return {"id": run_id}

    def detail(self, owner, set_id, audit_id, run_id):
        with closing(self.library.connect()) as db, db:
            row, binding, source, scenarios, evaluated = self._load(db, owner, set_id, audit_id, run_id)
            complete = len(scenarios) == len(binding["omission_order"])
            claims = []
            for index, item in enumerate(source.binding["claims"]):
                baseline = source.checks[index]
                variants = []
                for position, omitted in enumerate(binding["omission_order"]):
                    check = scenarios[position]["checks"][index] if position < len(scenarios) else None
                    variants.append(
                        {
                            "omitted_source_id": omitted,
                            "check": check,
                            "effect": omission_effect(baseline, check) if check else "pending",
                            "observations": evaluated[position][1][index] if check else [],
                        }
                    )
                claims.append(
                    {
                        "claim": item["claim"],
                        "baseline": baseline,
                        "variants": variants,
                        "stable_numeric_support": all(v["check"]["status"] == "supported" for v in variants)
                        if complete and baseline["status"] == "supported"
                        else None,
                    }
                )
            report = {
                "id": run_id,
                "name": binding["name"],
                "mode": binding["mode"],
                "diagnostic_only": True,
                "checkpoint_fingerprint": row["checkpoint_fingerprint"],
                "source_checkpoint_fingerprint": binding["source_checkpoint_fingerprint"],
                "sources": source.sources,
                "processed": len(scenarios),
                "total": len(binding["omission_order"]),
                "complete": complete,
                "claims": claims,
                "evaluated_checks": len(scenarios) * len(claims),
                "support_lost_cases": sum(
                    v["effect"] == "support_lost" for c in claims for v in c["variants"]
                ),
                "support_appeared_cases": sum(
                    v["effect"] == "support_appeared_after_omission" for c in claims for v in c["variants"]
                ),
                "model_calls": 0,
                "policy": self.audits.sets._config().model_dump(mode="json"),
            }
            if len(json.dumps(report).encode()) > self.max_report_bytes:
                raise LibraryError(413, "claim_stress_report_budget")
            return report

    def delete(self, owner, set_id, audit_id, run_id):
        with closing(self.library.connect()) as db, db:
            row = db.execute(
                "SELECT s.id FROM evidence_claim_stress s JOIN evidence_claim_audits a ON a.id=s.source_audit_id WHERE s.id=? AND s.owner=? AND a.owner=? AND a.id=? AND a.set_id=?",
                (run_id, owner, owner, audit_id, set_id),
            ).fetchone()
            if row is None:
                raise LibraryError(404, "claim_stress_unavailable")
            db.execute("DELETE FROM evidence_claim_stress WHERE id=? AND owner=?", (run_id, owner))
