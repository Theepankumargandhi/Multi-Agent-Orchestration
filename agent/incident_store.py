"""Encrypted, tenant-scoped reproduction cases; immutable trials and independent reviews."""
from __future__ import annotations

import hashlib
import hmac
import json
import os
import secrets
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path
from uuid import uuid4

from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from agent.incident_investigation import analyze
from agent.incident_models import (
    IncidentBundle,
    InvestigationPlan,
    TrialReview,
    fingerprint,
    runtime_fingerprint,
)


class IncidentConflict(ValueError):
    pass


class IncidentStore:
    def __init__(self, path: str | Path, key: bytes, *, clock=None, max_records=200):
        if len(key) != 32:
            raise ValueError("incident encryption requires an independent 32-byte key")
        self.path, self.key, self.clock, self.max_records = Path(path), key, clock or time.time, max_records
        self.cipher = AESGCM(hmac.new(key, b"incident-encryption-v1", hashlib.sha256).digest())
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.db() as db:
            db.execute("CREATE TABLE IF NOT EXISTS incidents (id TEXT PRIMARY KEY, tenant TEXT NOT NULL, expires REAL NOT NULL, parent TEXT, family TEXT NOT NULL, body BLOB NOT NULL)")
            db.execute("CREATE UNIQUE INDEX IF NOT EXISTS incident_shadow_family ON incidents(tenant,parent,family) WHERE parent IS NOT NULL")
            db.execute("CREATE TABLE IF NOT EXISTS incident_exposures (tenant TEXT, parent TEXT, family TEXT, expires REAL, PRIMARY KEY(tenant,parent,family))")
            if "content" not in [row[1] for row in db.execute("PRAGMA table_info(incident_exposures)")]:
                db.execute("ALTER TABLE incident_exposures ADD COLUMN content TEXT NOT NULL DEFAULT ''")
            db.execute("CREATE UNIQUE INDEX IF NOT EXISTS incident_shadow_content ON incident_exposures(tenant,parent,content) WHERE content!=''")

    @contextmanager
    def db(self):
        db = sqlite3.connect(self.path, timeout=10)
        db.row_factory = sqlite3.Row
        try:
            db.execute("BEGIN IMMEDIATE")
            yield db
            db.commit()
        except BaseException:
            db.rollback()
            raise
        finally:
            db.close()

    def principal(self, name: str) -> str:
        if not name or len(name) > 200:
            raise ValueError("authenticated principal is required")
        return hmac.new(self.key, b"incident-principal-v1\0" + name.encode(), hashlib.sha256).hexdigest()

    def aad(self, row) -> bytes:
        return json.dumps(["incident-v1", row["id"], row["tenant"], float(row["expires"]), row["parent"], row["family"]], separators=(",", ":")).encode()

    def content_id(self, bundle: IncidentBundle) -> str:
        value = {"query": " ".join(bundle.query.casefold().split()),
                 "documents": sorted(doc.version_sha256 for doc in bundle.documents)}
        return hmac.new(self.key, b"incident-content-v1\0" + json.dumps(value, sort_keys=True).encode(), hashlib.sha256).hexdigest()

    def seal(self, row, data: dict) -> bytes:
        encoded = json.dumps(data, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        if len(encoded) > 1_000_000:
            raise IncidentConflict("encrypted case capacity reached")
        nonce = os.urandom(12)
        return nonce + self.cipher.encrypt(nonce, encoded, self.aad(row))

    def unseal(self, row) -> dict:
        try:
            blob = bytes(row["body"])
            return json.loads(self.cipher.decrypt(blob[:12], blob[12:], self.aad(row)))
        except Exception:
            raise IncidentConflict("private case integrity verification failed") from None

    def load(self, db, identifier: str, actor: str, *, reviewer=False):
        row = db.execute("SELECT * FROM incidents WHERE id=?", (identifier,)).fetchone()
        if row is None or row["expires"] <= self.clock():
            raise KeyError("case unavailable")
        principal = self.principal(actor)
        if not reviewer and row["tenant"] != principal:
            raise KeyError("case unavailable")
        data = self.unseal(row)
        if reviewer and principal != row["tenant"] and principal not in data["reviewers"]:
            raise KeyError("case unavailable")
        return row, data

    def save(self, db, row, data):
        db.execute("UPDATE incidents SET body=? WHERE id=?", (self.seal(row, data), row["id"]))

    def create(self, owner: str, bundle: IncidentBundle, plan: InvestigationPlan, *, consent: bool,
               ttl_seconds=86400, parent_id: str | None = None) -> str:
        if consent is not True or not 60 <= ttl_seconds <= 7 * 86400:
            raise ValueError("explicit content-capture consent and a 1-minute to 7-day TTL are required")
        bundle = IncidentBundle.model_validate(bundle.model_dump())
        identifier = uuid4().hex
        tenant = self.principal(owner)
        family = hmac.new(self.key, b"incident-family-v1\0" + bundle.family.encode(), hashlib.sha256).hexdigest()
        with self.db() as db:
            db.execute("DELETE FROM incidents WHERE expires<=?", (self.clock(),))
            db.execute("DELETE FROM incident_exposures WHERE expires<=?", (self.clock(),))
            if db.execute("SELECT count(*) FROM incidents").fetchone()[0] >= self.max_records:
                raise IncidentConflict("incident capacity reached")
            data = {"id": identifier, "bundle": bundle.model_dump(), "bundle_fingerprint": fingerprint(bundle.model_dump()),
                "plan": plan.model_dump(), "state": "captured", "reviewers": [], "trials": [], "reviews": {},
                "parent_id": parent_id, "execution_kind": None}
            data["runtime_fingerprint"] = runtime_fingerprint()
            data["content_fingerprint"] = self.content_id(bundle)
            if parent_id:
                parent_row, parent = self.load(db, parent_id, owner)
                proposal = analyze(parent)
                if proposal["decision"] != "supported_under_replay":
                    raise IncidentConflict("a reviewed live replay proposal is required")
                if (family == parent_row["family"] or data["content_fingerprint"] == parent.get("content_fingerprint")
                        or bundle.model != parent["bundle"]["model"] or bundle.pipeline.model_dump() != parent["bundle"]["pipeline"]
                        or bundle.prompt != parent["bundle"]["prompt"] or bundle.application_revision != parent["bundle"]["application_revision"]
                        or plan.model_dump() != parent["plan"] or data["runtime_fingerprint"] != parent["runtime_fingerprint"]):
                    raise IncidentConflict("shadow requires a fresh family, the same model and the frozen base policy")
                data["candidate_config"] = proposal["candidate_config"]
                data["candidate_fingerprint"] = proposal["candidate_fingerprint"]
                try:
                    db.execute("INSERT INTO incident_exposures(tenant,parent,family,expires,content) VALUES (?,?,?,?,?)",
                        (tenant, parent_id, family, parent_row["expires"], data["content_fingerprint"]))
                except sqlite3.IntegrityError:
                    raise IncidentConflict("this shadow family has already been exposed") from None
            row = {"id": identifier, "tenant": tenant, "expires": self.clock() + ttl_seconds,
                   "parent": parent_id, "family": family}
            try:
                db.execute("INSERT INTO incidents VALUES (?,?,?,?,?,?)", (identifier, tenant, row["expires"], parent_id, family, self.seal(row, data)))
            except sqlite3.IntegrityError:
                raise IncidentConflict("this shadow family has already been exposed") from None
        return identifier

    def get(self, identifier: str, owner: str) -> dict:
        with self.db() as db:
            return self.load(db, identifier, owner)[1]

    def enqueue(self, identifier: str, owner: str, kind: str, *, authorized_live=False):
        if kind not in {"live_provider", "extractive_control"} or (kind == "live_provider" and authorized_live is not True):
            raise ValueError("explicit execution authorization required")
        with self.db() as db:
            row, data = self.load(db, identifier, owner)
            if data["state"] != "captured" or data.get("job"):
                raise IncidentConflict("investigation was already submitted")
            data["job"] = {"state": "queued", "owner": owner, "kind": kind, "authorized_live": authorized_live,
                           "queued_at": self.clock(), "cancel_requested": False, "lease_token": None, "lease_until": None}
            self.save(db, row, data)
        return self.report(identifier, owner)

    def _check_lease(self, data, token):
        job = data.get("job")
        if job and (job["state"] != "leased" or not token or not secrets.compare_digest(job["lease_token"] or "", token)
                    or job["lease_until"] <= self.clock() or job["cancel_requested"]):
            raise IncidentConflict("worker lease is unavailable, cancelled or expired")

    def claim_next(self, *, lease_seconds=60):
        if not 5 <= lease_seconds <= 300:
            raise ValueError("bounded worker lease required")
        with self.db() as db:
            rows = db.execute("SELECT * FROM incidents WHERE expires>? ORDER BY rowid", (self.clock(),)).fetchall()
            for row in rows:
                data = self.unseal(row)
                job = data.get("job")
                if not job:
                    continue
                if job["state"] == "leased" and job["lease_until"] <= self.clock():
                    if data["state"] == "captured" and not data["trials"]:
                        job.update(state="queued", lease_token=None, lease_until=None)
                    else:
                        # A started call may have been billed. Never replay it after a crash.
                        job["state"] = "completed" if data["state"] == "awaiting_reviews" else "interrupted_unknown_usage"
                        if data["state"] == "running":
                            data["state"] = "incomplete"
                    self.save(db, row, data)
                if job["state"] != "queued":
                    continue
                token = secrets.token_hex(32)
                job.update(state="leased", lease_token=token, lease_until=self.clock() + lease_seconds)
                self.save(db, row, data)
                return {"id": row["id"], "owner": job["owner"], "kind": job["kind"], "token": token}
        return None

    def heartbeat(self, identifier, owner, token, *, lease_seconds=60):
        with self.db() as db:
            row, data = self.load(db, identifier, owner)
            self._check_lease(data, token)
            data["job"]["lease_until"] = self.clock() + lease_seconds
            self.save(db, row, data)

    def finish_job(self, identifier, owner, token, state):
        with self.db() as db:
            row, data = self.load(db, identifier, owner)
            job = data.get("job")
            if not job or job["state"] != "leased" or not secrets.compare_digest(job["lease_token"] or "", token):
                raise IncidentConflict("stale worker cannot finalize a job")
            if job["lease_until"] <= self.clock():
                raise IncidentConflict("expired worker cannot finalize a job")
            job["state"] = "cancelled" if job["cancel_requested"] else state
            job.update(finished_at=self.clock(), lease_token=None, lease_until=None)
            if data["state"] == "running":
                data["state"] = "incomplete"
            self.save(db, row, data)

    def cancel(self, identifier, owner):
        with self.db() as db:
            row, data = self.load(db, identifier, owner)
            job = data.get("job")
            if not job or job["state"] not in {"queued", "leased"}:
                raise IncidentConflict("no active investigation job")
            job["cancel_requested"] = True
            if job["state"] == "queued":
                job["state"] = "cancelled"
                data["state"] = "cancelled"
            self.save(db, row, data)
        return self.report(identifier, owner)

    def review_bundle(self, identifier: str, actor: str) -> dict:
        with self.db() as db:
            # Explicit grant gives access to sensitive reproduction content, not only labels.
            return self.load(db, identifier, actor, reviewer=True)[1]

    def grant(self, identifier: str, owner: str, reviewer: str):
        principal = self.principal(reviewer)
        if principal == self.principal(owner):
            raise IncidentConflict("the case owner cannot independently review their own replay")
        with self.db() as db:
            row, data = self.load(db, identifier, owner)
            if principal not in data["reviewers"]:
                if len(data["reviewers"]) >= 10:
                    raise IncidentConflict("reviewer grant capacity reached")
                data["reviewers"].append(principal)
            self.save(db, row, data)

    def revoke(self, identifier: str, owner: str, reviewer: str):
        with self.db() as db:
            row, data = self.load(db, identifier, owner)
            principal = self.principal(reviewer)
            data["reviewers"] = [item for item in data["reviewers"] if item != principal]
            self.save(db, row, data)

    def claim(self, identifier: str, owner: str, kind: str, *, required_seconds=0, lease_token=None):
        if kind not in {"live_provider", "extractive_control"}:
            raise ValueError("unknown replay execution kind")
        with self.db() as db:
            row, data = self.load(db, identifier, owner)
            self._check_lease(data, lease_token)
            if data.get("job") and data["job"]["kind"] != kind:
                raise IncidentConflict("worker execution kind differs from authorization")
            if data["state"] != "captured":
                raise IncidentConflict("case already admitted; uncertain or interrupted spend cannot be retried automatically")
            if data["runtime_fingerprint"] != runtime_fingerprint():
                raise IncidentConflict("replay adapter source changed since capture; create a newly declared study")
            if row["expires"] - self.clock() <= required_seconds:
                raise IncidentConflict("case expires before its reserved replay window")
            data.update(state="running", execution_kind=kind, admitted_at=self.clock())
            self.save(db, row, data)

    def start_trial(self, identifier: str, owner: str, trial: dict, *, lease_token=None):
        with self.db() as db:
            row, data = self.load(db, identifier, owner)
            self._check_lease(data, lease_token)
            if data["state"] != "running" or any(item["trial_id"] == trial["trial_id"] for item in data["trials"]):
                raise IncidentConflict("trial admission is not unique")
            data["trials"].append(trial)
            self.save(db, row, data)

    def finish_trial(self, identifier: str, owner: str, trial: dict, *, lease_token=None):
        with self.db() as db:
            row, data = self.load(db, identifier, owner)
            self._check_lease(data, lease_token)
            current = next((item for item in data["trials"] if item["trial_id"] == trial["trial_id"]), None)
            if data["state"] != "running" or current is None or current["status"] != "started":
                raise IncidentConflict("completed trial outcomes are immutable")
            data["trials"][data["trials"].index(current)] = trial
            self.save(db, row, data)

    def finish(self, identifier: str, owner: str, state: str, *, lease_token=None):
        with self.db() as db:
            row, data = self.load(db, identifier, owner)
            self._check_lease(data, lease_token)
            if data["state"] != "running" or state not in {"incomplete", "awaiting_reviews"}:
                raise IncidentConflict("invalid investigation transition")
            data["state"] = state
            self.save(db, row, data)

    def review(self, identifier: str, actor: str, review: TrialReview):
        with self.db() as db:
            row, data = self.load(db, identifier, actor, reviewer=True)
            principal = self.principal(actor)
            if principal == row["tenant"]:
                raise IncidentConflict("independent reviewer identity is required")
            trial = next((item for item in data["trials"] if item["trial_id"] == review.trial_id), None)
            if data["state"] != "awaiting_reviews" or not trial or trial["status"] != "completed" or trial["fingerprint"] != review.trial_fingerprint:
                raise IncidentConflict("review must bind an immutable completed trial")
            label = dict(review.model_dump(), reviewer=principal)
            previous = data["reviews"].get(review.trial_id)
            if previous and previous != label:
                raise IncidentConflict("review labels cannot be replaced")
            data["reviews"][review.trial_id] = label
            self.save(db, row, data)

    def report(self, identifier: str, owner: str) -> dict:
        data = self.get(identifier, owner)
        trials = data["trials"]
        # No prompts, answers, passages, user IDs or review rationales in ordinary reports.
        public = {"id": identifier, "state": data["state"], "bundle_fingerprint": data["bundle_fingerprint"],
            "plan_fingerprint": fingerprint(data["plan"]), "execution_kind": data["execution_kind"],
            "runtime_fingerprint": data["runtime_fingerprint"],
            "calls_admitted": len(trials), "calls_with_usage": sum(
                item["input_tokens"] is not None and item["output_tokens"] is not None for item in trials),
            "reported_input_tokens": sum(item["input_tokens"] or 0 for item in trials),
            "reported_output_tokens": sum(item["output_tokens"] or 0 for item in trials),
            "usage_complete": bool(trials) and all(item["input_tokens"] is not None and item["output_tokens"] is not None for item in trials),
            "trials": [{key: item.get(key) for key in ("trial_id", "arm", "status", "fingerprint", "config_fingerprint", "context_fingerprint", "duration_ms")} for item in trials],
            "reviews_received": len(data["reviews"]), "analysis": analyze(data), "serving_changed": False}
        if data.get("job"):
            public["job"] = {key: data["job"].get(key) for key in ("state", "kind", "queued_at", "finished_at", "cancel_requested")}
        return public

    def shadow_gate(self, identifier: str, owner: str) -> dict:
        data = self.get(identifier, owner)
        proposal = analyze(data)
        gate = {"decision": "hold", "reason": "live_proposal_and_three_fresh_reviewed_shadow_families_required",
                "shadow_families": 0, "serving_changed": False, "candidate_fingerprint": proposal.get("candidate_fingerprint")}
        if proposal["decision"] != "supported_under_replay":
            return gate
        with self.db() as db:
            rows = db.execute("SELECT * FROM incidents WHERE tenant=? AND parent=? AND expires>?", (self.principal(owner), identifier, self.clock())).fetchall()
            shadows = [self.unseal(row) for row in rows]
            exposure_rows = db.execute("SELECT content FROM incident_exposures WHERE tenant=? AND parent=?", (self.principal(owner), identifier)).fetchall()
            exposures = len(exposure_rows)
        gate["shadow_families"] = len(shadows)
        if len(shadows) < 3 or len(shadows) != exposures or any(not row["content"] for row in exposure_rows):
            return gate
        for shadow in shadows:
            result = analyze(shadow)
            if (shadow["bundle"]["origin"] == "synthetic" or shadow["execution_kind"] != "live_provider"
                    or shadow.get("candidate_fingerprint") != proposal["candidate_fingerprint"]
                    or not result.get("candidate_safe_correct") or result.get("regression", True)):
                gate["reason"] = "shadow_missing_labels_regression_or_unsafe_outcome"
                return gate
        gate.update(decision="ready_for_owner_review", reason="small_reviewed_shadow_check_passed_not_production_certification")
        return gate

    def delete(self, identifier: str, owner: str):
        with self.db() as db:
            self.load(db, identifier, owner)
            db.execute("DELETE FROM incidents WHERE id=? OR (parent=? AND tenant=?)", (identifier, identifier, self.principal(owner)))

    def purge_expired(self) -> int:
        with self.db() as db:
            db.execute("DELETE FROM incident_exposures WHERE expires<=?", (self.clock(),))
            return db.execute("DELETE FROM incidents WHERE expires<=?", (self.clock(),)).rowcount
