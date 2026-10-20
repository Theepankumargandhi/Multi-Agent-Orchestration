"""Explicit source-backed re-extraction; never rewrite stored evidence or grant approval."""

import hashlib
import json
import re
import time
from contextlib import closing
from pathlib import Path
from uuid import uuid4

from agentforge_sdk import load_claim_replay_bundle, verify_claim_replay_bundle
from agentforge_sdk.claim_bundle import ClaimBundleError
from agentforge_sdk.models import fingerprint
from agentforge_sdk.spatial import MultimodalDocument
from service.document_cell_concordance import check_cells, concordance_defaults
from service.document_claim_audits import DocumentClaimAudits
from service.document_store import LibraryError


class DocumentSourceReplay:
    def __init__(self, library):
        self.library = library
        self.audits = DocumentClaimAudits(library)

    def _context(self, db, owner, set_id, audit_id, source_id, expected):
        row, binding, checks, inputs, evidence, _, _ = self.audits._load(db, owner, set_id, audit_id)
        if source_id not in evidence.source_ids:
            raise LibraryError(404, "audit_source_unavailable")
        if row["checkpoint_fingerprint"] != expected:
            raise LibraryError(409, "source_replay_checkpoint_changed")
        if len(checks) != len(binding["claims"]):
            raise LibraryError(409, "complete_source_audit_first")
        source = self.library.row(db, owner, source_id)
        original_row = db.execute(
            "SELECT length(original) AS size,CASE WHEN typeof(original)='blob' AND length(original) BETWEEN 1 AND 3000000 THEN original ELSE NULL END AS body FROM documents WHERE id=? AND owner=?",
            (source_id, owner),
        ).fetchone()
        original = original_row["body"] or b""
        document = MultimodalDocument.model_validate_json(source["manifest"])
        signature = fingerprint(
            dict(
                audit_checkpoint=expected,
                audit_binding=row["digest"],
                collection=evidence.fingerprint,
                manifest=document.fingerprint,
                name=source["name"],
                version=source["version"],
                family=source["family_id"],
                ocr=source["ocr"],
                digest=source["digest"],
                original_length=original_row["size"],
                raw_hash=hashlib.sha256(original).hexdigest(),
            )
        )
        return source, original, document, signature, (binding, checks, inputs, evidence)

    def replay(self, owner, set_id, audit_id, source_id, request):
        return self._run(owner, set_id, audit_id, source_id, request, operation="replay")

    def crosscheck(self, owner, set_id, audit_id, source_id, request):
        return self._run(owner, set_id, audit_id, source_id, request, operation="text")

    def crosscheck_pixels(self, owner, set_id, audit_id, source_id, request):
        return self._run(owner, set_id, audit_id, source_id, request, operation="pixels")

    def _run(self, owner, set_id, audit_id, source_id, request, *, operation):
        crosscheck, pixels = operation != "replay", operation == "pixels"
        token, started = uuid4().hex, time.monotonic()
        with closing(self.library.connect()) as db, db:
            db.execute("BEGIN IMMEDIATE")
            source, raw, expected_doc, signature, context = self._context(
                db, owner, set_id, audit_id, source_id, request["checkpoint_fingerprint"]
            )
            if not crosscheck and source["ocr"] and not request["allow_original_ocr"]:
                raise LibraryError(409, "source_replay_ocr_consent_required")
            try:
                bundle = self.audits._bundle(*context)
            except ClaimBundleError:
                raise LibraryError(409, "source_replay_audit_outside_bundle_budget") from None
            report = {
                "protocol": "agentforge-pixel-concordance-v1"
                if pixels
                else "agentforge-cell-concordance-v1"
                if crosscheck
                else "agentforge-source-backed-replay-v1",
                "source_id": source_id,
                "audit_checkpoint_fingerprint": request["checkpoint_fingerprint"],
                "collection_fingerprint": bundle.collection_fingerprint,
                "bundle_fingerprint": bundle.fingerprint,
                "expected_manifest_fingerprint": expected_doc.fingerprint,
                "original_bytes_match": False,
                "runtime_image_id": None,
                "publisher_identity_verified": False,
                "release_authorized": False,
                "model_calls": 0,
            }
            report.update(
                concordance_defaults(pixels=pixels)
                if crosscheck
                else {
                    "fresh_manifest_fingerprint": None,
                    "profile_matches": None,
                    "snapshot_matches": False,
                    "extraction_reexecuted": False,
                    "numeric_checks_reproduced": 0,
                    "numeric_counts": None,
                }
            )
            if not 0 < len(raw) <= 3_000_000:
                return self._finish(report, "original_outside_budget", started)
            digest = hashlib.sha256(raw).hexdigest()
            if digest != source["digest"] or digest != expected_doc.file_sha256:
                return self._finish(report, "original_bytes_mismatch", started)
            report["original_bytes_match"] = True
            if self.library.runtime_busy(db):
                raise LibraryError(429, "document_processor_busy")
            db.execute("INSERT INTO document_runtime_lease VALUES (1,?,?)", (token, time.time()))
        try:
            try:
                result = (
                    check_cells(self.library.runtime, raw, source_id, expected_doc, context[2], pixels=pixels)
                    if crosscheck
                    else self._reextract(raw, source, expected_doc, bundle)
                )
            except Exception:
                result = {"status": "extraction_failed"}
            status = result.pop("status")
            report.update(result)
            # A request finishing after deletion, mutation or lease expiry cannot report success.
            with closing(self.library.connect()) as db, db:
                db.execute("BEGIN IMMEDIATE")
                current = self._context(
                    db, owner, set_id, audit_id, source_id, request["checkpoint_fingerprint"]
                )
                lease = db.execute("SELECT token FROM document_runtime_lease WHERE slot=1").fetchone()
                if current[3] != signature or lease is None or lease["token"] != token:
                    raise LibraryError(409, "source_replay_context_changed")
            return self._finish(report, status, started)
        finally:
            with closing(self.library.connect()) as db, db:
                db.execute("DELETE FROM document_runtime_lease WHERE slot=1 AND token=?", (token,))

    def _reextract(self, raw, source, expected_doc, bundle):
        extracted = self.library.runtime.extract(
            raw,
            name=source["name"],
            suffix=Path(source["name"]).suffix.lower(),
            document_id=source["family_id"],
            version=source["version"],
            ocr=bool(source["ocr"]),
        )
        fresh = MultimodalDocument.model_validate_json(json.dumps(extracted["manifest"]))
        image_id = extracted.get("runtime_image_id")
        report = {
            "extraction_reexecuted": True,
            "runtime_image_id": image_id
            if isinstance(image_id, str) and re.fullmatch(r"sha256:[a-f0-9]{64}", image_id)
            else None,
            "fresh_manifest_fingerprint": fresh.fingerprint,
            "profile_matches": fresh.config == expected_doc.config
            and fresh.extractor == expected_doc.extractor
            and fresh.ocr_backend == expected_doc.ocr_backend,
        }
        if (
            fresh.file_sha256 != expected_doc.file_sha256
            or fresh.document_id != expected_doc.document_id
            or fresh.version != expected_doc.version
            or fresh.source != expected_doc.source
            or fresh.media_type != expected_doc.media_type
        ):
            status = "runtime_binding_mismatch"
        elif not fresh.ready:
            status = "extraction_held"
        elif not report["profile_matches"]:
            status = "profile_changed"
        elif fresh.fingerprint != expected_doc.fingerprint:
            status = "snapshot_changed"
        else:
            # Recompute using the fresh manifest, retaining every other frozen source.
            payload = bundle.model_dump(mode="json")
            next(s for s in payload["sources"] if s["source_id"] == source["id"])["document"] = (
                fresh.model_dump(mode="json")
            )
            refreshed = load_claim_replay_bundle(json.dumps(payload))
            verified = verify_claim_replay_bundle(refreshed, expected_fingerprint=bundle.fingerprint)
            report.update(
                snapshot_matches=True,
                numeric_checks_reproduced=verified.total,
                numeric_counts={
                    key: getattr(verified, key) for key in ("supported", "refuted", "unresolved")
                },
            )
            status = "exact_snapshot_match"
        return {**report, "status": status}

    @staticmethod
    def _finish(report, status, started):
        payload = {
            **report,
            "status": status,
            "checked_at": time.time(),
            "elapsed_seconds": max(0.0, time.monotonic() - started),
        }
        return {**payload, "fingerprint": fingerprint(payload)}
