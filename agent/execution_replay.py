"""Private, consented observations and immutable delayed correctness labels.

This is not an RL transition log: planning actions and propensities are never
inferred from deterministic search. Only selection-evaluated candidates enter it.
"""

from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import os
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Callable, Literal

from pydantic import BaseModel, ConfigDict, Field

from agent.adaptive_compute import ComputePlan, DeliberationReceipt, verify_plan, verify_receipt
from agent.uncertainty import CalibrationExample


def digest(key: bytes, domain: str, value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hmac.new(key, domain.encode() + b"\0" + payload, hashlib.sha256).hexdigest()


class Observation(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    schema_version: Literal["1.0"] = "1.0"
    event_id: str = Field(pattern=r"^[a-f0-9]{64}$")
    tenant: str = Field(pattern=r"^[a-f0-9]{64}$")
    request_group: str = Field(pattern=r"^[a-f0-9]{64}$")
    candidate: str = Field(pattern=r"^[a-f0-9]{64}$")
    receipt: str = Field(pattern=r"^[a-f0-9]{64}$")
    origin: Literal["runtime", "synthetic"]
    route: Literal["web", "hybrid", "rag", "kg", "math", "general", "code"]
    confidence: float = Field(ge=0, le=1)
    grounded: bool
    selected: bool
    high_risk: bool
    estimated_output_tokens: int = Field(ge=0)
    latency_ms: float = Field(ge=0)
    observed_at: float = Field(ge=0)
    fingerprint: str = ""


class ReviewLabel(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    event_id: str = Field(pattern=r"^[a-f0-9]{64}$")
    observation_fingerprint: str = Field(pattern=r"^[a-f0-9]{64}$")
    verdict: Literal["correct", "incorrect", "ambiguous"]
    unsafe: bool
    reviewer: str = Field(pattern=r"^[a-f0-9]{64}$")
    reviewed_at: float = Field(ge=0)
    fingerprint: str = ""


class RequestFamily(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    request_group: str = Field(pattern=r"^[a-f0-9]{64}$")
    tenant: str = Field(pattern=r"^[a-f0-9]{64}$")
    task_family: str = Field(default="", pattern=r"^(|[a-f0-9]{64})$")
    captured_at: float = Field(ge=0)
    fingerprint: str = ""


def _seal(item: Observation | ReviewLabel | RequestFamily, key: bytes):
    item.fingerprint = digest(key, type(item).__name__, item.model_dump(exclude={"fingerprint"}))
    return item


def _verify(item: Observation | ReviewLabel | RequestFamily, key: bytes) -> None:
    expected = digest(key, type(item).__name__, item.model_dump(exclude={"fingerprint"}))
    if not hmac.compare_digest(expected, item.fingerprint):
        raise ValueError("replay integrity verification failed")


class ExecutionReplayStore:
    def __init__(self, path: str | Path, key: bytes, *, clock: Callable[[], float] | None = None):
        if len(key) < 32:
            raise ValueError("replay requires an independent key of at least 32 bytes")
        self.path, self.key = Path(path), key
        self.clock = clock or time.time
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._db() as db:
            db.execute("PRAGMA journal_mode=WAL")
            db.execute(
                "CREATE TABLE IF NOT EXISTS observations (event_id TEXT PRIMARY KEY, tenant TEXT NOT NULL, payload TEXT NOT NULL)"
            )
            db.execute("CREATE INDEX IF NOT EXISTS replay_tenant ON observations(tenant)")
            db.execute(
                "CREATE TABLE IF NOT EXISTS labels (event_id TEXT PRIMARY KEY REFERENCES observations(event_id) ON DELETE CASCADE, payload TEXT NOT NULL)"
            )
            db.execute(
                "CREATE TABLE IF NOT EXISTS request_families (request_group TEXT PRIMARY KEY, tenant TEXT NOT NULL, payload TEXT NOT NULL)"
            )

    @contextmanager
    def _db(self):
        db = sqlite3.connect(self.path, timeout=5)
        db.execute("PRAGMA foreign_keys=ON")
        try:
            with db:
                yield db
        finally:
            db.close()

    def capture(
        self,
        plan: ComputePlan,
        receipt: DeliberationReceipt,
        *,
        tenant: str,
        request_id: str,
        consent: bool,
        compute_key: bytes | None = None,
        origin: Literal["runtime", "synthetic"] = "runtime",
        task_family: str | None = None,
        task_family_fingerprint: str | None = None,
        on_capture: Callable[[sqlite3.Connection], None] | None = None,
    ) -> list[str]:
        if consent is not True or not tenant.strip() or not request_id.strip():
            return []
        if not verify_plan(plan, compute_key) or not verify_receipt(receipt, compute_key):
            raise ValueError("compute integrity verification failed")
        if receipt.plan_fingerprint != plan.plan_fingerprint:
            raise ValueError("receipt does not belong to plan")
        if task_family is not None and (not task_family.strip() or len(task_family) > 128):
            raise ValueError("task family must contain 1 to 128 characters")
        if task_family is not None and task_family_fingerprint is not None:
            raise ValueError("provide a raw family ID or a prehashed family, not both")
        group = digest(self.key, "request", [tenant, request_id])
        tenant_hash = digest(self.key, "tenant", tenant)
        rows = []
        for summary in receipt.candidate_summaries:
            candidate = digest(self.key, "candidate", summary.candidate_id)
            event = _seal(
                Observation(
                    event_id=digest(self.key, "event", [group, candidate]),
                    tenant=tenant_hash,
                    request_group=group,
                    candidate=candidate,
                    receipt=digest(self.key, "receipt", receipt.receipt_fingerprint),
                    origin=origin,
                    route=plan.route,
                    confidence=summary.confidence,
                    grounded=summary.grounded,
                    high_risk=plan.high_risk,
                    selected=summary.candidate_id == receipt.selected_candidate_id,
                    estimated_output_tokens=summary.token_count,
                    latency_ms=summary.latency_ms,
                    observed_at=self.clock(),
                ),
                self.key,
            )
            rows.append(event)
        if len({row.event_id for row in rows}) != len(rows):
            raise ValueError("duplicate candidate identities")
        with self._db() as db:
            db.execute("BEGIN IMMEDIATE")
            if rows:
                family_hash = (
                    task_family_fingerprint
                    if task_family_fingerprint is not None
                    else digest(self.key, "task-family", [tenant, task_family.strip()])
                    if task_family
                    else ""
                )
                existing_family = db.execute(
                    "SELECT payload FROM request_families WHERE request_group=?", (group,)
                ).fetchone()
                if existing_family:
                    previous_family = RequestFamily.model_validate_json(existing_family[0])
                    _verify(previous_family, self.key)
                    if previous_family.request_group != group or previous_family.tenant != tenant_hash:
                        raise ValueError("family index integrity verification failed")
                    if (
                        task_family is not None or task_family_fingerprint is not None
                    ) and previous_family.task_family != family_hash:
                        raise ValueError("task family is immutable after first capture")
                else:
                    # Legacy observations remain valid but cannot acquire a family
                    # after their outcomes have already been observed/reviewed.
                    legacy = db.execute(
                        "SELECT 1 FROM observations WHERE tenant=? AND json_extract(payload, '$.request_group')=? LIMIT 1",
                        (tenant_hash, group),
                    ).fetchone()
                    if legacy and family_hash:
                        raise ValueError("task family must be assigned at first capture")
                    family = _seal(
                        RequestFamily(
                            request_group=group,
                            tenant=tenant_hash,
                            task_family=family_hash,
                            captured_at=min(row.observed_at for row in rows),
                        ),
                        self.key,
                    )
                    db.execute(
                        "INSERT INTO request_families VALUES (?, ?, ?)",
                        (group, tenant_hash, family.model_dump_json()),
                    )
            for event in rows:
                existing = db.execute(
                    "SELECT payload FROM observations WHERE event_id=?", (event.event_id,)
                ).fetchone()
                if existing:
                    previous = Observation.model_validate_json(existing[0])
                    _verify(previous, self.key)
                    ignored = {"fingerprint", "observed_at"}
                    if previous.model_dump(exclude=ignored) != event.model_dump(exclude=ignored):
                        raise ValueError("conflicting replay retry")
                else:
                    db.execute(
                        "INSERT INTO observations VALUES (?, ?, ?)",
                        (event.event_id, tenant_hash, event.model_dump_json()),
                    )
            # Internal deployment binding must commit with the observations.
            # A failed hook rolls back both; callbacks must not do external I/O.
            if on_capture is not None:
                on_capture(db)
        return [row.event_id for row in rows]

    def records(self, tenant: str) -> list[tuple[Observation, ReviewLabel | None]]:
        return self.snapshot(tenant)[0]

    def snapshot(
        self, tenant: str
    ) -> tuple[list[tuple[Observation, ReviewLabel | None]], dict[str, RequestFamily]]:
        """Observations, labels and family assignments from one read transaction."""
        tenant_hash = digest(self.key, "tenant", tenant)
        with self._db() as db:
            db.execute("BEGIN")
            rows = db.execute(
                "SELECT o.event_id, o.payload, l.payload FROM observations o LEFT JOIN labels l ON o.event_id=l.event_id WHERE o.tenant=? ORDER BY o.event_id",
                (tenant_hash,),
            ).fetchall()
            family_rows = db.execute(
                "SELECT request_group, payload FROM request_families WHERE tenant=?", (tenant_hash,)
            ).fetchall()
        families = {}
        for group, payload in family_rows:
            family = RequestFamily.model_validate_json(payload)
            _verify(family, self.key)
            if group != family.request_group or family.tenant != tenant_hash:
                raise ValueError("family index integrity verification failed")
            families[group] = family
        result = []
        for event_id, payload, label_payload in rows:
            event = Observation.model_validate_json(payload)
            _verify(event, self.key)
            if event.event_id != event_id or event.tenant != tenant_hash:
                raise ValueError("replay index integrity verification failed")
            label = ReviewLabel.model_validate_json(label_payload) if label_payload else None
            if label:
                _verify(label, self.key)
                if label.event_id != event_id or label.observation_fingerprint != event.fingerprint:
                    raise ValueError("label observation binding failed")
            result.append((event, label))
        return result, families

    def review(self, tenant: str, event_id: str, *, verdict: str, unsafe: bool, reviewer: str) -> ReviewLabel:
        if not reviewer.strip():
            raise ValueError("reviewer identity is required")
        # BEGIN IMMEDIATE makes the read + immutable-label write one transaction.
        with self._db() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute(
                "SELECT payload FROM observations WHERE event_id=? AND tenant=?",
                (event_id, digest(self.key, "tenant", tenant)),
            ).fetchone()
            if row is None:
                raise ValueError("event not found for tenant")
            event = Observation.model_validate_json(row[0])
            _verify(event, self.key)
            if event.event_id != event_id or event.tenant != digest(self.key, "tenant", tenant):
                raise ValueError("replay index integrity verification failed")
            label = _seal(
                ReviewLabel(
                    event_id=event_id,
                    observation_fingerprint=event.fingerprint,
                    verdict=verdict,
                    unsafe=unsafe,
                    reviewer=digest(self.key, "reviewer", reviewer),
                    reviewed_at=self.clock(),
                ),
                self.key,
            )
            previous = db.execute("SELECT payload FROM labels WHERE event_id=?", (event_id,)).fetchone()
            if previous:
                stored = ReviewLabel.model_validate_json(previous[0])
                _verify(stored, self.key)
                ignored = {"reviewed_at", "fingerprint"}
                if stored.model_dump(exclude=ignored) != label.model_dump(exclude=ignored):
                    raise ValueError("review labels are immutable; conflicting retry")
                return stored
            db.execute("INSERT INTO labels VALUES (?, ?)", (event_id, label.model_dump_json()))
        return label

    def delete_tenant(self, tenant: str) -> int:
        with self._db() as db:
            db.execute("DELETE FROM request_families WHERE tenant=?", (digest(self.key, "tenant", tenant),))
            deleted = db.execute(
                "DELETE FROM observations WHERE tenant=?", (digest(self.key, "tenant", tenant),)
            ).rowcount
            if db.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='preference_shadow_studies'").fetchone():
                db.execute("DELETE FROM preference_shadow_studies WHERE tenant=?", (digest(self.key, "tenant", tenant),))
            for table in ("preference_served", "preference_deployments", "preference_deployment_states", "preference_deployment_audit", "process_step_reviews", "process_snapshots"):
                if db.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)).fetchone():
                    db.execute(f"DELETE FROM {table} WHERE tenant=?", (digest(self.key, "tenant", tenant),))
            return deleted

    def dataset(self, tenant: str, *, allow_synthetic: bool = False) -> tuple[list[CalibrationExample], dict]:
        """One pre-label-selected candidate per request; stable request-level holdout.

        Selected answer if one exists, otherwise smallest opaque event ID. Choice
        uses no labels. All other candidates are excluded, not independent samples.
        """
        groups: dict[str, list] = {}
        for event, label in self.records(tenant):
            groups.setdefault(event.request_group, []).append((event, label))
        examples, lineage, exclusions = [], [], {}
        for group, rows in sorted(groups.items()):
            event, label = min(rows, key=lambda row: (not row[0].selected, row[0].event_id))
            reason = (
                "synthetic"
                if event.origin == "synthetic" and not allow_synthetic
                else "unreviewed"
                if label is None
                else "ambiguous"
                if label.verdict == "ambiguous"
                else "unsupported_route"
                if event.route not in {"web", "hybrid", "rag", "kg", "math"}
                else ""
            )
            if reason:
                exclusions[reason] = exclusions.get(reason, 0) + 1
                continue
            split = "test" if int(digest(self.key, "split-v1", group)[:8], 16) % 4 == 0 else "calibration"
            examples.append(
                CalibrationExample(
                    id=event.event_id,
                    split=split,
                    route=event.route,
                    confidence=event.confidence,
                    correct=label.verdict == "correct" and not label.unsafe,
                    high_risk=event.high_risk,
                )
            )
            lineage.append(
                {
                    "group": group,
                    "event_id": event.event_id,
                    "event": event.fingerprint,
                    "label": label.fingerprint,
                    "unsafe": label.unsafe,
                    "split": split,
                }
            )
        manifest = {
            "schema_version": "1.0",
            "split_policy": "hmac-request-group-v1-75-25",
            "candidate_policy": "selected-else-smallest-event-id-before-labels",
            "simulation": allow_synthetic,
            "request_groups": len(groups),
            "exported_groups": len(examples),
            "excluded_groups": exclusions,
            "excluded_sibling_candidates": sum(max(0, len(rows) - 1) for rows in groups.values()),
            "lineage": lineage,
        }
        manifest["fingerprint"] = digest(self.key, "dataset", manifest)
        return examples, manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store", default="data/execution-replay/replay.sqlite3")
    parser.add_argument("--tenant", required=True)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("list")
    review = commands.add_parser("review")
    review.add_argument("event_id")
    review.add_argument("--verdict", choices=["correct", "incorrect", "ambiguous"], required=True)
    review.add_argument("--unsafe", action="store_true")
    review.add_argument("--reviewer", required=True)
    export = commands.add_parser("export")
    export.add_argument("--output", default="data/execution-replay/reviewed.json")
    commands.add_parser("delete-tenant")
    args = parser.parse_args()
    store = ExecutionReplayStore(args.store, os.getenv("EXECUTION_REPLAY_KEY", "").encode())
    if args.command == "list":
        print(
            json.dumps(
                [
                    {"event": event.model_dump(), "review": label.model_dump() if label else None}
                    for event, label in store.records(args.tenant)
                ],
                indent=2,
            )
        )
    elif args.command == "review":
        print(
            store.review(
                args.tenant, args.event_id, verdict=args.verdict, unsafe=args.unsafe, reviewer=args.reviewer
            ).model_dump_json(indent=2)
        )
    elif args.command == "delete-tenant":
        print(json.dumps({"deleted_observations": store.delete_tenant(args.tenant)}))
    else:
        examples, manifest = store.dataset(args.tenant)
        target = Path(args.output)
        if target.resolve() == store.path.resolve():
            parser.error("export must not overwrite the replay store")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(
                {"examples": [item.model_dump() for item in examples], "manifest": manifest}, indent=2
            ),
            encoding="utf-8",
        )
        print(json.dumps({"exported_groups": len(examples), "fingerprint": manifest["fingerprint"]}))


if __name__ == "__main__":
    main()
