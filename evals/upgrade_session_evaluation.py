"""Authored sequential migration controls, actual Docker and scripted repairs, no inference."""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
from pathlib import Path

from code_agent.models import CodeAction
from code_agent.upgrade_models import MigrationSpec, fingerprint
from code_agent.upgrade_runner import run_migration
from code_agent.upgrade_sessions import SessionPolicy, session_commitment
from evals.upgradepilot_evaluation import ROOT, ScriptedSpecialist

FIXTURE = ROOT / "tests/fixtures/upgrade_sessions"
POLICY = ROOT / "evals/experiments/upgrade_sessions.policy.json"
LOCK = ROOT / "evals/experiments/upgrade_sessions.lock.json"


def recipe():
    return MigrationSpec.model_validate_json((FIXTURE / "migration.json").read_bytes())


def policy():
    return SessionPolicy.model_validate_json(POLICY.read_bytes())


def corpus_fingerprint():
    return fingerprint({path.relative_to(FIXTURE).as_posix(): hashlib.sha256(path.read_text(encoding="utf-8").encode()).hexdigest()
        for path in sorted(FIXTURE.rglob("*")) if path.is_file() and "__pycache__" not in path.parts})


class SessionSpecialist(ScriptedSpecialist):
    """Fixture-aware controls: full fix versus intentional discovery-only overfit."""
    def __init__(self, role, overfit=False):
        super().__init__(role, "stateful")
        self.overfit, self.messages = overfit, []

    async def ainvoke(self, messages, config=None):
        self.messages.extend(str(row.content) for row in messages)
        result = await super().ainvoke(messages, config)
        if isinstance(result, CodeAction) and result.kind == "edit":
            text = result.new_text
            if self.overfit:
                text = text.replace("    def add(self, quantity):", "    def add(self, quantity):\n        if quantity == '12':\n            quantity = 12")
            else:
                text = text.replace("from pydantic import BaseModel", "from pydantic import BaseModel, Field")
                text = text.replace("quantity: Union[int, str]", "quantity: Union[int, str] = Field(union_mode='left_to_right')")
            result = result.model_copy(update={"new_text": text})
        return result


async def evaluate():
    locked = json.loads(LOCK.read_bytes())
    profile = policy()
    if locked["corpus_fingerprint"] != corpus_fingerprint() or locked["session_commitment"] != session_commitment(profile):
        raise ValueError("session controls changed; review the frozen lock")
    rows = []
    for name in ("codemod_only", "session_repair", "discovery_overfit", "revoked_session"):
        models = tuple(SessionSpecialist(role, name == "discovery_overfit") for role in ("analysis", "actions", "review")) if name != "codemod_only" else None
        report = await run_migration(FIXTURE, recipe(), session_policy=profile, scripted_models=models,
            expected_session_commitment="0" * 64 if name == "revoked_session" else locked["session_commitment"])
        discovery = report.get("session_discovery", {})
        final = report.get("final_session_evaluation", {})
        leaked = any("withheld-rollback-and-reset" in message or '"37"' in message
            for specialist in models or () for message in specialist.messages)
        if name == "codemod_only":
            expected = (report["baseline"]["pre_session_gates_passed"] and discovery.get("regressions") == 1
                and report["status"] == "held" and not final)
        elif name == "revoked_session":
            expected = report["reason"] == "hold_at:preflight" and not report["model_calls"] and not discovery and not final
        else:
            expected = (discovery.get("regressions") == 1 and report.get("session_recheck", {}).get("passed")
                and bool(final.get("passed")) == (name == "session_repair")
                and report["status"] == ("ready_for_owner_review" if name == "session_repair" else "held"))
        rows.append({"arm": name, "passed": bool(expected and not leaked and report["source_unchanged"]
            and not report["live_provider_calls"]), "withheld_inputs_in_model_messages": leaked, "report": report})
        print(json.dumps({"arm": name, "passed": rows[-1]["passed"], "status": report["status"], "phase": report["phase"]}), flush=True)
    return {"evidence_kind": "authored_frozen_stateful_controls_actual_docker_scripted_specialists",
        "corpus_fingerprint": locked["corpus_fingerprint"], "session_commitment": locked["session_commitment"],
        "total": len(rows), "passed": sum(row["passed"] for row in rows), "outcomes": rows,
        "live_provider_calls": 0, "real_repository_migrations_measured": False,
        "interpretation": "Control sensitivity and final-gate isolation, not general model repair quality or production impact."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = asyncio.run(evaluate())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "outcomes"}, indent=2))
    if result["passed"] != result["total"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
