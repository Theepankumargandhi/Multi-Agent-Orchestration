"""Frozen authored migration controls with the real codemod and scripted specialist repairs."""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
from pathlib import Path

from code_agent.models import AnalysisPlan, CodeAction, ReviewVerdict
from code_agent.upgrade_models import MigrationSpec, fingerprint
from code_agent.upgrade_runner import run_migration

ROOT = Path(__file__).resolve().parent.parent
FIXTURES = ROOT / "tests/fixtures/upgradepilot"
CASES = ("constraints", "union", "validator")


class ScriptedSpecialist:
    """Authored control, not a model prediction or independently reviewed real migration."""
    def __init__(self, role, family):
        self.role, self.family, self.calls = role, family, 0

    async def ainvoke(self, messages, config=None):
        self.calls += 1
        if self.role == "analysis":
            return AnalysisPlan(summary="Authored migration control: preserve source contract behavior.")
        if self.role == "review":
            return ReviewVerdict(approved=True, summary="Scripted review control. Independent execution gates remain mandatory.")
        if self.calls == 1:
            return CodeAction(kind="read_lines", path="app/models.py", end_line=100)
        if self.calls == 2:
            data = json.loads(json.loads(messages[1].content)["recent_observations"][0]["output"])
            old = data["text"]
            if self.family == "union":
                new = old.replace("from pydantic import BaseModel", "from pydantic import BaseModel, Field").replace(
                    "value: Union[int, str]", "value: Union[int, str] = Field(union_mode='left_to_right')")
            elif self.family == "validator":
                new = old.replace("validator", "field_validator").replace("value, field", "value, info").replace("field.name", "info.field_name")
                # The real codemod may already have changed the decorator/import.
                new = new.replace("field_field_validator", "field_validator")
            else:
                new = old
            return CodeAction(kind="edit", path="app/models.py", expected_sha256=data["file_sha256"], old_text=old, new_text=new)
        return CodeAction(kind="test")  # Successful tests complete each specialist stage.


def corpus_fingerprint():
    files = sorted(path for name in CASES for path in (FIXTURES / name).rglob("*") if path.is_file() and "__pycache__" not in path.parts)
    return fingerprint({path.relative_to(FIXTURES).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest() for path in files})


async def evaluate():
    lock = json.loads((ROOT / "evals/experiments/upgradepilot.lock.json").read_text(encoding="utf-8"))
    if lock["corpus_fingerprint"] != corpus_fingerprint() or lock["cases"] != list(CASES):
        raise ValueError("migration control corpus changed; review and explicitly update its lock")
    rows = []
    for family in CASES:
        repository = FIXTURES / family
        spec = MigrationSpec.model_validate_json((repository / "migration.json").read_text(encoding="utf-8"))
        models = tuple(ScriptedSpecialist(role, family) for role in ("analysis", "actions", "review"))
        report = await run_migration(repository, spec, scripted_models=models)
        baseline_pass = bool(report.get("baseline") and report["baseline"]["passed"])
        ready = report["status"] == "ready_for_owner_review"
        rows.append({"case": family, "baseline_passed": baseline_pass, "final_passed": ready,
            "passed": ready and report["source_unchanged"] and report["live_provider_calls"] == 0
                and (family == "constraints" or (not baseline_pass and report["repair"] and report["repair"]["passed"])),
            "report": report})
    return {"evidence_kind": "authored_frozen_codemod_and_scripted_specialists_real_docker", "corpus_fingerprint": lock["corpus_fingerprint"],
        "live_provider_calls": 0, "human_reviews": 0, "real_maintainer_migrations_measured": False,
        "total": len(rows), "passed": sum(bool(row["passed"]) for row in rows), "outcomes": rows,
        "interpretation": "Control sensitivity and workflow verification, not AI superiority over bump-pydantic."}


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
