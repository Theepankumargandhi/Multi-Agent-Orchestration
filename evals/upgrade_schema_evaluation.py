"""Authored nested-model ablation with actual schemas and Docker execution, not live AI."""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
from pathlib import Path

from code_agent.models import CodeAction
from code_agent.upgrade_inputs import CounterexamplePolicy
from code_agent.upgrade_models import MigrationSpec, fingerprint
from code_agent.upgrade_runner import run_migration
from code_agent.upgrade_schema import SchemaPolicy
from evals.upgradepilot_evaluation import ROOT, ScriptedSpecialist

FIXTURE = ROOT / "tests/fixtures/upgrade_schema"
LOCK = ROOT / "evals/experiments/upgrade_schema.lock.json"


def corpus_fingerprint():
    return fingerprint({path.relative_to(FIXTURE).as_posix(): hashlib.sha256(path.read_text(encoding="utf-8").encode()).hexdigest()
        for path in sorted(FIXTURE.rglob("*")) if path.is_file() and "__pycache__" not in path.parts})


class NestedSpecialist(ScriptedSpecialist):
    """Fixture-aware scripted correction; no general model-quality claim."""
    def __init__(self, role):
        super().__init__(role, "nested-schema")
        self.messages = []

    async def ainvoke(self, messages, config=None):
        self.messages.extend(str(message.content) for message in messages)
        result = await super().ainvoke(messages, config)
        if isinstance(result, CodeAction) and result.kind == "edit":
            text = result.new_text.replace("quantity: Union[int, str]", "quantity: Union[int, str] = Field(union_mode='left_to_right')")
            text = text.replace("    request = Request(**payload)",
                "    if isinstance(payload, dict) and isinstance(payload.get('items'), list):\n"
                "        payload = {**payload, 'items': [\n"
                "            {**item, 'quantity': int(item['quantity'])}\n"
                "            if isinstance(item, dict) and isinstance(item.get('quantity'), float) else item\n"
                "            for item in payload['items']]}\n"
                "    request = Request(**payload)")
            result = result.model_copy(update={"new_text": text})
        return result


async def evaluate():
    locked = json.loads(LOCK.read_bytes())
    if locked["corpus_fingerprint"] != corpus_fingerprint():
        raise ValueError("schema fixture changed; review and update the frozen lock")
    spec = MigrationSpec.model_validate_json((FIXTURE / "migration.json").read_bytes())
    schema_policy = SchemaPolicy.model_validate_json((FIXTURE / "schema-policy.json").read_bytes())
    input_policy = CounterexamplePolicy()
    if locked["input_policy"] != input_policy.model_dump():
        raise ValueError("schema study input budget changed")
    rows = []
    for name, guided, repair in (("blind_discovery", False, False), ("schema_discovery", True, False),
            ("schema_repair", True, True), ("revoked_campaign", True, True)):
        models = tuple(NestedSpecialist(role) for role in ("analysis", "actions", "review")) if repair else None
        report = await run_migration(FIXTURE, spec, counterexample_policy=input_policy, schema_policy=schema_policy,
            schema_discovery=guided, scripted_models=models,
            expected_campaign_commitment="0" * 64 if name == "revoked_campaign" else locked["campaign_commitment"])
        discovery = report.get("discovery", {})
        final = report.get("final_evaluation", {})
        if name == "blind_discovery":
            expected = discovery.get("passed") and final and not final["passed"] and report["status"] == "held"
        elif name == "schema_discovery":
            expected = discovery.get("regressions", 0) > 0 and report["status"] == "held" and not final
        elif name == "schema_repair":
            expected = (discovery.get("regressions", 0) > 0 and report.get("discovery_recheck", {}).get("passed")
                and final.get("passed") and report["status"] == "ready_for_owner_review")
        else:
            expected = (report["status"] == "held" and report["reason"] == "hold_at:source_schema_extraction"
                and not report["model_calls"] and not report.get("discovery") and not final)
        declarations = report.get("schema_intelligence", {}).get("baseline_target", {}).get("declarations", {})
        same_declarations = bool(declarations) and not any(value["changed"] for value in declarations.values())
        source_bound = report.get("schema_intelligence", {}).get("source", {}).get("fingerprint") == locked["source_schema_fingerprint"]
        rows.append({"arm": name, "passed": bool(expected and (same_declarations or name == "revoked_campaign")
            and source_bound and report["source_unchanged"] and not report["live_provider_calls"]),
            "equal_normalized_declarations": same_declarations, "report": report})
        print(json.dumps({"arm": name, "passed": rows[-1]["passed"], "phase": report["phase"], "status": report["status"]}), flush=True)
    final_fingerprints = {row["report"].get("input_campaign", {}).get("final_fingerprint") for row in rows[:3]}
    shared_final = len(final_fingerprints) == 1 and None not in final_fingerprints
    return {"evidence_kind": "one_authored_nested_model_real_docker_scripted_repair", "corpus_fingerprint": corpus_fingerprint(),
        "same_frozen_final_inputs": shared_final, "total": len(rows), "passed": sum(row["passed"] for row in rows) if shared_final else 0,
        "live_provider_calls": 0, "real_repository_migrations": 0, "human_reviews": 0, "outcomes": rows,
        "interpretation": "Schema-guided coverage and final-gate sensitivity on an authored task; not real-model superiority."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = asyncio.run(evaluate())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8", newline="") as stream:
        stream.write(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key != "outcomes"}, indent=2))
    if result["passed"] != result["total"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
