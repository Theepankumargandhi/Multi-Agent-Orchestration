"""Frozen authored metamorphic-law controls in actual Docker; scripted repairs, never live AI."""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
from pathlib import Path

from code_agent.models import CodeAction
from code_agent.upgrade_models import MigrationSpec, fingerprint
from code_agent.upgrade_relations import RelationPolicy, generate_relations
from code_agent.upgrade_runner import run_migration
from evals.upgradepilot_evaluation import ROOT, ScriptedSpecialist

FIXTURE = ROOT / "tests/fixtures/upgrade_relations"
POLICY = ROOT / "evals/experiments/upgrade_relations.policy.json"
LOCK = ROOT / "evals/experiments/upgrade_relations.lock.json"


def recipe():
    return MigrationSpec.model_validate_json((FIXTURE / "migration.json").read_bytes())


def policy():
    return RelationPolicy.model_validate_json(POLICY.read_bytes())


def corpus_fingerprint():
    return fingerprint({path.relative_to(FIXTURE).as_posix(): hashlib.sha256(path.read_text(encoding="utf-8").encode()).hexdigest()
        for path in sorted(FIXTURE.rglob("*")) if path.is_file() and "__pycache__" not in path.parts})


class LawSpecialist(ScriptedSpecialist):
    """Known source correction, or deliberate discovery-only overfit; not model-quality evidence."""
    def __init__(self, role, overfit=False):
        super().__init__(role, "metamorphic")
        self.overfit, self.messages = overfit, []

    async def ainvoke(self, messages, config=None):
        self.messages.extend(str(row.content) for row in messages)
        result = await super().ainvoke(messages, config)
        if isinstance(result, CodeAction) and result.kind == "edit":
            text = result.new_text
            if self.overfit:
                text = text.replace("        request = Deposit(event=event, amount=amount)",
                    "        if event == '12':\n            event = 12\n        request = Deposit(event=event, amount=amount)")
            else:
                text = text.replace("from pydantic import BaseModel", "from pydantic import BaseModel, Field")
                text = text.replace("event: Union[int, str]", "event: Union[int, str] = Field(union_mode='left_to_right')")
            result = result.model_copy(update={"new_text": text})
        return result


def broken_policy(profile):
    data = profile.model_dump()
    for seed in data["seeds"]:
        seed["scenario"]["factory"] = "app.models:create_broken_wallet"
    return RelationPolicy.model_validate(data)


async def evaluate():
    locked, spec, profile = json.loads(LOCK.read_bytes()), recipe(), policy()
    campaign = generate_relations(spec, profile)
    if locked["corpus_fingerprint"] != corpus_fingerprint() or locked["campaign_commitment"] != campaign.commitment:
        raise ValueError("metamorphic controls changed; explicitly review the lock")
    rows = []
    for name in ("codemod_only", "law_guided_repair", "discovery_overfit", "invalid_source_oracle", "revoked_campaign"):
        models = tuple(LawSpecialist(role, name == "discovery_overfit") for role in ("analysis", "actions", "review")) if name != "codemod_only" else None
        selected = broken_policy(profile) if name == "invalid_source_oracle" else profile
        report = await run_migration(FIXTURE, spec, relation_policy=selected, scripted_models=models,
            expected_relation_commitment="0" * 64 if name == "revoked_campaign" else None)
        source = report.get("relation_source_calibration", {})
        discovery, final = report.get("relation_discovery", {}), report.get("final_relation_evaluation", {})
        leaked = any(case.case_id in message or scenario.scenario_id in message or '"37"' in message
            for case in campaign.final for scenario in case.scenarios
            for specialist in models or () for message in specialist.messages)
        if name == "codemod_only":
            expected = (source.get("passed") and report["baseline"]["pre_relation_gates_passed"]
                and discovery.get("violations") == 1 and report["status"] == "held")
        elif name == "invalid_source_oracle":
            expected = (not source.get("passed", True) and report["reason"] == "hold_at:metamorphic_source_calibration"
                and not report["model_calls"] and not discovery and not final)
        elif name == "revoked_campaign":
            expected = report["reason"] == "hold_at:preflight" and not report["model_calls"] and not source
        else:
            expected = (source.get("passed") and discovery.get("violations") == 1 and report.get("relation_recheck", {}).get("passed")
                and bool(final.get("passed")) == (name == "law_guided_repair")
                and report["status"] == ("ready_for_owner_review" if name == "law_guided_repair" else "held"))
        rows.append({"arm": name, "passed": bool(expected and not leaked and report["source_unchanged"]
            and not report["live_provider_calls"]), "final_case_ids_in_model_messages": leaked, "report": report})
        print(json.dumps({"arm": name, "passed": rows[-1]["passed"], "status": report["status"], "phase": report["phase"]}), flush=True)
    return {"evidence_kind": "authored_frozen_owner_laws_real_docker_scripted_repairs", "total": len(rows),
        "passed": sum(row["passed"] for row in rows), "outcomes": rows, "live_provider_calls": 0,
        "corpus_fingerprint": locked["corpus_fingerprint"], "campaign_commitment": campaign.commitment,
        "real_repository_quality_measured": False,
        "interpretation": "Generated-law sensitivity, source-oracle calibration and final isolation; not live-model repair quality."}


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
