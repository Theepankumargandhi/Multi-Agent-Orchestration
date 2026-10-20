"""Frozen authored migration comparison and overfitting control; real Docker, no live AI."""
from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from code_agent.models import CodeAction
from code_agent.upgrade_contracts import canonical
from code_agent.upgrade_inputs import REDUCER_VERSION, CounterexamplePolicy, generate_campaign
from code_agent.upgrade_models import MigrationSpec, fingerprint
from code_agent.upgrade_runner import run_migration
from evals.upgradepilot_evaluation import FIXTURES, ROOT, ScriptedSpecialist, corpus_fingerprint

LOCK = ROOT / "evals/experiments/upgrade_counterexamples.lock.json"


def control_spec():
    spec = MigrationSpec.model_validate_json((FIXTURES / "union/migration.json").read_bytes())
    # Deliberately incomplete visible contract: ordinary integer input only.
    return MigrationSpec.model_validate({**spec.model_dump(), "contracts": [spec.contracts[0].model_dump()]})


class CounterexampleSpecialist(ScriptedSpecialist):
    """Knows the authored correction. Not a model-quality or independent-review result."""
    def __init__(self, role, *, overfit=False):
        super().__init__(role, "union")
        self.overfit = overfit
        self.messages = []

    async def ainvoke(self, messages, config=None):
        self.messages.extend(str(message.content) for message in messages)
        result = await super().ainvoke(messages, config)
        if isinstance(result, CodeAction) and result.kind == "edit":
            replacement = result.new_text.replace("    model = Request(**payload)",
                "    if isinstance(payload, dict) and isinstance(payload.get('value'), float):\n"
                "        payload = {**payload, 'value': int(payload['value'])}\n"
                "    model = Request(**payload)")
            if self.overfit:
                # An explicitly adversarial authored control, not agent access to final inputs.
                campaign = generate_campaign(control_spec(), CounterexamplePolicy())
                seen = [canonical(case.args[0]) for case in campaign.discovery if len(case.args) == 1
                    and not case.kwargs and isinstance(case.args[0], dict)]
                replacement = "import json\n\n" + replacement
                replacement = replacement.replace("    model = Request(**payload)",
                    "    model = Request(**payload)\n"
                    f"    if isinstance(payload, dict) and json.dumps(payload, sort_keys=True, separators=(',', ':'), ensure_ascii=False) not in {seen!r}:\n"
                    "        if isinstance(payload.get('value'), (str, float)):\n"
                    "            return {'value': 'deliberate-overfit', 'type': 'str'}")
            result = result.model_copy(update={"new_text": replacement})
        return result


async def evaluate():
    lock = json.loads(LOCK.read_bytes())
    spec, policy = control_spec(), CounterexamplePolicy()
    campaign = generate_campaign(spec, policy)
    if (lock["corpus_fingerprint"] != corpus_fingerprint() or lock["spec_fingerprint"] != fingerprint(spec.model_dump())
            or lock["campaign_commitment"] != campaign.commitment or lock["policy"] != policy.model_dump()
            or lock["reducer_version"] != REDUCER_VERSION):
        raise ValueError("counterexample study changed; explicitly review its frozen lock")
    arms = (("codemod_only", False, False), ("existing_repair", False, False),
        ("counterexample_repair", True, False), ("overfit_negative_control", True, True))
    outcomes = []
    for name, feedback, overfit in arms:
        models = None if name == "codemod_only" else tuple(CounterexampleSpecialist(role, overfit=overfit)
            for role in ("analysis", "actions", "review"))
        report = await run_migration(FIXTURES / "union", spec, scripted_models=models,
            counterexample_policy=policy, discovery_feedback=feedback)
        exposed = bool(models and any(case.case_id in message for case in campaign.final
            for model in models for message in model.messages))
        expected_ready = name == "counterexample_repair"
        ready = report["status"] == "ready_for_owner_review"
        final = report.get("final_evaluation", {})
        passed = (ready == expected_ready and bool(final) and bool(final.get("passed")) == expected_ready
            and report["source_unchanged"] and not report["live_provider_calls"] and not exposed)
        if overfit:
            passed = passed and report.get("discovery_recheck", {}).get("passed", False)
        outcomes.append({"arm": name, "passed": bool(passed), "final_ids_exposed_in_prompts": exposed, "report": report})
        print(json.dumps({"arm": name, "control_passed": bool(passed), "status": report["status"],
            "phase": report["phase"], "model_calls": report["model_calls"]}), flush=True)
    return {"evidence_kind": "one_authored_family_real_docker_scripted_repairs", "campaign_commitment": campaign.commitment,
        "total": len(outcomes), "passed": sum(row["passed"] for row in outcomes), "outcomes": outcomes,
        "live_provider_calls": 0, "real_repository_migrations": 0, "independent_human_reviews": 0,
        "interpretation": "Pipeline sensitivity and overfitting controls; not measured AI superiority or production impact."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
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
