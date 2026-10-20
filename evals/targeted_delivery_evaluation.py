"""Scripted specialist workflow in real Docker; not live model repair quality."""
from __future__ import annotations

import argparse
import asyncio
import json
import tempfile
from pathlib import Path

from code_agent.delivery import result_delivery_valid
from code_agent.models import AnalysisPlan, CodeAction, CodeTask, ReviewVerdict, SandboxPolicy
from code_agent.sandbox import DockerSandbox
from code_agent.verified_pr import VerifiedPRAgent


class ControlledModel:
    def __init__(self, role):
        self.role, self.calls = role, 0

    async def ainvoke(self, messages, config=None):
        self.calls += 1
        if self.role == "analyst":
            return AnalysisPlan(summary="Authored control: change VALUE to two and add a focused assertion.")
        if self.role == "reviewer":
            return ReviewVerdict(approved=True, summary="Scripted reviewer control; execution gates decide eligibility.")
        if self.calls == 1:
            return CodeAction(kind="read_lines", path="app.py", end_line=1)
        if self.calls == 2:
            read = json.loads(json.loads(messages[1].content)["recent_observations"][0]["output"])
            return CodeAction(kind="edit", path="app.py", expected_sha256=read["file_sha256"], old_text="VALUE = 1", new_text="VALUE = 2")
        if self.calls == 4:
            return CodeAction(kind="write", path="tests/test_regression.py", content="from app import VALUE\ndef test_regression():\n    assert VALUE != 1\n")
        return CodeAction(kind="test")


async def evaluate(image):
    if not DockerSandbox.available() or not DockerSandbox.image_available(image):
        return {"status": "unavailable", "reason": "docker_or_image_unavailable", "live_provider_calls": 0}
    with tempfile.TemporaryDirectory(prefix="agentforge-delivery-control-") as temporary:
        original = Path(temporary) / "original"
        original.mkdir()
        (original / "app.py").write_bytes(b"VALUE = 1\n")
        (original / "test_owner.py").write_bytes(b"from app import VALUE\ndef test_owner():\n    assert VALUE == 2\n")
        policy = SandboxPolicy(image=image, allowed_images={image}, targeted_edits_only=True,
            delivery_verification_enabled=True, require_specialist_completion=True, max_repair_rounds=0,
            max_iterations=6, command_timeout_seconds=30)
        task = CodeTask(repository="authored", issue="Fix VALUE and add a focused regression assertion.", policy=policy, model="scripted-control")
        sandbox = DockerSandbox(original, policy)
        try:
            await asyncio.to_thread(sandbox.start)
            result = await VerifiedPRAgent(ControlledModel("analyst"), ControlledModel("actions"), ControlledModel("reviewer")).solve(task, sandbox)
            proof = next(gate.evidence for gate in result.verification.rounds[-1].quality_gates if gate.name == "fresh_patch_delivery")
            passed = result.status == "completed" and result_delivery_valid(result, task.test_command)
            passed = passed and (original / "app.py").read_bytes() == b"VALUE = 1\n"
            return {"status": "passed" if passed else "failed", "evidence_kind": "scripted_roles_real_docker",
                "live_provider_calls": 0, "real_issue_quality_measured": False, "stage_outcomes": result.verification.stage_outcomes,
                "model_calls_scripted": result.model_calls, "usage_accounting_complete": result.usage_accounting_complete,
                "delivery": proof, "source_unchanged": result.sandbox["original_repository_unchanged"]}
        finally:
            await asyncio.to_thread(sandbox.close)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", default="agentforge-code-sandbox:local")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = asyncio.run(evaluate(args.image))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    if result["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
