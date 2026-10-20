"""Migration CI jobs must provision their own required sandbox aliases."""

from __future__ import annotations

import shlex
from pathlib import Path

import pytest
import yaml

from code_agent.upgrade_models import MigrationSpec

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    "job,additional_images",
    [
        ("metamorphic-migration-controls", {"agentforge-upgradepilot-sessions:local"}),
        ("tests", {"agentforge-upgradepilot-schema:local", "agentforge-upgradepilot-sessions:local"}),
        ("portable-patch-controls", set()),
    ],
)
def test_job_builds_all_required_migration_aliases_together(job, additional_images):
    fixture = ROOT / "tests/fixtures/upgradepilot/union/migration.json"
    recovery_image = MigrationSpec.model_validate_json(fixture.read_text(encoding="utf-8")).image
    required = {recovery_image, *additional_images}
    workflow = yaml.safe_load((ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8"))
    builds = []
    for step in workflow["jobs"][job]["steps"]:
        script = step.get("run", "").replace("\\\n", " ")
        for line in script.splitlines():
            if not line.strip().startswith("docker build "):
                continue
            command = shlex.split(line)
            if "-f" not in command or command[command.index("-f") + 1] != "docker/Dockerfile.upgradepilot":
                continue
            builds.append(
                {command[index + 1] for index, word in enumerate(command) if word in {"-t", "--tag"}}
            )
    assert any(required <= tags for tags in builds), (
        f"{job} must build {sorted(required)} together; observed migration builds: {builds}"
    )
