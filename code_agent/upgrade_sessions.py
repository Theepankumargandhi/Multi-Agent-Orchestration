"""Owner-authored stateful contracts; JSON calls only, never application imports on host."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import Field, model_validator

from code_agent.upgrade_models import ContractCase, StrictModel, fingerprint
from code_agent.upgrade_plan import source_files

SESSION_PROTOCOL = "upgrade-session-v1"
SESSION_CONFIG = "upgradepilot-session.json"
SESSION_PROBE = "/opt/upgradepilot/session_probe.py"
SESSION_IMAGE = "agentforge-upgradepilot-sessions:local"
SESSION_REDUCER = "ordered-step-deletion-v1"


class SessionStep(StrictModel):
    method: str = Field(pattern=r"^[A-Za-z][A-Za-z0-9_]{0,79}$")
    args: list[Any] = Field(default_factory=list, max_length=20)
    kwargs: dict[str, Any] = Field(default_factory=dict, max_length=20)


class SessionScenario(StrictModel):
    scenario_id: str = Field(pattern=r"^[A-Za-z0-9_.-]{1,80}$")
    factory: str = Field(pattern=r"^[A-Za-z][A-Za-z0-9_]*(?:\.[A-Za-z][A-Za-z0-9_]*)*:[A-Za-z][A-Za-z0-9_]*$", max_length=200)
    args: list[Any] = Field(default_factory=list, max_length=20)
    kwargs: dict[str, Any] = Field(default_factory=dict, max_length=20)
    steps: list[SessionStep] = Field(min_length=1, max_length=12)

    @model_validator(mode="after")
    def bounded(self):
        # Reuse the public-entrypoint and finite JSON contract without executing it.
        ContractCase(case_id=self.scenario_id, entrypoint=self.factory, args=self.args, kwargs=self.kwargs)
        if len(self.model_dump_json().encode()) > 12000:
            raise ValueError("session scenario exceeds its JSON byte budget")
        json.dumps(self.model_dump(), allow_nan=False)
        return self


def session_key(scenario):
    return fingerprint(scenario.model_dump(exclude={"scenario_id"}))


class SessionPolicy(StrictModel):
    discovery: list[SessionScenario] = Field(min_length=1, max_length=4)
    final: list[SessionScenario] = Field(min_length=1, max_length=4)
    max_shrink_evaluations: int = Field(default=8, ge=0, le=12)
    max_counterexamples: int = Field(default=1, ge=1, le=2)

    @model_validator(mode="after")
    def independent_sets(self):
        rows = [*self.discovery, *self.final]
        if len({row.scenario_id for row in rows}) != len(rows) or len({session_key(row) for row in rows}) != len(rows):
            raise ValueError("session IDs and complete input sequences must be unique across both sets")
        if len(self.model_dump_json().encode()) > 40000:
            raise ValueError("session policy exceeds its JSON byte budget")
        return self


def validate_session_scope(repository: Path, spec, scenarios):
    allowed = {path.relative_to(repository.resolve()).as_posix() for path in source_files(repository, spec)}
    for row in scenarios:
        module = row.factory.split(":", 1)[0].replace(".", "/")
        if module + ".py" not in allowed and module + "/__init__.py" not in allowed:
            raise ValueError("session factories must be inside the owner source scope")
    # Never copy a final policy from a fixture into an agent workspace.
    if (repository / SESSION_CONFIG).exists():
        raise ValueError("reserved session controller path already exists")


def validate_sessions(repository: Path, spec, policy):
    validate_session_scope(repository, spec, (*policy.discovery, *policy.final))


def session_commitment(policy):
    return fingerprint({"protocol": SESSION_PROTOCOL, "reducer": SESSION_REDUCER, "policy": policy.model_dump()})


def verification_reserve(policy):
    """Seconds reserved for recheck + final: two versions, two fresh processes each."""
    return 4 * sum(2 + len(row.steps) for row in (*policy.discovery, *policy.final))
