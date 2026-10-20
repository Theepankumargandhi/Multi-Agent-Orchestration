"""Actual process/session isolation, mutation safety and harness negative controls."""
import pytest

from code_agent.models import SandboxPolicy
from code_agent.sandbox import DockerSandbox
from code_agent.upgrade_models import SOURCE_VERSION, TARGET_VERSION, V1_PYTHON, V2_PYTHON
from code_agent.upgrade_runner import resolve_image
from code_agent.upgrade_session_runtime import SessionVerifier, compare_sessions
from code_agent.upgrade_sessions import SESSION_IMAGE, SessionScenario
from code_agent.verification import repository_fingerprint
from evals.upgrade_session_evaluation import FIXTURE, recipe

CONTROLS = '''
import pydantic

constructions = 0

class Session:
    def __init__(self):
        global constructions
        constructions += 1
        self.values = []

    def snapshot(self):
        return self.values

    def count(self):
        return constructions

    def add(self):
        self.values.append("added")

    def reject(self):
        if pydantic.__version__.startswith("2."):
            self.values.append("regression-after-error")
        raise ValueError("same exception class, different state")

    def invalid_json(self):
        return {1, 2}

    def oversized(self):
        return "x" * 9000

    def timeout(self):
        while True:
            pass

    def mutate(self):
        from pathlib import Path
        Path("/workspace/app/touched.txt").write_text("not allowed")

def create():
    return Session()

def broken():
    raise ValueError("factory failure is not an application-step result")
'''


@pytest.fixture
def sandbox():
    if not DockerSandbox.available() or not DockerSandbox.image_available(SESSION_IMAGE):
        pytest.skip("Dedicated session image required; never execute application imports on host")
    image = resolve_image(SESSION_IMAGE)
    policy = SandboxPolicy(image=image, allowed_images={image}, max_output_chars=100000,
        allowed_test_executables={V1_PYTHON, V2_PYTHON})
    before = repository_fingerprint(FIXTURE, 5000, 50000000)
    with DockerSandbox(FIXTURE, policy) as active:
        active.workspace.write_file("app/session_controls.py", CONTROLS)
        yield active
    assert repository_fingerprint(FIXTURE, 5000, 50000000) == before


def verifier():
    instance = object.__new__(SessionVerifier)
    instance.commands, instance.spec = 0, recipe()
    return instance


def scenario(*methods, factory="create"):
    return SessionScenario(scenario_id="runtime-control", factory="app.session_controls:" + factory,
        steps=[{"method": method} for method in methods])


@pytest.mark.asyncio
async def test_real_fresh_processes_preserve_session_state_and_snapshot_aliases(sandbox):
    probe = verifier()
    contract = scenario("count", "snapshot", "add", "snapshot")
    old = await probe.observe(sandbox, contract, SOURCE_VERSION)
    new = await probe.observe(sandbox, contract, TARGET_VERSION)
    assert old[0]["result"]["value"] == new[0]["result"]["value"] == 1
    assert old[1]["result"]["value"] == [] and old[3]["result"]["value"] == ["added"]
    assert compare_sessions(contract, old, new)["passed"] and probe.commands == 4


@pytest.mark.asyncio
async def test_same_exception_does_not_hide_post_failure_state_regression(sandbox):
    probe = verifier()
    contract = scenario("add", "reject", "snapshot")
    old = await probe.observe(sandbox, contract, SOURCE_VERSION)
    new = await probe.observe(sandbox, contract, TARGET_VERSION)
    assert old[1]["result"] == new[1]["result"] == {"kind": "error", "type": "ValueError"}
    diagnosis = compare_sessions(contract, old, new)
    assert not diagnosis["passed"] and diagnosis["first_divergence"]["index"] == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("method", ["missing", "invalid_json", "oversized", "timeout", "mutate"])
async def test_actual_unavailable_session_evidence_holds(sandbox, method):
    with pytest.raises(ValueError):
        await verifier().observe(sandbox, scenario(method), SOURCE_VERSION)


@pytest.mark.asyncio
async def test_failed_factory_cannot_be_misreported_as_valid_application_error(sandbox):
    with pytest.raises(ValueError):
        await verifier().observe(sandbox, scenario("snapshot", factory="broken"), SOURCE_VERSION)
