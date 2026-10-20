from pathlib import Path

import pytest

from code_agent.patch_inputs import PatchVerifySpec
from code_agent.patch_inputs import test_source_path as source_path
from code_agent.patch_verify import verify_patch


@pytest.mark.parametrize("value", ["tests/check.py::Check::test_edge", "checks.py::test_value", "tests"])
def test_named_test_selectors_are_data_not_arbitrary_pytest_flags(value):
    assert source_path(value) == value.split("::")[0]
    spec = PatchVerifySpec(task_id="selector", source_paths=["app.py"], test_paths=[value])
    assert spec.test_paths == [value]


@pytest.mark.parametrize(
    "value",
    [
        "../x.py::check",
        "x.py::",
        "x.py::-k",
        "x.py::test[a]",
        "--help",
        "x.py::A::B::C::D",
        "x.txt::Test",
        "x.py::A;bad",
        "x.py::a/b",
    ],
)
def test_unsafe_or_unsupported_selectors_are_rejected(value):
    with pytest.raises(ValueError):
        PatchVerifySpec(task_id="selector", source_paths=["app.py"], test_paths=[value])


def test_named_owner_tests_cannot_be_declared_as_editable_production():
    with pytest.raises(ValueError, match="test sources"):
        PatchVerifySpec(task_id="selector", source_paths=["checks.py"], test_paths=["checks.py::test_value"])


@pytest.mark.asyncio
async def test_valid_selector_reaches_execution_admission_without_requiring_literal_colons_on_disk():
    fixture = Path(__file__).parent / "fixtures/patch_verify"
    spec = PatchVerifySpec.model_validate_json((fixture / "fee-correct.json").read_bytes())
    spec = PatchVerifySpec.model_validate(
        {**spec.model_dump(), "test_paths": ["owner_checks.py::test_positive_fee"]}
    )
    called = []

    def unavailable(image):
        called.append(image)
        raise RuntimeError("controlled stop before any Docker execution")

    result = await verify_patch(
        fixture, (fixture / "fee-correct.diff").read_text(), spec, image_resolver=unavailable
    )
    assert called and result["status"] == "inconclusive" and not result["trials"]
