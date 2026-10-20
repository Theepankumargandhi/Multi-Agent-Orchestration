"""Keep application and service encryption dependencies on the reviewed release line."""

import tomllib
from pathlib import Path

import pytest
from packaging.requirements import Requirement

ROOT = Path(__file__).resolve().parents[1]


def cryptography_requirements():
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    sources = {
        "project": project["project"]["dependencies"],
        "service": (ROOT / "requirements-service.txt").read_text(encoding="utf-8").splitlines(),
    }
    requirements = {}
    for name, lines in sources.items():
        matches = [Requirement(line) for line in lines if line.startswith("cryptography")]
        assert len(matches) == 1, f"{name} must declare cryptography exactly once"
        requirements[name] = matches[0].specifier
    return requirements


def test_encryption_dependency_constraints_are_consistent():
    requirements = cryptography_requirements()
    assert requirements["project"] == requirements["service"]


@pytest.mark.parametrize("version", ["46.0.7", "47.0.0", "48.0.1", "49.0.0", "50.0.0", "50.0.1"])
def test_encryption_dependency_rejects_releases_below_reviewed_floor(version):
    assert all(not specifier.contains(version) for specifier in cryptography_requirements().values())


@pytest.mark.parametrize("version", ["50.0.2", "50.0.3"])
def test_encryption_dependency_accepts_reviewed_release_line(version):
    assert all(specifier.contains(version) for specifier in cryptography_requirements().values())


def test_encryption_dependency_holds_the_next_major_release_for_review():
    assert all(not specifier.contains("51.0.0") for specifier in cryptography_requirements().values())
