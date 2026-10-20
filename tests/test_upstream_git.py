"""Real Git object admission tests using tiny local, authored repositories only."""

import hashlib
import os
import subprocess

import pytest

from code_agent.upstream_inputs import GitSource

URL = "https://github.com/example/unit"


@pytest.fixture
def objects(tmp_path):
    path = tmp_path / (hashlib.sha256(URL.encode()).hexdigest()[:24] + ".git")
    subprocess.run(["git", "-c", "init.templateDir=", "init", "--bare", "--quiet", str(path)], check=True)
    env = {
        **os.environ,
        "GIT_AUTHOR_NAME": "Unit Control",
        "GIT_COMMITTER_NAME": "Unit Control",
        "GIT_AUTHOR_EMAIL": "unit@example.invalid",
        "GIT_COMMITTER_EMAIL": "unit@example.invalid",
    }

    def git(*args, raw=None):
        return subprocess.check_output(
            ["git", "--git-dir=" + str(path), *args], input=raw, env=env, timeout=10
        )

    raw = b"raise RuntimeError('never import submitted source on the host')\n"
    license_bytes = b"Authored unit license\r\n"
    oid = git("hash-object", "-w", "--stdin", raw=raw).strip()
    license_oid = git("hash-object", "-w", "--stdin", raw=license_bytes).strip()
    tree = (
        git(
            "mktree",
            raw=b"100644 blob "
            + oid
            + b"\tapp.py\n100644 blob "
            + license_oid
            + b"\tLICENSE\n120000 blob "
            + oid
            + b"\tlink.py\n",
        )
        .decode()
        .strip()
    )
    commit = git("commit-tree", tree, "-m", "Authored unit data only").decode().strip()
    git("config", "remote.origin.url", URL)
    source = GitSource(tmp_path, URL)
    return source, commit, raw, license_bytes, git


def test_git_source_verifies_objects_without_importing_them_and_preserves_license_crlf(objects):
    source, commit, raw, license_bytes, _ = objects
    assert source.commit(commit).startswith(b"tree ")
    assert source.blob(commit, "app.py", hashlib.sha256(raw).hexdigest()) == raw
    assert source.blob(commit, "LICENSE", hashlib.sha256(license_bytes).hexdigest()) == license_bytes


@pytest.mark.parametrize("failure", ["hash", "traversal", "symlink", "wrong_kind", "short_commit"])
def test_bad_git_admission_never_becomes_a_snapshot(objects, failure):
    source, commit, raw, _, _ = objects
    with pytest.raises(ValueError):
        if failure == "hash":
            source.blob(commit, "app.py", "f" * 64)
        elif failure == "traversal":
            source.blob(commit, "../app.py", hashlib.sha256(raw).hexdigest())
        elif failure == "symlink":
            source.blob(commit, "link.py", hashlib.sha256(raw).hexdigest())
        elif failure == "wrong_kind":
            source.object(commit, "blob", 1000)
        else:
            source.commit(commit[:7])


def test_object_payload_cannot_disagree_with_git_identity(objects, monkeypatch):
    source, commit, raw, _, _ = objects
    original = source.read

    def corrupt(*args, **kwargs):
        value = original(*args, **kwargs)
        return b"!" + value[1:] if args[:2] == ("cat-file", "blob") else value

    monkeypatch.setattr(source, "read", corrupt)
    with pytest.raises(ValueError, match="identity"):
        source.blob(commit, "app.py", hashlib.sha256(raw).hexdigest())


def test_cache_origin_is_not_trusted_just_because_the_directory_exists(objects, tmp_path):
    _, _, _, _, git = objects
    git("config", "remote.origin.url", "https://github.com/example/different")
    with pytest.raises(ValueError, match="origin"):
        GitSource(tmp_path, URL)
