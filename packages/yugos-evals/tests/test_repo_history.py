"""SWE repositories keep only the base commit's history before the agent starts."""

import os
import subprocess
from types import SimpleNamespace

import pytest

from yugos_evals import repo_history


def _git(repo, *args):
    return subprocess.run(
        ["git", "-C", str(repo), *args], check=True, capture_output=True, text=True
    ).stdout.strip()


def _commit(repo, name):
    (repo / name).write_text(name)
    _git(repo, "add", name)
    _git(repo, "commit", "-qm", name)
    return _git(repo, "rev-parse", "HEAD")


def test_scrub_leaves_no_commit_after_the_base(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    for key, value in {"GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t",
                       "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t"}.items():
        monkeypatch.setenv(key, value)  # fmt: skip
    upstream, repo = tmp_path / "upstream", tmp_path / "testbed"
    upstream.mkdir()
    _git(upstream, "init", "-q", "-b", "main")
    _commit(upstream, "old")
    _git(upstream, "tag", "v1")
    base = _commit(upstream, "base")
    fix = _commit(upstream, "fix")
    _git(upstream, "tag", "v2")
    subprocess.run(["git", "clone", "-q", str(upstream), str(repo)], check=True)
    _git(repo, "checkout", "-q", base)
    (repo / "base").write_text("agent work in progress")
    _git(repo, "stash")

    subprocess.run(
        ["bash", "-c", repo_history.HISTORY_SCRUB, "scrub", base, str(repo)], check=True
    )

    assert _git(repo, "rev-parse", "HEAD") == base
    assert _git(repo, "remote") == ""
    assert _git(repo, "tag").split() == ["v1"]
    assert _git(repo, "rev-list", "--all", "--not", base) == ""
    assert _git(repo, "for-each-ref", "--format=%(refname)").split() == [
        "refs/heads/main", "refs/tags/v1",
    ]  # fmt: skip
    missing = subprocess.run(["git", "-C", str(repo), "cat-file", "-e", fix])
    assert missing.returncode != 0
    assert os.listdir(repo) and (repo / "base").read_text() == "base"


@pytest.mark.asyncio
async def test_setup_passes_only_a_hex_base_commit(monkeypatch):
    commands = []

    class Sandbox:
        async def exec(self, command, **kwargs):
            commands.append(command)
            return SimpleNamespace(returncode=0)

    monkeypatch.setattr(repo_history, "sandbox", lambda: Sandbox())
    state = SimpleNamespace(metadata={"base_commit": "abc123"})
    await repo_history.base_history_only()(state, None)
    assert commands == [
        ["bash", "-c", repo_history.HISTORY_SCRUB, "scrub", "abc123", "/testbed"]
    ]
    state.metadata["base_commit"] = "abc123; curl x"
    with pytest.raises(ValueError, match="invalid SWE base commit"):
        await repo_history.base_history_only()(state, None)
