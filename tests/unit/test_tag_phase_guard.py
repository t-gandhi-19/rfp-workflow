"""`make tag-phase` refuses to tag anything it should not.

v0.2 was once created against a `main` that had not actually received the phase
branch, so the tag pointed at the previous phase's code. It was caught and
corrected, but being caught was luck. These tests drive the guard against real
throwaway repositories, because a guard asserted only in a unit stub is a guard
nobody has watched refuse anything.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "tag_phase.sh"

pytestmark = pytest.mark.skipif(
    shutil.which("git") is None or shutil.which("bash") is None,
    reason="needs git and bash",
)


def run_git(cwd: Path, *args: str) -> None:
    subprocess.run(  # noqa: S603
        ["git", *args],  # noqa: S607
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    )


def tag_phase(cwd: Path, tag: str | None) -> subprocess.CompletedProcess[str]:
    env = {"PATH": "/usr/bin:/bin:/usr/local/bin", "HOME": str(cwd)}
    if tag is not None:
        env["TAG"] = tag
    return subprocess.run(  # noqa: S603
        ["bash", str(SCRIPT)],  # noqa: S607
        cwd=cwd,
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    """A repo with an `origin` remote, on main, clean and in sync."""
    origin = tmp_path / "origin.git"
    origin.mkdir()
    run_git(origin, "init", "--bare", "--initial-branch=main")

    work = tmp_path / "work"
    work.mkdir()
    run_git(work, "init", "--initial-branch=main")
    run_git(work, "config", "user.email", "t@example.com")
    run_git(work, "config", "user.name", "Test")
    (work / "file.txt").write_text("one\n", encoding="utf-8")
    run_git(work, "add", ".")
    run_git(work, "commit", "-m", "initial")
    run_git(work, "remote", "add", "origin", str(origin))
    run_git(work, "push", "-u", "origin", "main")
    return work


class TestItRefuses:
    def test_without_a_tag(self, repo: Path) -> None:
        result = tag_phase(repo, None)
        assert result.returncode != 0
        assert "TAG is not set" in result.stderr

    @pytest.mark.parametrize("tag", ["0.3", "release-3", "v", "phase3"])
    def test_a_tag_that_is_not_a_phase_tag(self, repo: Path, tag: str) -> None:
        result = tag_phase(repo, tag)
        assert result.returncode != 0
        assert "does not look like a phase tag" in result.stderr

    def test_off_main(self, repo: Path) -> None:
        """A tag on a feature branch names the wrong tree."""
        run_git(repo, "checkout", "-b", "phase-9-something")
        result = tag_phase(repo, "v0.9")
        assert result.returncode != 0
        assert "not main" in result.stderr

    def test_with_a_dirty_worktree(self, repo: Path) -> None:
        (repo / "file.txt").write_text("uncommitted\n", encoding="utf-8")
        result = tag_phase(repo, "v0.9")
        assert result.returncode != 0
        assert "dirty" in result.stderr

    def test_with_an_untracked_file(self, repo: Path) -> None:
        (repo / "stray.txt").write_text("x\n", encoding="utf-8")
        result = tag_phase(repo, "v0.9")
        assert result.returncode != 0
        assert "dirty" in result.stderr

    def test_when_main_is_ahead_of_origin(self, repo: Path) -> None:
        (repo / "file.txt").write_text("two\n", encoding="utf-8")
        run_git(repo, "add", ".")
        run_git(repo, "commit", "-m", "unpushed")
        result = tag_phase(repo, "v0.9")
        assert result.returncode != 0
        assert "differ" in result.stderr

    def test_when_main_is_behind_origin(self, repo: Path) -> None:
        """The exact shape of the v0.2 mistake: local main lacks the merge."""
        run_git(repo, "checkout", "-b", "feature")
        (repo / "file.txt").write_text("three\n", encoding="utf-8")
        run_git(repo, "add", ".")
        run_git(repo, "commit", "-m", "feature work")
        run_git(repo, "push", "origin", "feature:main")
        run_git(repo, "checkout", "main")

        result = tag_phase(repo, "v0.9")
        assert result.returncode != 0
        assert "differ" in result.stderr
        assert "has\n    not yet received the phase merge" in result.stderr

    def test_when_the_tag_already_exists(self, repo: Path) -> None:
        assert tag_phase(repo, "v0.9").returncode == 0
        second = tag_phase(repo, "v0.9")
        assert second.returncode != 0
        assert "already exists" in second.stderr


class TestItTags:
    def test_a_clean_synced_main_is_tagged_and_pushed(self, repo: Path) -> None:
        result = tag_phase(repo, "v0.9")
        assert result.returncode == 0, result.stderr
        assert "pushed v0.9" in result.stdout

        tags = subprocess.run(
            ["git", "tag", "-l"],  # noqa: S607
            cwd=repo,
            capture_output=True,
            text=True,
            check=True,
        )
        assert "v0.9" in tags.stdout

        remote = subprocess.run(
            ["git", "ls-remote", "--tags", "origin"],  # noqa: S607
            cwd=repo,
            capture_output=True,
            text=True,
            check=True,
        )
        assert "refs/tags/v0.9" in remote.stdout

    def test_it_reports_what_it_tagged(self, repo: Path) -> None:
        """So a mis-tag is visible in the terminal, not only in hindsight."""
        result = tag_phase(repo, "v0.9")
        assert "tagging v0.9 at" in result.stdout
        assert "initial" in result.stdout
