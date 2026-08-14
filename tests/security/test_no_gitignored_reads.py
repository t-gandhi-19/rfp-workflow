"""No test may depend on a file that is not in the repository (amendment R).

WHAT WENT WRONG. `tests/unit/test_scoring.py` read `fixtures/calibration.json`
at module scope. That file is gitignored on purpose (amendment J): it is bound
to the embedding model tag and corpus hash it was measured against, so a
committed copy would be loaded on a machine whose model may differ. It exists
only on a host that has run `make calibrate`.

The consequence was not one skipped test. Collection of the module raised
`FileNotFoundError`, so pytest aborted the run: on a clean clone `pytest
tests/unit` exited 2 with **zero** tests executed, and tests/security never ran
at all because the CI job had already failed. The local suite reported 616
passed at the same commit, honestly, because the developing host had the file.

THE CLASS, which is what this guard is for and why it lives beside the SDK and
SQL walks rather than next to the test it fixes. It is the same shape as the
`make preflight` wiring fault one layer down:

    green on the host that built it, broken on a clean checkout,
    invisible to the habit that verifies the work.

A test suite is a claim about the repository. If a test needs something the
repository does not contain, the claim is about the developer's machine.

THE RULE, stated precisely, because a blunter version is wrong. A test module
may not name a gitignored path **relative to the repository**. Writing a file
that happens to share a name into a temporary directory is a different act and a
perfectly good one — `tmp_path / "calibration.json"` is how the artifact's
round-trip and fail-closed loading are tested, and those tests depend on nothing
outside the repo.

So the check is: every string literal, matched against the concrete file paths
in `.gitignore` (derived from that file, so a newly ignored path is covered the
day it is added) and against their basenames — because paths are built a segment
at a time and only the last segment is a literal — EXCEPT where the enclosing
path expression is rooted at a pytest temporary-directory fixture.

The first draft of this guard omitted that exception and flagged six correct
tests in `test_calibration.py`. Worth recording: a guard that cries wolf gets
weakened, and the weakening is what lets the real thing through next time.

WHAT IT DOES NOT CATCH, stated plainly rather than implied away: a path assembled
from an f-string or handed around in a variable. Closing that would need real
dataflow analysis. This catches the literal form, which is the form that occurred
and the form almost anyone would write.

Committed fixtures are unaffected — `qa_pairs.json`, `answer_key_manual.json`
and the rest are in the repository, so reading them is a claim about the
repository and is exactly what tests should do.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
GITIGNORE = REPO_ROOT / ".gitignore"
TEST_ROOTS = (REPO_ROOT / "tests" / "unit", REPO_ROOT / "tests" / "security")

#: Glob metacharacters. An entry carrying one names a pattern rather than a
#: file, and this guard only reasons about concrete paths.
_GLOB = set("*?[]!")


def gitignored_files() -> set[str]:
    """Concrete FILE paths named in `.gitignore`.

    Directory entries (trailing `/`) and patterns are skipped: a test naming a
    directory is not the failure this guards, and a pattern would need matching
    rather than comparison. Negations are skipped because they un-ignore.
    """
    entries: set[str] = set()
    for raw in GITIGNORE.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or line.startswith("!"):
            continue
        if line.endswith("/"):
            continue
        if _GLOB & set(line):
            continue
        entries.add(line.lstrip("/"))
    return entries


def forbidden_literals() -> set[str]:
    """The ignored paths, plus their basenames.

    The basename matters more than the full path in practice: nobody writes
    `open("fixtures/calibration.json")`, they write
    `FIXTURES / "calibration.json"`, and only the last segment is a literal.
    """
    ignored = gitignored_files()
    return ignored | {path.rsplit("/", 1)[-1] for path in ignored}


#: pytest's temporary-directory fixtures. A path rooted at one of these is
#: scratch space that exists for the duration of the test, so naming a
#: gitignored basename under it depends on nothing outside the repository.
TMP_ROOTS = {"tmp_path", "tmp_path_factory", "tmpdir", "tmpdir_factory"}

#: This module necessarily spells out the paths it forbids, in its own prose and
#: in its own assertions. Excluded from its own scan for that reason, and named
#: explicitly rather than filtered by a rule that might exclude something else.
SELF = Path(__file__).name


def discover_test_modules() -> list[Path]:
    """Every test module in scope. Not named `test_*` — it is not a test."""
    return sorted(
        path for root in TEST_ROOTS for path in root.rglob("test_*.py") if path.name != SELF
    )


def _parents(tree: ast.AST) -> dict[ast.AST, ast.AST]:
    links: dict[ast.AST, ast.AST] = {}
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            links[child] = node
    return links


def _is_under_a_temp_root(node: ast.AST, parents: dict[ast.AST, ast.AST]) -> bool:
    """Is this literal part of a `/` path chain rooted at a pytest tmp fixture?

    Climbs out of the `Path / "a" / "b"` chain to its outermost `/` expression,
    then asks whether any name in it is a temporary-directory fixture. That is
    enough to tell `tmp_path / "calibration.json"` (fine) from
    `FIXTURES / "calibration.json"` (the fault).
    """
    current = node
    while True:
        parent = parents.get(current)
        if not isinstance(parent, ast.BinOp) or not isinstance(parent.op, ast.Div):
            break
        current = parent
    if current is node:
        return False  # not part of a path chain at all
    return any(isinstance(inner, ast.Name) and inner.id in TMP_ROOTS for inner in ast.walk(current))


def repo_path_literals(path: Path) -> list[tuple[int, str]]:
    """String constants that name a path relative to the repository.

    An AST walk rather than a text search, so a literal buried in a call is
    still seen, and a mention inside a `#` comment is correctly ignored —
    comments are not constants, and prose about `.env` is not a read of it.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    parents = _parents(tree)
    found: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Constant) and isinstance(node.value, str)):
            continue
        if _is_under_a_temp_root(node, parents):
            continue
        found.append((node.lineno, node.value))
    return found


class TestTheGuardItself:
    """A guard that matches nothing passes silently. These pin its inputs."""

    def test_the_gitignored_set_is_derived_and_not_empty(self) -> None:
        ignored = gitignored_files()
        assert ignored, ".gitignore parsed to nothing; the guard would be vacuous"
        assert "fixtures/calibration.json" in ignored
        assert ".env" in ignored

    def test_directories_and_patterns_are_excluded(self) -> None:
        ignored = gitignored_files()
        assert "out/" not in ignored and "out" not in ignored
        assert not any(_GLOB & set(entry) for entry in ignored)

    def test_basenames_are_forbidden_too(self) -> None:
        """The form the real fault took."""
        assert "calibration.json" in forbidden_literals()

    def test_there_are_test_modules_to_scan(self) -> None:
        modules = discover_test_modules()
        assert len(modules) > 20, f"only found {len(modules)} test modules; scan is wrong"

    def test_committed_fixtures_are_not_forbidden(self) -> None:
        """Reading a committed fixture is a claim about the repository."""
        allowed = {"qa_pairs.json", "answer_key_manual.json", "question_paraphrases.json"}
        assert not (allowed & forbidden_literals())

    def test_a_repo_rooted_path_is_seen(self) -> None:
        """The fault's exact shape, as a positive control.

        Without this the exception below could silently swallow everything and
        the guard would pass by matching nothing.
        """
        source = 'FIXTURES = root / "fixtures"\nX = FIXTURES / "calibration.json"\n'
        values = {value for _, value in _literals_of(source)}
        assert "calibration.json" in values

    def test_a_temp_rooted_path_is_exempt(self) -> None:
        source = 'X = tmp_path / "calibration.json"\n'
        values = {value for _, value in _literals_of(source)}
        assert "calibration.json" not in values


def _literals_of(source: str) -> list[tuple[int, str]]:
    """`repo_path_literals` over a source string, for the controls above."""
    tree = ast.parse(source)
    parents = _parents(tree)
    return [
        (node.lineno, node.value)
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and not _is_under_a_temp_root(node, parents)
    ]


@pytest.mark.parametrize("module", discover_test_modules(), ids=lambda p: p.name)
def test_no_test_module_names_a_gitignored_path(module: Path) -> None:
    """The assertion itself, one failure per offending module."""
    forbidden = forbidden_literals()
    offenders = [
        f"{module.relative_to(REPO_ROOT)}:{line} -> {value!r}"
        for line, value in repo_path_literals(module)
        if value in forbidden
    ]
    assert offenders == [], (
        "test modules must not depend on files absent from a clean clone; "
        f"these name gitignored paths: {offenders}. If the value is genuinely "
        "needed, transcribe it into the test with a self-check (see "
        "tests/unit/test_scoring.py's COMMISSIONED artifact) rather than reading "
        "the generated file."
    )
