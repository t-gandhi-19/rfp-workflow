"""The harness's entry point — argument handling and the order it does things in.

Everything here runs without a database, a graph, or a model. The categories
themselves are covered by `test_eval_extraction.py` and the retrieval eval's own
suite; what is left, and what these cover, is the wiring the CLI decides: which
categories run, what a deselected one reports, and when the SHA is read.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from src.evals.registry import IMPLEMENTED, Category, deselected

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "evals.py"


def run_function() -> ast.AsyncFunctionDef:
    tree = ast.parse(SCRIPT.read_text(encoding="utf-8"), filename=str(SCRIPT))
    return next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.AsyncFunctionDef) and node.name == "run"
    )


def statement_index(predicate: object) -> int:
    """Position of the first top-level statement in `run` matching `predicate`."""
    for index, node in enumerate(run_function().body):
        if predicate(ast.dump(node)):  # type: ignore[operator]
            return index
    return -1


class TestTheShaIsReadBeforeAnythingIsMeasured:
    """A rerank-ON run takes ~35 minutes.

    Reading HEAD at the END lets a commit landing during the run relabel the
    numbers: the code that produced them was loaded at process start, so a SHA
    read afterwards can name a commit whose code never ran. It loses the
    `-dirty` marking in the other direction too — start dirty, commit mid-run,
    and the numbers get attributed to a clean tree they were never measured
    against. That happened: a run begun at one commit reported at another.

    Asserted structurally rather than by running the harness, because reproducing
    it honestly would mean committing halfway through a 35-minute measurement.
    """

    def test_git_sha_is_called_exactly_once(self) -> None:
        source = SCRIPT.read_text(encoding="utf-8")
        assert source.count("sha = git_sha()") == 1

    def test_it_is_read_before_the_categories_run(self) -> None:
        sha_at = statement_index(lambda dump: "git_sha" in dump)
        extraction_at = statement_index(lambda dump: "run_extraction_eval" in dump)
        retrieval_at = statement_index(lambda dump: "_retrieval" in dump)

        assert sha_at >= 0, "run() no longer reads the SHA"
        assert extraction_at >= 0 or retrieval_at >= 0, "run() no longer runs a category"
        for position in (extraction_at, retrieval_at):
            if position >= 0:
                assert sha_at < position, (
                    "the SHA must be captured before any measurement, or a commit "
                    "landing mid-run relabels numbers the new code never produced"
                )


class TestCategorySelection:
    def test_the_default_is_every_implemented_category(self) -> None:
        source = SCRIPT.read_text(encoding="utf-8")
        assert 'default=",".join(sorted(IMPLEMENTED))' in source

    def test_a_deselected_category_is_reported_not_dropped(self) -> None:
        """Dropping it would make the report claim the category does not exist."""
        result = deselected(Category.RETRIEVAL)
        assert result.key == Category.RETRIEVAL
        assert result.metrics == []

    def test_a_deselected_category_says_deselected_not_unbuilt(self) -> None:
        """ "This run did not execute it" and "this code does not exist" are
        different facts, and CI relies on the difference."""
        reason = deselected(Category.RETRIEVAL).not_implemented_reason or ""
        assert "DESELECTED" in reason
        assert "not about the build" in reason

    @pytest.mark.parametrize("key", sorted(IMPLEMENTED))
    def test_every_implemented_category_can_be_deselected(self, key: str) -> None:
        assert deselected(key).not_implemented_reason
