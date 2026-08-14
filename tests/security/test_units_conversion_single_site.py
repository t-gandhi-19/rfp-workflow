"""The index-score conversion exists at exactly one site (amendment S).

`2 * score - 1` turns Neo4j's cosine-index score into an actual cosine. It is
correct in exactly one place — the boundary in `src.graph.queries` that knows
where the number came from — and dangerous everywhere else:

* applied TWICE, a cosine of 0.6 becomes 0.2;
* applied to a number that is already a cosine, every score is silently halved
  and shifted, which is the original defect wearing different clothes;
* NOT applied where it should be, and the floor judges the wrong scale — which
  is what shipped, and made every golden question MATCH including the three the
  corpus cannot answer.

A second copy is also a second thing to forget when the index similarity
changes. `find_similar_questions` refuses outright for a non-cosine index; a
stray conversion elsewhere would not.

Same family as the SDK-import and SQL-interpolation walks, and for the same
reason: the invariant is "this appears in one place", which no amount of local
review of a diff can confirm.
"""

from __future__ import annotations

import ast
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SCANNED = (REPO_ROOT / "src", REPO_ROOT / "scripts")

#: The one function permitted to hold it, as `module:function`.
CANONICAL_SITE = ("src/graph/queries.py", "index_score_to_cosine")


def _python_files() -> list[Path]:
    return sorted(path for root in SCANNED for path in root.rglob("*.py"))


def _is_the_conversion(node: ast.AST) -> bool:
    """Does this expression compute `2 * x - 1` or `x * 2 - 1`?

    Matched structurally rather than by text, so whitespace, `2.0` versus `2`
    and the operand order are all the same finding, and a mention in a comment
    or a docstring correctly is not.
    """
    if not isinstance(node, ast.BinOp) or not isinstance(node.op, ast.Sub):
        return False
    if not (isinstance(node.right, ast.Constant) and node.right.value in (1, 1.0)):
        return False
    left = node.left
    if not isinstance(left, ast.BinOp) or not isinstance(left.op, ast.Mult):
        return False
    operands = (left.left, left.right)
    return any(isinstance(side, ast.Constant) and side.value in (2, 2.0) for side in operands)


def _enclosing_function(tree: ast.AST, target: ast.AST) -> str | None:
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            for inner in ast.walk(node):
                if inner is target:
                    return node.name
    return None


def find_conversion_sites() -> list[tuple[str, str | None, int]]:
    """Every `2 * x - 1` in src/ and scripts/, as (path, function, line)."""
    sites: list[tuple[str, str | None, int]] = []
    for path in _python_files():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.BinOp) and _is_the_conversion(node):
                sites.append(
                    (
                        path.relative_to(REPO_ROOT).as_posix(),
                        _enclosing_function(tree, node),
                        node.lineno,
                    )
                )
    return sites


class TestTheMatcher:
    """A scan that matches nothing passes silently."""

    def test_it_matches_the_canonical_form(self) -> None:
        tree = ast.parse("y = 2.0 * score - 1.0")
        assert any(_is_the_conversion(node) for node in ast.walk(tree))

    def test_it_matches_the_operands_the_other_way_round(self) -> None:
        tree = ast.parse("y = score * 2 - 1")
        assert any(_is_the_conversion(node) for node in ast.walk(tree))

    def test_it_ignores_unrelated_arithmetic(self) -> None:
        for source in ("y = 2 * score + 1", "y = 3 * score - 1", "y = 2 * score - 2"):
            tree = ast.parse(source)
            assert not any(_is_the_conversion(node) for node in ast.walk(tree))

    def test_a_docstring_mention_is_not_a_site(self) -> None:
        tree = ast.parse('"""cos = 2 * score - 1, see the boundary."""')
        assert not any(_is_the_conversion(node) for node in ast.walk(tree))


def test_the_conversion_appears_at_exactly_one_site() -> None:
    sites = find_conversion_sites()
    assert len(sites) == 1, (
        f"the index-score conversion must exist once and only once; found {len(sites)}: {sites}. "
        "Applied twice it inverts the scale; applied to an already-converted cosine it "
        "reproduces the original defect. Convert at the graph boundary and pass a cosine "
        "everywhere else."
    )


def test_the_one_site_is_the_documented_boundary() -> None:
    """Where it lives is part of the invariant, not just how many there are."""
    path, function, _line = find_conversion_sites()[0]
    assert (path, function) == CANONICAL_SITE
