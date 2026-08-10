"""No SQL is ever built by string interpolation (CLAUDE.md rule 4, build prompt §17).

This walks the AST rather than grepping, because a grep for `execute(f"` is
defeated by a line break, a variable, or a differently-quoted string. Two checks
run together:

1. **Call sites** — anything handed to `execute`/`text`/similar must not be an
   f-string, a `%` format, a `+` concatenation, or a `.format()` call.
2. **SQL-shaped f-strings anywhere** — catches the common evasion of building the
   query into a variable first and executing the variable later.

Both are deliberately strict. If a legitimate case ever needs an exception, it
should be argued in review and added to ALLOWED_DYNAMIC_SQL below, not worked
around by loosening the check (CLAUDE.md rule 5's "do not weaken them" applies
here too).
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[2] / "src"

# Functions whose first argument is a SQL string.
SQL_SINKS = {
    "execute",
    "executemany",
    "exec_driver_sql",
    "execute_batch",
    "execute_values",
    "text",
}

# Keyword pairs that make a string SQL rather than prose. Both halves must be
# present, so "Select the best candidate from the list" is not flagged.
SQL_SIGNATURES = [
    ("select", "from"),
    ("insert", "into"),
    ("update", "set"),
    ("delete", "from"),
    ("drop", "table"),
    ("alter", "table"),
    ("create", "table"),
]

# Documented, reviewed exceptions. Empty, and intended to stay that way.
ALLOWED_DYNAMIC_SQL: set[str] = set()


def _python_files() -> list[Path]:
    return sorted(SRC.rglob("*.py"))


def _file_id(path: Path) -> str:
    return str(path.relative_to(SRC.parent))


def _is_interpolated(node: ast.expr) -> str | None:
    """Return a description if `node` builds a string dynamically, else None."""
    if isinstance(node, ast.JoinedStr):
        return "f-string"
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mod):
        return "%-format"
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        return "string concatenation"
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "format"
    ):
        return ".format() call"
    return None


def _looks_like_sql(text: str) -> bool:
    lowered = text.lower()
    return any(first in lowered and second in lowered for first, second in SQL_SIGNATURES)


def _fstring_literal_text(node: ast.JoinedStr) -> str:
    return " ".join(
        part.value
        for part in node.values
        if isinstance(part, ast.Constant) and isinstance(part.value, str)
    )


class TestNoInterpolationAtSqlCallSites:
    @pytest.mark.parametrize("path", _python_files(), ids=_file_id)
    def test_file(self, path: Path) -> None:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        offenders: list[str] = []

        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not node.args:
                continue
            func = node.func
            name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", None)
            if name not in SQL_SINKS:
                continue
            problem = _is_interpolated(node.args[0])
            if problem and f"{path.name}:{node.lineno}" not in ALLOWED_DYNAMIC_SQL:
                offenders.append(f"{path}:{node.lineno} — {name}() called with a {problem}")

        assert offenders == [], "SQL built by interpolation:\n" + "\n".join(offenders)


class TestNoSqlShapedFStrings:
    """Catches SQL assembled into a variable and executed somewhere else."""

    @pytest.mark.parametrize("path", _python_files(), ids=_file_id)
    def test_file(self, path: Path) -> None:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        offenders = [
            f"{path}:{node.lineno} — f-string containing SQL keywords"
            for node in ast.walk(tree)
            if isinstance(node, ast.JoinedStr)
            and _looks_like_sql(_fstring_literal_text(node))
            and f"{path.name}:{node.lineno}" not in ALLOWED_DYNAMIC_SQL
        ]
        assert offenders == [], "SQL-shaped f-strings:\n" + "\n".join(offenders)


class TestTheCheckItselfWorks:
    """A guard that never fires is indistinguishable from a guard that is broken."""

    def test_detects_an_fstring_at_a_call_site(self) -> None:
        tree = ast.parse('cursor.execute(f"SELECT * FROM runs WHERE id = {run_id}")')
        call = next(n for n in ast.walk(tree) if isinstance(n, ast.Call))
        assert _is_interpolated(call.args[0]) == "f-string"

    def test_detects_percent_formatting(self) -> None:
        tree = ast.parse('cursor.execute("SELECT * FROM runs WHERE id = %s" % run_id)')
        call = next(n for n in ast.walk(tree) if isinstance(n, ast.Call))
        assert _is_interpolated(call.args[0]) == "%-format"

    def test_detects_concatenation(self) -> None:
        tree = ast.parse('cursor.execute("SELECT * FROM runs WHERE id = " + run_id)')
        call = next(n for n in ast.walk(tree) if isinstance(n, ast.Call))
        assert _is_interpolated(call.args[0]) == "string concatenation"

    def test_allows_a_parameterized_literal(self) -> None:
        """The correct form must not trip the check."""
        tree = ast.parse('cursor.execute("SELECT * FROM runs WHERE id = %s", (run_id,))')
        call = next(n for n in ast.walk(tree) if isinstance(n, ast.Call))
        assert _is_interpolated(call.args[0]) is None

    def test_sql_detection_is_deliberately_conservative(self) -> None:
        """It flags real SQL, and accepts also flagging prose that reads like SQL.

        A false positive costs one reviewed line in ALLOWED_DYNAMIC_SQL; a false
        negative costs an injection. The trade is intentional, and prompts live
        in `config/prompts/` rather than in Python, so SQL-shaped prose is
        unlikely to appear in `src/` anyway.
        """
        assert _looks_like_sql("SELECT id FROM runs") is True
        assert _looks_like_sql("INSERT INTO drafts VALUES (%s)") is True
        assert _looks_like_sql("Select the strongest candidate from the shortlist") is True
        assert _looks_like_sql("Summarize this answer for the reviewer") is False

    def test_sql_shaped_fstring_is_detected(self) -> None:
        tree = ast.parse('query = f"SELECT * FROM answers WHERE id = {answer_id}"')
        node = next(n for n in ast.walk(tree) if isinstance(n, ast.JoinedStr))
        assert _looks_like_sql(_fstring_literal_text(node))
