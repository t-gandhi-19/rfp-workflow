"""Every script module imports.

WHY THIS EXISTS. `scripts/run.py` shipped importing `configure_logging`, a name
`src/observability/logging.py` does not export — it is `configure_app_logging`.
The whole suite was green: 1685 unit tests, 124 integration tests, ruff and mypy
clean. Nothing caught it because nothing imported the module. mypy did not,
because `scripts/` is outside the packages it is pointed at; the tests did not,
because the entry point is only ever executed by `make run`.

It surfaced on the first real invocation, which is the worst place for an
`ImportError` to surface: after the stack is up, after the graph is ingested,
and in front of whoever was about to watch the demo.

An import is the cheapest possible assertion and it catches the whole class —
a renamed helper, a deleted module, a typo in a symbol. It does NOT prove a
script works; that is what the make-target wiring test and a real run are for.
"""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = REPO_ROOT / "scripts"


def script_modules() -> list[str]:
    """Derived from the directory, so a new script is covered the day it lands."""
    return sorted(
        f"scripts.{path.stem}" for path in SCRIPTS.glob("*.py") if path.stem != "__init__"
    )


class TestEveryScriptImports:
    def test_the_list_is_not_empty(self) -> None:
        """A path typo would make every assertion below vacuous."""
        assert len(script_modules()) >= 5

    @pytest.mark.parametrize("module", script_modules())
    def test_it_imports(self, module: str) -> None:
        """Import only — no `main()`, no side effects. Several of these open
        database connections or call a gateway when run, and this is a check
        that the module is WELL-FORMED, not that the stack is up."""
        importlib.import_module(module)

    def test_the_entry_points_expose_a_main(self) -> None:
        """`make run` and the rest invoke `python -m scripts.<name>`, which
        needs the module to be runnable, not merely importable."""
        for name in ("run", "ingest", "evals", "preflight"):
            module = importlib.import_module(f"scripts.{name}")
            assert hasattr(module, "main"), f"scripts/{name}.py has no main()"
