"""Every model call leaves through the LiteLLM gateway (CLAUDE.md rule 5).

`src/gateway/` is the only module permitted to import a provider SDK. Everything
else reaches models over HTTP through the proxy, which is what makes the alias
table, the retry policy, the budget ceiling, and the cost accounting single
points of control rather than conventions.

Like the SQL guard, this is an AST walk: `import openai` inside a function body,
a conditional import, or an aliased one are all caught, and none of them would
be caught reliably by grep.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[2] / "src"
GATEWAY = SRC / "gateway"

# Top-level module names that mean "talking to a model vendor directly".
FORBIDDEN_ROOTS = {
    "anthropic",
    "cohere",
    "google.generativeai",
    "google.genai",
    "groq",
    "litellm",
    "mistralai",
    "ollama",
    "openai",
    "vertexai",
}


def _python_files_outside_gateway() -> list[Path]:
    return sorted(p for p in SRC.rglob("*.py") if GATEWAY not in p.parents)


def _file_id(path: Path) -> str:
    return str(path.relative_to(SRC.parent))


def _imported_roots(tree: ast.AST) -> set[str]:
    """Every module root this file imports, however it phrases the import."""
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                roots.add(alias.name)
                roots.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            roots.add(node.module)
            roots.add(node.module.split(".")[0])
    return roots


class TestNoProviderSdkOutsideGateway:
    @pytest.mark.parametrize("path", _python_files_outside_gateway(), ids=_file_id)
    def test_file(self, path: Path) -> None:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        found = _imported_roots(tree) & FORBIDDEN_ROOTS
        assert found == set(), (
            f"{path} imports {sorted(found)}. Provider SDKs belong in src/gateway/ only; "
            "everything else calls the LiteLLM proxy over HTTP."
        )

    def test_the_gateway_is_actually_excluded(self) -> None:
        """If the exclusion silently matched nothing, the guard would be vacuous."""
        scanned = set(_python_files_outside_gateway())
        gateway_files = set(GATEWAY.rglob("*.py"))
        assert gateway_files, "src/gateway/ has no Python files — check the path"
        assert not (scanned & gateway_files)

    def test_it_scans_a_meaningful_number_of_files(self) -> None:
        """Guards against a path typo turning this into a no-op."""
        assert len(_python_files_outside_gateway()) >= 10


class TestTheCheckItselfWorks:
    def test_detects_a_plain_import(self) -> None:
        assert "openai" in _imported_roots(ast.parse("import openai"))

    def test_detects_an_aliased_import(self) -> None:
        assert "anthropic" in _imported_roots(ast.parse("import anthropic as llm"))

    def test_detects_a_from_import(self) -> None:
        assert "groq" in _imported_roots(ast.parse("from groq import Groq"))

    def test_detects_a_submodule_import(self) -> None:
        roots = _imported_roots(ast.parse("import google.generativeai as genai"))
        assert "google.generativeai" in roots

    def test_detects_an_import_nested_in_a_function(self) -> None:
        source = "def call():\n    import openai\n    return openai\n"
        assert "openai" in _imported_roots(ast.parse(source))

    def test_detects_an_import_inside_a_try_block(self) -> None:
        source = "try:\n    import groq\nexcept ImportError:\n    groq = None\n"
        assert "groq" in _imported_roots(ast.parse(source))

    def test_ignores_relative_imports(self) -> None:
        """`from . import x` has no module root to judge."""
        assert _imported_roots(ast.parse("from . import helpers")) == set()

    def test_allows_httpx(self) -> None:
        """The permitted way to reach the gateway."""
        assert not (_imported_roots(ast.parse("import httpx")) & FORBIDDEN_ROOTS)
