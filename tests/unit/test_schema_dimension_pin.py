"""The vector-index dimension in schema.cypher matches config/embedding.yaml.

Neo4j accepts no parameter inside index options, so the dimension is a literal
in the Cypher file. That makes it the one number in the system that can drift
away from its config without anything noticing — until embeddings of the wrong
width are silently rejected by the index at ingest time.

Checked statically, by reading the file. The live-index behaviour is also
exercised in the integration suite, but that only runs with a stack up; this
runs everywhere, including on a fresh clone.
"""

from __future__ import annotations

import re
from pathlib import Path

from src.contracts.embedding import embedding_config

SCHEMA_PATH = Path(__file__).resolve().parents[2] / "src" / "graph" / "schema.cypher"

_DIMENSION_RE = re.compile(r"`vector\.dimensions`\s*:\s*(?P<value>\d+)")
_SIMILARITY_RE = re.compile(r"`vector\.similarity_function`\s*:\s*'(?P<value>\w+)'")


def test_schema_dimension_matches_the_pinned_config() -> None:
    schema = SCHEMA_PATH.read_text(encoding="utf-8")
    matches = _DIMENSION_RE.findall(schema)
    assert len(matches) == 1, f"expected exactly one vector.dimensions literal, found {matches}"
    assert int(matches[0]) == embedding_config().model.dimensions


def test_schema_similarity_matches_the_pinned_config() -> None:
    schema = SCHEMA_PATH.read_text(encoding="utf-8")
    matches = _SIMILARITY_RE.findall(schema)
    assert len(matches) == 1
    assert matches[0] == embedding_config().index.similarity


def test_the_index_name_matches_the_pinned_config() -> None:
    """The query passes this name to db.index.vector.queryNodes by parameter."""
    schema = SCHEMA_PATH.read_text(encoding="utf-8")
    assert f"CREATE VECTOR INDEX {embedding_config().index.name} " in schema
