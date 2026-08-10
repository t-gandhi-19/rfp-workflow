"""Neo4j connection management and text normalisation.

Nothing outside this package opens a session. Agents reach the graph only
through the tested functions in :mod:`src.graph.queries`, exposed as MCP tools
(CLAUDE.md rule 4).
"""

from __future__ import annotations

import os
import re
import unicodedata
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from functools import lru_cache

from neo4j import AsyncDriver, AsyncGraphDatabase

#: Punctuation and corporate suffixes stripped before a name is compared.
_SUFFIXES = (
    "incorporated",
    "inc",
    "limited",
    "ltd",
    "llc",
    "plc",
    "gmbh",
    "ag",
    "sa",
    "nv",
    "bv",
    "co",
    "corp",
    "corporation",
    "company",
    "group",
)


def normalise_name(name: str) -> str:
    """Fold a name to the form entity resolution compares on.

    Case, accents, punctuation, and corporate suffixes all vary between how a
    drafter writes a name and how the registry records it. Normalising both
    sides is what makes the closed-world check usable rather than pedantic —
    "CloudNova Partners, Inc." and "cloudnova partners" must resolve to the
    same vendor, while "CloudNova Systems" must not.
    """
    folded = unicodedata.normalize("NFKD", name)
    folded = "".join(ch for ch in folded if not unicodedata.combining(ch))
    folded = folded.casefold()
    folded = re.sub(r"[^a-z0-9\s]", " ", folded)
    tokens = [token for token in folded.split() if token]
    while tokens and tokens[-1] in _SUFFIXES:
        tokens.pop()
    return " ".join(tokens)


def neo4j_uri() -> str:
    return os.environ.get("NEO4J_URI", "bolt://localhost:7687")


def neo4j_auth() -> tuple[str, str]:
    return (
        os.environ.get("NEO4J_USER", "neo4j"),
        os.environ.get("NEO4J_PASSWORD", ""),
    )


@lru_cache(maxsize=1)
def get_driver() -> AsyncDriver:
    """Process-wide async driver, created on first use."""
    return AsyncGraphDatabase.driver(neo4j_uri(), auth=neo4j_auth())


async def close_driver() -> None:
    if get_driver.cache_info().currsize:
        await get_driver().close()
        get_driver.cache_clear()


@asynccontextmanager
async def session(driver: AsyncDriver | None = None) -> AsyncIterator[object]:
    """A session against the configured database.

    `driver` is injectable so tests can supply their own connection.
    """
    resolved = driver or get_driver()
    async with resolved.session() as neo_session:
        yield neo_session
