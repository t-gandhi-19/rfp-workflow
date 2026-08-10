"""Apply src/graph/schema.cypher. Idempotent — safe to run on every startup.

The Neo4j driver takes one statement per call, so the file is split on lines
containing only `;`.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

from src.graph.driver import close_driver, get_driver

SCHEMA_PATH = Path(__file__).resolve().parents[1] / "src" / "graph" / "schema.cypher"


def statements(text: str) -> list[str]:
    """Split the schema file into executable statements, dropping comments."""
    chunks: list[str] = []
    current: list[str] = []
    for raw in text.splitlines():
        line = raw.strip()
        if line == ";":
            statement = "\n".join(current).strip()
            if statement:
                chunks.append(statement)
            current = []
            continue
        if line.startswith("//") or not line:
            continue
        current.append(raw)
    tail = "\n".join(current).strip()
    if tail:
        chunks.append(tail)
    return chunks


async def apply() -> int:
    schema = SCHEMA_PATH.read_text(encoding="utf-8")
    todo = statements(schema)
    driver = get_driver()
    applied = 0
    try:
        async with driver.session() as session:
            for statement in todo:
                await session.run(statement)
                applied += 1
    finally:
        await close_driver()
    sys.stdout.write(f"graph schema: {applied} statement(s) applied (idempotent)\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(apply()))
