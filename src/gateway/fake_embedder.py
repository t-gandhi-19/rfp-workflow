"""A deterministic stand-in embedder, for CI only.

CI has no Ollama (CLAUDE.md rule 26), but ingest and the vector-index tests still
need vectors. This produces a stable unit vector from a hash of the text: the
same string always yields the same vector, similar strings do NOT yield similar
vectors, and the width matches the pinned dimension so the Neo4j index accepts
it.

That last point is the trade. These vectors exercise the *plumbing* — ingest,
the index, query shape, idempotency — and say nothing about retrieval quality.
Retrieval quality is measured in Phase 3 against real embeddings, and any eval
that depends on semantic similarity must not run against this.

Enabling it takes an explicit opt-in (`RFP_FAKE_EMBEDDINGS=1`). It is off by
default, `make ingest` never sets it, and a test asserts both.
"""

from __future__ import annotations

import hashlib
import math
import os

#: The opt-in. Deliberately verbose and unlikely to be set by accident.
FAKE_EMBEDDINGS_ENV = "RFP_FAKE_EMBEDDINGS"


def fake_embeddings_enabled(env: dict[str, str] | None = None) -> bool:
    """Whether the stand-in embedder is switched on. Default: no."""
    source = os.environ if env is None else env
    return source.get(FAKE_EMBEDDINGS_ENV, "").strip() == "1"


def fake_embedding(text: str, dimensions: int) -> list[float]:
    """A stable pseudo-random unit vector derived from `text`.

    Built by hashing with a counter until enough bytes exist, so the result
    depends only on the text and the width — not on Python's hash seed, the
    platform, or the order things were embedded in.
    """
    if dimensions <= 0:
        raise ValueError("dimensions must be positive")

    needed = dimensions * 2
    material = bytearray()
    counter = 0
    while len(material) < needed:
        digest = hashlib.sha256(f"{counter}:{text}".encode()).digest()
        material.extend(digest)
        counter += 1

    values: list[float] = []
    for index in range(dimensions):
        chunk = material[index * 2 : index * 2 + 2]
        # Map two bytes onto [-1, 1).
        values.append((int.from_bytes(chunk, "big") / 32768.0) - 1.0)

    norm = math.sqrt(sum(value * value for value in values))
    if norm == 0:  # pragma: no cover - astronomically unlikely
        values[0] = 1.0
        norm = 1.0
    return [value / norm for value in values]


def fake_embeddings(texts: list[str], dimensions: int) -> list[list[float]]:
    return [fake_embedding(text, dimensions) for text in texts]
