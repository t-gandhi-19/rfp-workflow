"""Measure the calibration anchors against real embeddings. Run via `make calibrate`.

Raw cosine is not a number that means anything on its own. Its usable range
depends on the embedding model, on whether task prefixes are applied, and on how
alike the corpus is to itself — so a floor picked by eye is a floor that means
something different on the next corpus, and nothing says so.

This measures the two anchors the floor is derived from:

* **background** — cross-subject pairs. What "unrelated, but both about cloud
  migration" scores.
* **same-subject** — pairs sharing a `topic_family`: the paraphrases, plus the
  supersession chains. What "genuinely the same question" scores.

Every question is embedded TWICE, once with each task prefix, and only ever
compared across them: `search_query:` question i against `search_document:`
question j. That is the only geometry the floor will ever judge, and it is not
the same distribution as document-against-document — measuring the wrong one is
what previously made the background p99 appear to sit above the lowest real
match.

    python -m scripts.calibrate            # measure, write, persist
    python -m scripts.calibrate --dry-run  # measure and print, write nothing
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from datetime import UTC, datetime

import httpx

from src.contracts.embedding import EmbedRole, embedding_config
from src.gateway.client import GatewayClient
from src.gateway.fake_embedder import fake_embeddings, fake_embeddings_enabled
from src.retrieval.calibration import (
    CALIBRATION_PATH,
    CalibrationArtifact,
    CalibrationError,
    calibration_corpus,
    compute,
    corpus_hash,
    save,
)

BATCH = 16


async def _embed(texts: list[str], role: EmbedRole) -> list[list[float]]:
    """Embed one side of the geometry, through the gateway or the CI stand-in."""
    config = embedding_config()
    if fake_embeddings_enabled():
        return fake_embeddings(texts, config.model.dimensions, role=role)

    client = GatewayClient.from_env()
    vectors: list[list[float]] = []
    for start in range(0, len(texts), BATCH):
        vectors.extend(
            await client.embed(texts[start : start + BATCH], alias=config.model.alias, role=role)
        )
    bad = sorted({len(vector) for vector in vectors if len(vector) != config.model.dimensions})
    if bad:
        raise CalibrationError(
            f"gateway returned {bad}-dim vectors; config/embedding.yaml pins "
            f"{config.model.dimensions}. Run: make preflight"
        )
    return vectors


async def measure() -> CalibrationArtifact:
    rows = calibration_corpus()
    texts = [row["question"] for row in rows]
    ids = [row["question_id"] for row in rows]

    # Both sides of the geometry, from the same source text.
    query_vectors = dict(zip(ids, await _embed(texts, EmbedRole.QUERY), strict=True))
    document_vectors = dict(zip(ids, await _embed(texts, EmbedRole.DOCUMENT), strict=True))

    return compute(
        query_vectors=query_vectors,
        document_vectors=document_vectors,
        topic_by_question={row["question_id"]: row["topic_family"] for row in rows},
        embed_model_tag=embedding_config().model.tag,
        corpus_content_hash=corpus_hash(rows),
        # Passed in rather than taken inside `compute`, so the measurement itself
        # stays a pure function of its inputs and is testable without freezing a
        # clock.
        computed_at=datetime.now(UTC).isoformat(timespec="seconds"),
    )


async def persist(artifact: CalibrationArtifact) -> str:
    """Record the artifact in Postgres, through the write-api.

    Never a direct database write: Postgres writes go only through the write-api
    (CLAUDE.md rule 4), and calibration is not an exception to that just because
    it happens to run from a script.
    """
    port = os.environ.get("WRITE_API_PORT", "8001")
    base = os.environ.get("WRITE_API_BASE") or f"http://localhost:{port}"
    token = os.environ.get("WRITE_API_TOKEN", "")
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(
            f"{base.rstrip('/')}/v1/calibration",
            json=artifact.model_dump(),
            headers={"Authorization": f"Bearer {token}"} if token else {},
        )
    if response.status_code >= 300:
        raise CalibrationError(
            f"write-api refused the calibration artifact ({response.status_code}): "
            f"{response.text[:300]}"
        )
    return str(response.json().get("id", ""))


def report(artifact: CalibrationArtifact) -> str:
    """The statistics, in the geometry they were measured in."""
    floor = artifact.derived_floor()
    lines = [
        "",
        "calibration — query x document geometry",
        "",
        f"  embed model        {artifact.embed_model_tag}",
        f"  corpus             {artifact.corpus_hash}",
        f"  computed at        {artifact.computed_at}",
        "",
        f"  background pairs   {artifact.background_pair_count}",
        f"    p50              {artifact.bg_p50:.4f}",
        f"    p95              {artifact.bg_p95:.4f}",
        f"    p99              {artifact.bg_p99:.4f}",
        "",
        f"  same-subject pairs {artifact.same_topic_pair_count}",
        f"    p05              {artifact.same_topic_p05:.4f}",
        f"    p50              {artifact.same_topic_p50:.4f}",
        "",
        f"  separation         {artifact.separation:+.4f}  (same_topic_p05 - bg_p99)",
        "",
        "  in calibrated space:",
        f"    bg_p99           {artifact.calibrated(artifact.bg_p99):.4f}",
        f"    same_topic_p05   {artifact.calibrated(artifact.same_topic_p05):.4f}",
        f"    DERIVED FLOOR    {floor:.4f}   <- config/scoring.yaml "
        "retrieval.match_floor_calibrated",
        "",
    ]
    return "\n".join(lines)


async def run(*, dry_run: bool, skip_persist: bool) -> int:
    if fake_embeddings_enabled():
        sys.stderr.write(
            "WARNING: RFP_FAKE_EMBEDDINGS=1 — calibrating against the deterministic "
            "stand-in embedder. The geometry is real but the model is not; these anchors "
            "describe CI, not production retrieval.\n"
        )

    artifact = await measure()
    sys.stdout.write(report(artifact))

    if dry_run:
        sys.stdout.write("--dry-run: nothing written.\n\n")
        return 0

    save(artifact)
    sys.stdout.write(f"wrote {CALIBRATION_PATH}\n")

    if not skip_persist:
        record_id = await persist(artifact)
        sys.stdout.write(f"recorded in Postgres via the write-api (id {record_id})\n")
    sys.stdout.write("\n")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Measure the retrieval calibration anchors")
    parser.add_argument(
        "--dry-run", action="store_true", help="measure and print without writing anything"
    )
    parser.add_argument(
        "--skip-persist",
        action="store_true",
        help="write the fixture cache but do not call the write-api (CI has no stack)",
    )
    args = parser.parse_args()
    try:
        return asyncio.run(run(dry_run=args.dry_run, skip_persist=args.skip_persist))
    except CalibrationError as exc:
        sys.stderr.write(f"\ncalibration failed: {exc}\n\n")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
