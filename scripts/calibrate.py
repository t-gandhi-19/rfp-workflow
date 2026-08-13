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
from pathlib import Path

import httpx

from src.contracts.embedding import EmbedRole, embedding_config
from src.contracts.thresholds import config_dir, reload_config, scoring_config
from src.gateway.client import GatewayClient
from src.gateway.fake_embedder import fake_embeddings, fake_embeddings_enabled
from src.retrieval.calibration import (
    CALIBRATION_PATH,
    CalibrationArtifact,
    CalibrationError,
    calibration_corpus,
    check_erosion_ratchet,
    compute,
    corpus_hash,
    indexed_ids,
    save,
)

BATCH = 16

CONFIG_PATH = config_dir() / "scoring.yaml"


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
        # The population rule (D18): only indexed originals may be documents.
        indexed_ids=indexed_ids(rows),
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
    floor = artifact.derived_floor
    guards = scoring_config().calibration.separation_guards
    commissioned = (
        "UNCOMMISSIONED" if guards.commissioned_tail is None else f"{guards.commissioned_tail:.4f}"
    )
    required = (
        "n/a"
        if guards.commissioned_tail is None
        else f"{guards.commissioned_tail * guards.tail_retention:.4f}"
    )
    lines = [
        "",
        "calibration",
        "",
        f"  geometry           {artifact.geometry}",
        f"  population         {artifact.population}",
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
        "  separation guards (D18):",
        f"    tier 1 body      {artifact.body_separation:+.4f}  (same_p50 - bg_p50)"
        f"   min {guards.body_min}",
        f"    tier 2 tail      {artifact.separation:+.4f}  (same_p05 - bg_p99)"
        f"   commissioned {commissioned}, retention {guards.tail_retention:.0%}"
        f" -> required {required}",
        "",
        "  in calibrated space:",
        f"    bg_p99           {artifact.calibrated(artifact.bg_p99):.4f}",
        f"    same_topic_p05   {artifact.calibrated(artifact.same_topic_p05):.4f}",
        f"    DERIVED FLOOR    {floor:.4f}   <- stored on the artifact",
        "",
    ]
    return "\n".join(lines)


#: The generated block in scoring.yaml. Rewritten in place by `stamp_config`.
_STAMP_START = "  #   last derived floor:"
_STAMP_LINES = 7

#: The commissioned-baseline line, rewritten by `commission`.
_TAIL_KEY = "    commissioned_tail:"
#: The tier-1 floor, rewritten at commissioning from the measured median gap.
_BODY_KEY = "    body_min:"
#: Tier 1 is set at half the commissioned median gap: tolerant of drift, and
#: firing only on the wholesale collapse it exists to catch.
BODY_MIN_FRACTION = 0.5


def commission(artifact: CalibrationArtifact, path: Path | None = None) -> tuple[float, float]:
    """Set both guard baselines from a real measurement. The one moment they move.

    Tier 2's baseline cannot be chosen in advance — nobody knows what tail
    separation a given corpus and embedder will produce — so it is measured once
    and thereafter defended. Tier 1's floor is derived from the same measurement
    at `BODY_MIN_FRACTION` of the median gap.

    This is deliberately a SEPARATE command from `make calibrate`. Ordinary
    calibration must never be able to move the baseline it is being judged
    against; a ratchet that resets itself on every run is a ratchet that never
    catches anything.
    """
    target = path or CONFIG_PATH
    body_min = round(artifact.body_separation * BODY_MIN_FRACTION, 4)
    tail = round(artifact.separation, 4)

    # VALIDATE BEFORE WRITING. A measurement that cannot be a baseline must not
    # become one: an earlier version wrote first and validated on reload, which
    # left a negative `commissioned_tail` in the config after a failed run and
    # made every subsequent command fail at config load — including the ones
    # needed to diagnose it.
    if tail <= 0:
        raise CalibrationError(
            f"refusing to commission a non-positive tail separation ({tail:+.4f}). The "
            f"weakest genuine match (p05 {artifact.same_topic_p05:.4f}) does not clear the "
            f"strongest unrelated pair (p99 {artifact.bg_p99:.4f}), so there is no margin to "
            f"defend. This is a corpus or embedding-model question, not a threshold to set."
        )
    if body_min <= 0:
        raise CalibrationError(
            f"refusing to commission a non-positive body floor ({body_min:+.4f}); the "
            f"distributions do not separate in bulk."
        )

    lines = target.read_text(encoding="utf-8").splitlines()
    seen_tail = seen_body = False
    for index, line in enumerate(lines):
        if line.startswith(_TAIL_KEY):
            lines[index] = f"{_TAIL_KEY} {tail}"
            seen_tail = True
        elif line.startswith(_BODY_KEY):
            lines[index] = f"{_BODY_KEY} {body_min}"
            seen_body = True
    if not (seen_tail and seen_body):
        raise CalibrationError(
            f"could not find both guard keys in {target} "
            f"(commissioned_tail: {seen_tail}, body_min: {seen_body})"
        )
    target.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    return body_min, tail


def stamp_config(artifact: CalibrationArtifact, path: Path | None = None) -> None:
    """Record what the derivation rule last produced, for human readers.

    Amendment O keeps the floor VALUE out of config and in the artifact, which
    is right for the runtime and awkward for anyone reading the YAML to
    understand what the rule above it does. So the value is written back as a
    generated comment: visible, dated, and inert — retrieval never reads it.

    Rewritten rather than appended, so re-running calibration does not
    accumulate a history of stamps in a config file.
    """
    target = path or CONFIG_PATH
    lines = target.read_text(encoding="utf-8").splitlines()
    for index, line in enumerate(lines):
        if line.startswith(_STAMP_START):
            lines[index : index + _STAMP_LINES] = [
                f"  #   last derived floor: {artifact.derived_floor:.4f}",
                f"  #   geometry:           {artifact.geometry}",
                f"  #   population:         {artifact.population}",
                f"  #   computed at:        {artifact.computed_at}",
                f"  #   embed model:        {artifact.embed_model_tag}",
                f"  #   body separation:    {artifact.body_separation:+.4f}",
                f"  #   tail separation:    {artifact.separation:+.4f}",
            ]
            target.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
            return
    raise CalibrationError(
        f"no generated stamp block found in {target}. Expected a line starting "
        f"'{_STAMP_START}' under calibration.floor_derivation."
    )


async def run(*, dry_run: bool, skip_persist: bool, commissioning: bool) -> int:
    if fake_embeddings_enabled():
        sys.stderr.write(
            "WARNING: RFP_FAKE_EMBEDDINGS=1 — calibrating against the deterministic "
            "stand-in embedder. The geometry is real but the model is not; these anchors "
            "describe CI, not production retrieval.\n"
        )

    # `measure` enforces tier 1 (collapse) via compute(). Tier 2 is checked
    # separately, and at commissioning it is SET from this measurement instead.
    artifact = await measure()

    if commissioning:
        if fake_embeddings_enabled():
            raise CalibrationError(
                "refusing to commission the guards against the stand-in embedder. The "
                "baseline they defend has to come from the model production actually uses."
            )
        body_min, tail = commission(artifact)
        reload_config()
        sys.stdout.write(report(artifact))
        sys.stdout.write(
            f"COMMISSIONED from this measurement:\n"
            f"  body_min          {body_min}   ({BODY_MIN_FRACTION:.0%} of the measured "
            f"median gap {artifact.body_separation:.4f})\n"
            f"  commissioned_tail {tail}   (the measured tail gap)\n"
        )
    else:
        sys.stdout.write(report(artifact))

    # Always enforced, including immediately after commissioning — so a
    # commissioning run that somehow produced a value it could not itself
    # satisfy would fail rather than be recorded.
    check_erosion_ratchet(artifact)

    if dry_run:
        sys.stdout.write("--dry-run: nothing further written.\n\n")
        return 0

    save(artifact)
    sys.stdout.write(f"wrote {CALIBRATION_PATH}\n")
    stamp_config(artifact)
    sys.stdout.write(f"stamped the derived floor into {CONFIG_PATH.name} (as a comment)\n")

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
    parser.add_argument(
        "--commission",
        action="store_true",
        help=(
            "SET the separation guard baselines from this measurement. The one command "
            "that may move them; ordinary calibration is judged against them."
        ),
    )
    args = parser.parse_args()
    try:
        return asyncio.run(
            run(
                dry_run=args.dry_run,
                skip_persist=args.skip_persist,
                commissioning=args.commission,
            )
        )
    except CalibrationError as exc:
        sys.stderr.write(f"\ncalibration failed: {exc}\n\n")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
