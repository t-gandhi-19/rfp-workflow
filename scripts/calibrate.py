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
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

import httpx

from src.contracts.embedding import EmbedRole, embedding_config
from src.contracts.thresholds import ScoringConfig, config_dir, reload_config, scoring_config
from src.gateway.client import GatewayClient
from src.gateway.fake_embedder import fake_embeddings, fake_embeddings_enabled
from src.retrieval.calibration import (
    BAITS,
    CALIBRATION_PATH,
    FIXTURES,
    UNANSWERABLE,
    CalibrationArtifact,
    CalibrationError,
    ProbeLanding,
    calibration_corpus,
    check_floor_discrimination,
    compute,
    corpus_hash,
    cosine,
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


def golden_probes() -> list[dict[str, str]]:
    """The twenty golden questions, tagged by what they should do.

    Read from the hand-written answer key rather than inferred: the key is the
    independent ground truth (D16), and which questions are unanswerable is a
    zero-tolerance property of the golden set, not a heuristic.
    """
    with (FIXTURES / "answer_key_manual.json").open(encoding="utf-8") as handle:
        key = json.load(handle)
    probes = []
    for question in key["questions"]:
        number = question["number"]
        if number in UNANSWERABLE:
            kind = "unanswerable"
        elif number in BAITS:
            kind = "bait"
        elif question["expected_best_match_answer_id"]:
            kind = "answerable"
        else:  # pragma: no cover - the key has no other shape
            kind = "other"
        probes.append({"number": number, "kind": kind, "text": question["text"]})
    return probes


async def land_probes(artifact: CalibrationArtifact) -> list[ProbeLanding]:
    """Embed each golden question and land its best corpus match on the floor.

    Compared against INDEXED ORIGINALS only, the same document side the D18
    population rule gives calibration and the same one retrieval sees.
    """
    rows = calibration_corpus()
    documents = sorted(indexed_ids(rows))
    text_by_id = {row["question_id"]: row["question"] for row in rows}

    document_vectors = await _embed([text_by_id[qid] for qid in documents], EmbedRole.DOCUMENT)
    probes = golden_probes()
    query_vectors = await _embed([probe["text"] for probe in probes], EmbedRole.QUERY)

    landings: list[ProbeLanding] = []
    for probe, query_vector in zip(probes, query_vectors, strict=True):
        best = max(cosine(query_vector, vector) for vector in document_vectors)
        calibrated = artifact.calibrated(best)
        # Signed so positive is always correct, whichever side the probe belongs.
        margin = (
            artifact.derived_floor - calibrated
            if probe["kind"] in {"unanswerable", "bait"}
            else calibrated - artifact.derived_floor
        )
        landings.append(
            ProbeLanding(
                number=probe["number"],
                kind=probe["kind"],
                raw=best,
                calibrated=calibrated,
                margin=margin,
            )
        )
    return landings


def probe_report(landings: list[ProbeLanding], *, config: ScoringConfig | None = None) -> str:
    """Every probe, its margin, and the baseline it is judged against."""
    resolved = config or scoring_config()
    rule = resolved.calibration.separation_guards.floor_discrimination
    baseline = rule.commissioned
    commissioned = {
        "2.7": baseline.unanswerable_2_7,
        "2.8": baseline.unanswerable_2_8,
        "3.5": baseline.unanswerable_3_5,
    }

    lines = ["", "  tier 2 — floor discrimination (D19), probes gate but never derive:", ""]
    for landing in sorted(landings, key=lambda item: item.number):
        if landing.kind == "unanswerable":
            base = commissioned[landing.number]
            required = "n/a" if base is None else f"{base * rule.retention:.4f}"
            verdict = "BELOW floor" if landing.margin > 0 else "AT OR ABOVE FLOOR (!)"
            note = f"GATED, required >= {required}"
        elif landing.kind == "answerable":
            verdict = "above floor" if landing.margin > 0 else "BELOW FLOOR (!)"
            note = "gated as part of the minimum"
        else:
            verdict = "below floor" if landing.margin > 0 else "clears floor"
            note = "ADVISORY ONLY, never gated"
        lines.append(
            f"    {landing.number:<5} {landing.kind:<12} raw {landing.raw:.4f}  "
            f"cal {landing.calibrated:.4f}  margin {landing.margin:+.4f}  {verdict}  ({note})"
        )

    answerable = [item for item in landings if item.kind == "answerable"]
    if answerable:
        weakest = min(answerable, key=lambda item: item.margin)
        required = (
            "n/a"
            if baseline.min_answerable is None
            else f"{baseline.min_answerable * rule.retention:.4f}"
        )
        lines.append("")
        lines.append(
            f"    weakest answerable: {weakest.number} at {weakest.margin:+.4f}  "
            f"GATED, required >= {required}"
        )
    lines.append("")
    return "\n".join(lines)


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
        f"  tier 1 body gap    {artifact.body_separation:+.4f}  (same_p50 - bg_p50)"
        f"   GATED, min {guards.body_min}",
        f"  tail gap           {artifact.separation:+.4f}  (same_p05 - bg_p99)"
        f"   DIAGNOSTIC ONLY, not gated (D19)",
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

#: Lines rewritten at commissioning. Flat scalars on their own lines, so the
#: rewrite is a substitution rather than a YAML round-trip that would discard
#: every comment in the file.
_BODY_KEY = "    body_min:"
_PROBE_KEYS = {
    "2.7": "        unanswerable_2_7:",
    "2.8": "        unanswerable_2_8:",
    "3.5": "        unanswerable_3_5:",
}
_MIN_ANSWERABLE_KEY = "        min_answerable:"
#: Tier 1 is set at half the commissioned median gap: tolerant of drift, and
#: firing only on the wholesale collapse it exists to catch.
BODY_MIN_FRACTION = 0.5


def commission(
    artifact: CalibrationArtifact,
    landings: list[ProbeLanding],
    path: Path | None = None,
) -> dict[str, float]:
    """Set every guard baseline from a real measurement. The one moment they move.

    Tier 1's floor comes from the measured median gap at `BODY_MIN_FRACTION`.
    Tier 2's baselines are the probe margins against the freshly derived floor —
    they cannot be chosen in advance, because nobody knows what margins a given
    corpus and embedder will produce.

    Deliberately a SEPARATE command from `make calibrate`. Ordinary calibration
    must never move the baseline it is judged against; a ratchet that resets
    itself on every run never catches anything.
    """
    target = path or CONFIG_PATH
    values: dict[str, float] = {"body_min": round(artifact.body_separation * BODY_MIN_FRACTION, 4)}

    by_number = {landing.number: landing for landing in landings}
    for number in UNANSWERABLE:
        if number not in by_number:
            raise CalibrationError(f"cannot commission: probe {number} was not measured")
        values[number] = round(by_number[number].margin, 4)

    answerable = [landing for landing in landings if landing.kind == "answerable"]
    if not answerable:
        raise CalibrationError("cannot commission: no answerable probes were measured")
    values["min_answerable"] = round(min(item.margin for item in answerable), 4)

    # VALIDATE BEFORE WRITING. A measurement that cannot be a baseline must not
    # become one: an earlier version wrote first and validated on reload, which
    # left a negative value in the config after a failed run and made every
    # subsequent command fail at config load — including the ones needed to
    # diagnose it.
    for name, value in sorted(values.items()):
        if value <= 0:
            raise CalibrationError(
                f"refusing to commission a non-positive baseline ({name} = {value:+.4f}). "
                f"The floor does not discriminate on this corpus and model, so there is no "
                f"margin to defend. This is a corpus or embedding-model question, not a "
                f"threshold to set."
            )

    replacements = {
        _BODY_KEY: values["body_min"],
        _MIN_ANSWERABLE_KEY: values["min_answerable"],
        **{_PROBE_KEYS[number]: values[number] for number in UNANSWERABLE},
    }
    lines = target.read_text(encoding="utf-8").splitlines()
    seen: set[str] = set()
    for index, line in enumerate(lines):
        for key, value in replacements.items():
            if line.startswith(key):
                lines[index] = f"{key} {value}"
                seen.add(key)
    missing = sorted(set(replacements) - seen)
    if missing:
        raise CalibrationError(f"could not find these keys in {target}: {missing}")
    target.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    return values


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
    # Tier 2 runs against the DERIVED floor, so the probes are landed after the
    # artifact exists. The probes gate; they never derive.
    landings = await land_probes(artifact)

    if commissioning:
        if fake_embeddings_enabled():
            raise CalibrationError(
                "refusing to commission the guards against the stand-in embedder. The "
                "baselines they defend have to come from the model production actually uses."
            )
        values = commission(artifact, landings)
        reload_config()
        sys.stdout.write(report(artifact))
        sys.stdout.write(probe_report(landings))
        sys.stdout.write("COMMISSIONED from this measurement:\n")
        sys.stdout.write(
            f"  body_min                 {values['body_min']}   "
            f"({BODY_MIN_FRACTION:.0%} of the measured median gap "
            f"{artifact.body_separation:.4f})\n"
        )
        for number in UNANSWERABLE:
            sys.stdout.write(f"  unanswerable {number}         {values[number]}\n")
        sys.stdout.write(f"  min_answerable           {values['min_answerable']}\n")
    else:
        sys.stdout.write(report(artifact))
        sys.stdout.write(probe_report(landings))

    # Always enforced, including immediately after commissioning — so a
    # commissioning run that somehow produced baselines it could not itself
    # satisfy would fail rather than be recorded.
    check_floor_discrimination(landings)

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
