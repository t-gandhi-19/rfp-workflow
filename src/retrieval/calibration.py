"""Corpus-derived calibration for the relevance floor (D17, Step 2).

Raw cosine is not a comparable number. Its usable range depends on the embedding
model, the prompt convention, and how alike the corpus is to itself — so a floor
picked by eye is a floor that means something different on the next corpus.

Instead the anchors are measured:

* **background** — cross-topic pairs. What "unrelated, but both about cloud
  migration" scores.
* **same-topic** — pairs sharing a `topic_key`: paraphrases and supersession
  chains. What "genuinely the same question" scores.

Two properties of the measurement are load-bearing, and both were learned the
hard way:

**Geometry.** Every score the floor will ever judge is a *query*-embedded
question against a *document*-embedded one. Measuring document-against-document
gives a different distribution in a prefix-asymmetric model — it was what made
the background p99 appear to sit above the lowest real match. Calibration embeds
each corpus question both ways and only ever compares across prefixes.

**The topic split.** Supersession chains are near-duplicates. Left in the
background they inflate it, and the floor derived from them lands inside
valid-match territory.

The artifact is fail-closed: retrieval refuses to run if it is missing, or if
the embed model tag or corpus hash has moved, on the same principle as preflight.
"""

from __future__ import annotations

import hashlib
import json
import statistics
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from src.contracts.embedding import embedding_config
from src.contracts.thresholds import ScoringConfig, scoring_config

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURES = REPO_ROOT / "fixtures"
CALIBRATION_PATH = FIXTURES / "calibration.json"

#: The only geometry these statistics are valid in.
GEOMETRY = "query_x_document"


class CalibrationError(RuntimeError):
    """Calibration is missing or does not match the current state."""


class CalibrationArtifact(BaseModel):
    """Measured anchors, plus everything needed to know they still apply."""

    model_config = ConfigDict(extra="forbid")

    #: Recorded explicitly, because the numbers are meaningless in any other.
    geometry: str = Field(min_length=1)
    embed_model_tag: str = Field(min_length=1)
    #: Content hash of the corpus the statistics were measured over.
    corpus_hash: str = Field(min_length=1)
    computed_at: str = Field(min_length=1)

    background_pair_count: int = Field(ge=0)
    same_topic_pair_count: int = Field(ge=0)

    bg_p50: float
    bg_p95: float
    bg_p99: float
    same_topic_p05: float
    same_topic_p50: float

    #: THE MATCH FLOOR. Derived at calibrate time and stored, because it is a
    #: measurement rather than a setting (amendment O).
    #:
    #: It used to live in scoring.yaml as a value someone was expected to
    #: overwrite once the real statistics existed. That shipped as
    #: `match_floor_calibrated: 0.50` — a number retrieval consumed at runtime,
    #: known to be wrong, and wrong silently. A config key holding a derived
    #: value is a placeholder waiting to be forgotten; the config now holds the
    #: RULE and the artifact holds the result.
    derived_floor: float = Field(ge=0.0, le=1.0)

    @property
    def separation(self) -> float:
        """How far genuine matches sit above unrelated ones."""
        return self.same_topic_p05 - self.bg_p99

    def calibrated(self, cosine: float) -> float:
        """Map a raw cosine onto [0, 1].

        0 is "indistinguishable from an unrelated question", 1 is "as close as a
        genuine paraphrase". Both anchors are measured, so the number means the
        same thing across models and corpora.
        """
        span = self.same_topic_p50 - self.bg_p50
        if span <= 0:
            raise CalibrationError(
                f"degenerate calibration: same_topic_p50 ({self.same_topic_p50}) does not "
                f"exceed bg_p50 ({self.bg_p50}); the corpus cannot distinguish itself"
            )
        return min(1.0, max(0.0, (cosine - self.bg_p50) / span))


def derive_floor(artifact: CalibrationArtifact, *, config: ScoringConfig | None = None) -> float:
    """Place the floor inside the gap between the two anchors.

    Between "as high as unrelated content ever scores" and "as low as a genuine
    match ever scores" — the widest defensible gap, with `midpoint_weight`
    choosing where in it to sit.

    A pure function of the anchors and the rule, so the derivation can be tested
    without a model and re-checked against a stored artifact later.
    """
    resolved = config or scoring_config()
    rule = resolved.calibration.floor_derivation
    low = artifact.calibrated(getattr(artifact, rule.background_anchor))
    high = artifact.calibrated(getattr(artifact, rule.same_topic_anchor))
    return low + rule.midpoint_weight * (high - low)


def corpus_hash(pairs: list[dict[str, Any]]) -> str:
    """Stable hash over the question texts and families the stats came from.

    The FAMILY is hashed, not the topic key. The family is what partitions the
    pairs into background and same-subject, so a regrouping that left every
    question's text untouched would still move both distributions — and an
    artifact whose hash did not notice would keep asserting statistics that no
    longer describe the corpus.
    """
    material = json.dumps(
        sorted((p["question_id"], p["topic_family"], p["question"]) for p in pairs),
        sort_keys=True,
    )
    return hashlib.sha256(material.encode()).hexdigest()[:16]


def cosine(a: list[float], b: list[float]) -> float:
    """Both sides are unit vectors from the embedding model, so this is the dot."""
    return sum(x * y for x, y in zip(a, b, strict=True))


def compute(
    *,
    query_vectors: dict[str, list[float]],
    document_vectors: dict[str, list[float]],
    topic_by_question: dict[str, str],
    embed_model_tag: str,
    corpus_content_hash: str,
    computed_at: str,
    config: ScoringConfig | None = None,
) -> CalibrationArtifact:
    """Measure the anchors in query-versus-document geometry."""
    resolved = config or scoring_config()

    background: list[float] = []
    same_topic: list[float] = []
    for query_id, query_vector in sorted(query_vectors.items()):
        for document_id, document_vector in sorted(document_vectors.items()):
            if query_id == document_id:
                continue  # a question against itself says nothing
            score = cosine(query_vector, document_vector)
            if topic_by_question[query_id] == topic_by_question[document_id]:
                same_topic.append(score)
            else:
                background.append(score)

    minimum = resolved.calibration.min_background_pairs
    if len(background) < minimum:
        raise CalibrationError(
            f"only {len(background)} cross-topic pairs; at least {minimum} are needed for "
            "stable percentiles. A corpus this small would give noise, not calibration."
        )

    # The same-subject anchor is the fragile one. It is smaller than the
    # background by two orders of magnitude, and a p05 over a handful of values
    # is that handful's minimum wearing a percentile's name.
    #
    # This guard is not hypothetical. The corpus once grouped by `topic_key`,
    # which is unique per record — so every group was a singleton and this
    # population was EMPTY. Requiring merely "not empty" would have accepted the
    # next version of that mistake: four supersession chains whose question text
    # is byte-identical, giving eight pairs that measure exact duplicates rather
    # than genuine paraphrases, and a floor derived from them sits above the
    # real matches it is supposed to admit.
    same_topic_minimum = resolved.calibration.min_same_topic_pairs
    if len(same_topic) < same_topic_minimum:
        raise CalibrationError(
            f"only {len(same_topic)} same-subject pairs; at least {same_topic_minimum} are "
            "needed. The corpus does not contain enough differently-worded askings of the "
            "same question to establish what a genuine match scores, so any floor derived "
            "from it would be an artifact of the few pairs that exist."
        )

    background.sort()
    same_topic.sort()

    def percentile(values: list[float], fraction: float) -> float:
        index = min(len(values) - 1, max(0, round(fraction * (len(values) - 1))))
        return values[index]

    # Built in two steps: the floor is derived FROM the anchors, so the anchors
    # have to exist before it can be. `derived_floor=0.0` is a placeholder that
    # lives for one statement and never leaves this function.
    artifact = CalibrationArtifact(
        geometry=GEOMETRY,
        embed_model_tag=embed_model_tag,
        corpus_hash=corpus_content_hash,
        computed_at=computed_at,
        background_pair_count=len(background),
        same_topic_pair_count=len(same_topic),
        bg_p50=statistics.median(background),
        bg_p95=percentile(background, 0.95),
        bg_p99=percentile(background, 0.99),
        same_topic_p05=percentile(same_topic, 0.05),
        same_topic_p50=statistics.median(same_topic),
        derived_floor=0.0,
    )
    artifact = artifact.model_copy(
        update={"derived_floor": derive_floor(artifact, config=resolved)}
    )

    if artifact.separation < resolved.calibration.min_separation_raw:
        raise CalibrationError(
            f"separation {artifact.separation:.4f} is below the required "
            f"{resolved.calibration.min_separation_raw}: genuine matches (p05 "
            f"{artifact.same_topic_p05:.4f}) barely clear unrelated ones (p99 "
            f"{artifact.bg_p99:.4f}). Retrieval cannot be trusted in this state."
        )
    return artifact


def save(artifact: CalibrationArtifact, path: Path | None = None) -> None:
    target = path or CALIBRATION_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(artifact.model_dump(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def load(
    path: Path | None = None,
    *,
    expected_model_tag: str | None = None,
    expected_corpus_hash: str | None = None,
) -> CalibrationArtifact:
    """Load and verify. Fail-closed, like preflight.

    A stale artifact is worse than none: the floor would be derived from a model
    or corpus that no longer exists, and nothing would say so.
    """
    target = path or CALIBRATION_PATH
    if not target.is_file():
        raise CalibrationError(
            f"calibration artifact missing at {target}. Run: make calibrate\n"
            "Retrieval will not run without it — the match floor is derived from it."
        )
    artifact = CalibrationArtifact.model_validate_json(target.read_text(encoding="utf-8"))

    if artifact.geometry != GEOMETRY:
        raise CalibrationError(
            f"calibration was measured in '{artifact.geometry}' geometry, but scores are "
            f"judged in '{GEOMETRY}'. These are different distributions."
        )
    if expected_model_tag and artifact.embed_model_tag != expected_model_tag:
        raise CalibrationError(
            f"calibration was measured against '{artifact.embed_model_tag}' but the pinned "
            f"model is '{expected_model_tag}'. Run: make reembed && make calibrate"
        )
    if expected_corpus_hash and artifact.corpus_hash != expected_corpus_hash:
        raise CalibrationError(
            f"calibration was measured over corpus {artifact.corpus_hash} but the corpus is "
            f"now {expected_corpus_hash}. Run: make calibrate"
        )
    return artifact


def calibration_corpus() -> list[dict[str, Any]]:
    """Every question the statistics are measured over: corpus plus paraphrases.

    Both populations are needed and they play different parts. The 40 corpus
    questions supply the background — what "unrelated, but both about cloud
    migration" scores. The paraphrases supply the same-subject anchor, which is
    the only reason they exist.
    """
    rows: list[dict[str, Any]] = []
    with (FIXTURES / "qa_pairs.json").open(encoding="utf-8") as handle:
        for pair in json.load(handle):
            rows.append(
                {
                    "question_id": pair["question_id"],
                    "topic_family": pair["topic_family"],
                    "question": pair["question"],
                }
            )
    with (FIXTURES / "question_paraphrases.json").open(encoding="utf-8") as handle:
        for row in json.load(handle):
            rows.append(
                {
                    "question_id": row["question_id"],
                    "topic_family": row["topic_family"],
                    "question": row["question"],
                }
            )
    return sorted(rows, key=lambda row: row["question_id"])


def load_for_current_corpus(path: Path | None = None) -> CalibrationArtifact:
    """Load the artifact and verify it still describes the corpus in the repo.

    The single entry point retrieval uses. Both bindings are checked here rather
    than left to the caller, because a caller that forgets one gets an artifact
    that looks valid and is not.
    """
    return load(
        path,
        expected_model_tag=embedding_config().model.tag,
        expected_corpus_hash=corpus_hash(calibration_corpus()),
    )
