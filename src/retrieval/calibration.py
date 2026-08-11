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

from src.contracts.thresholds import ScoringConfig, scoring_config

REPO_ROOT = Path(__file__).resolve().parents[2]
CALIBRATION_PATH = REPO_ROOT / "fixtures" / "calibration.json"

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

    def derived_floor(self) -> float:
        """The match floor, in calibrated space.

        Midpoint between "as high as unrelated content ever scores" (bg_p99) and
        "as low as a genuine match ever scores" (same_topic_p05) — the widest gap
        available, placed in the middle of it.
        """
        return (self.calibrated(self.bg_p99) + self.calibrated(self.same_topic_p05)) / 2


def corpus_hash(pairs: list[dict[str, Any]]) -> str:
    """Stable hash over the question texts and topic keys the stats came from."""
    material = json.dumps(
        sorted((p["question_id"], p["topic_key"], p["question"]) for p in pairs),
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
    if not same_topic:
        raise CalibrationError(
            "no same-topic pairs — the corpus has no paraphrases or supersession chains, "
            "so there is nothing to anchor 'genuinely the same question' against"
        )

    background.sort()
    same_topic.sort()

    def percentile(values: list[float], fraction: float) -> float:
        index = min(len(values) - 1, max(0, round(fraction * (len(values) - 1))))
        return values[index]

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
    )

    if artifact.separation < resolved.calibration.min_separation:
        raise CalibrationError(
            f"separation {artifact.separation:.4f} is below the required "
            f"{resolved.calibration.min_separation}: genuine matches (p05 "
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
