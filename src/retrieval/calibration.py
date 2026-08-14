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

#: The only POPULATION these statistics are valid over (D18).
#:
#: Geometry says which PREFIX each side carries. Population says which TEXTS are
#: eligible to be on each side. Both are properties of the comparison retrieval
#: actually performs, and getting either wrong measures a distribution the
#: system will never see:
#:
#:   query side     any family member — paraphrases model an unseen phrasing
#:                  arriving in a new RFP, originals model a near-verbatim one.
#:   document side  INDEXED ORIGINALS ONLY.
#:
#: The document side follows from amendment P rather than from taste. Paraphrases
#: are excluded from the vector index because they carry no answer of their own,
#: so no paraphrase can ever appear on the document side of a real comparison.
#: Measuring paraphrase-against-paraphrase pairs therefore measures something the
#: production system is structurally incapable of doing.
#:
#: That was not academic. Under the old all-pairs population the same-subject p05
#: was 0.6863 and was set almost entirely by para->para pairs — two of our own
#: rewordings compared to each other, neither of which is a question the corpus
#: contains. They dragged the tail into the background's and failed the guard.
POPULATION = "query:any_family_member x document:indexed_originals"


class CalibrationError(RuntimeError):
    """Calibration is missing or does not match the current state."""


class CalibrationArtifact(BaseModel):
    """Measured anchors, plus everything needed to know they still apply."""

    model_config = ConfigDict(extra="forbid")

    #: Recorded explicitly, because the numbers are meaningless in any other.
    geometry: str = Field(min_length=1)
    #: Which texts were eligible on each side (D18). Recorded for the same reason
    #: geometry is: the anchors describe one population and no other, and a
    #: reader comparing two artifacts needs to know they measured the same thing.
    population: str = Field(min_length=1)
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
        """Tail separation. REPORTED, NOT GATED (D19).

        How far the weakest genuine match clears the strongest unrelated one.
        Genuinely informative about how hard the paraphrase set is, and it is
        printed and stored for that reason — but it gated tier 2 until it proved
        it could not:

        * **Open-ended downward.** The same-subject lower tail is set by the
          hardest legitimate paraphrases. Where that boundary sits is an
          authorship judgement with no crisp edge.
        * **Non-convergent under data addition.** Every paraphrase added to firm
          up the estimate is also a new candidate for "worst match". Going from
          two paraphrases per family to three moved this from +0.0237 to
          -0.0245 while every operational margin held or improved.
        * **Correlated observations.** Version families contribute pairs against
          two byte-identical documents, so tail observations double-count.

        A statistic that is open-ended, non-convergent and correlated is not
        commissioning-grade at any retention factor.
        """
        return self.same_topic_p05 - self.bg_p99

    @property
    def body_separation(self) -> float:
        """Median separation: how far the two distributions sit apart in bulk.

        The tier-1 collapse detector watches this. It is the statistic that
        moves when something breaks wholesale — the missing-task-prefix bug
        compressed the whole band, and a median gap notices that immediately,
        where a tail percentile can be dragged around by a handful of pairs.
        """
        return self.same_topic_p50 - self.bg_p50

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
    indexed_ids: set[str],
    embed_model_tag: str,
    corpus_content_hash: str,
    computed_at: str,
    config: ScoringConfig | None = None,
) -> CalibrationArtifact:
    """Measure the anchors in the geometry AND population retrieval uses (D18).

    `indexed_ids` is the document side, and it is not a convenience parameter —
    it is the population rule made explicit. Only questions that are in the
    vector index can ever be on the document side of a real comparison, and
    amendment P keeps paraphrases out of that index because they carry no answer
    of their own. So a paraphrase may be a QUERY (it models an unseen phrasing
    arriving in a new RFP) and may never be a DOCUMENT.

    Both distributions are recomputed under this rule. They are never mixed with
    statistics gathered under another one: an old background measured over all
    pairs and a new same-subject measured over indexed documents would be two
    different experiments reported as one.
    """
    resolved = config or scoring_config()

    unknown = sorted(indexed_ids - set(document_vectors))
    if unknown:
        raise CalibrationError(
            f"indexed_ids names {len(unknown)} question(s) with no document vector: "
            f"{unknown[:5]}. The document side must be embeddable."
        )

    background: list[float] = []
    same_topic: list[float] = []
    for query_id, query_vector in sorted(query_vectors.items()):
        for document_id in sorted(indexed_ids):
            if query_id == document_id:
                continue  # a question against itself says nothing
            score = cosine(query_vector, document_vectors[document_id])
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
        population=POPULATION,
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

    # Tier 1 only. The erosion ratchet is a SEPARATE call, because commissioning
    # it is the one moment its value is set from data — and a measurement
    # function that refused to produce the very number it is about to be
    # measured against could never be commissioned at all.
    check_collapse(artifact, config=resolved)
    return artifact


def check_collapse(artifact: CalibrationArtifact, *, config: ScoringConfig | None = None) -> None:
    """Tier 1. Median gap against an absolute floor.

    This is the statistic that moves when something breaks WHOLESALE. When task
    prefixes were missing the entire similarity band compressed, and a median gap
    notices that at once, where a tail percentile can be dragged around by a
    handful of pairs.

    Absolute rather than a ratchet, because collapse has a recognisable scale:
    two distributions whose medians have merged are not a marginal corpus, they
    are a systemic fault.
    """
    resolved = config or scoring_config()
    guards = resolved.calibration.separation_guards
    if artifact.body_separation < guards.body_min:
        raise CalibrationError(
            f"COLLAPSE: median separation {artifact.body_separation:.4f} is below "
            f"{guards.body_min} — the two distributions have merged in bulk, not just in "
            f"the tails (same_p50 {artifact.same_topic_p50:.4f} vs bg_p50 "
            f"{artifact.bg_p50:.4f}). This is the signature of a systemic fault such as a "
            f"missing task prefix or a mismatched model, not of a marginal corpus."
        )


#: The three golden questions the corpus deliberately cannot answer.
UNANSWERABLE = ("2.7", "2.8", "3.5")
#: Deliberately NOT gated. 3.4 is refused on legal grounds regardless of
#: retrieval; 4.2 is owned by Phase 4's pricing block. Reported only.
BAITS = ("3.4", "4.2")


class ProbeLanding(BaseModel):
    """Where one golden question landed against the derived floor."""

    model_config = ConfigDict(extra="forbid")

    number: str = Field(min_length=1)
    kind: str = Field(min_length=1)
    raw: float
    calibrated: float
    #: Signed so that POSITIVE is always correct: distance below the floor for
    #: an unanswerable, above it for an answerable.
    margin: float


def check_floor_discrimination(
    landings: list[ProbeLanding], *, config: ScoringConfig | None = None
) -> None:
    """Tier 2 (D19). Does the derived floor still separate answerable from not?

    THE PROBES GATE; THEY NEVER DERIVE. The floor's value comes only from
    corpus-internal statistics. This check is a canary verifying that the
    derived floor still discriminates — the zero-tolerance eval core embedded
    into calibration, so that a corpus or model change run through
    `make calibrate` alone cannot silently ship a non-discriminating floor.

    Gated: each unanswerable must sit below the floor, and the weakest
    answerable above it, by at least `retention` of its commissioned margin.
    Advisory: the two baits are reported and never gated.

    Refuses outright when uncommissioned. A ratchet with no baseline is not a
    lenient guard, it is an absent one.

    WHAT THIS GUARD IS AND IS NOT. It protects the FUTURE: it detects change
    against a commissioned baseline. It is not a statement that retrieval is
    good. The absolute quality standard is the zero-tolerance retrieval evals on
    real embeddings, which bind regardless of guard parameters. Loosening a
    guard can never make retrieval acceptable; it can only stop it reporting.
    """
    resolved = config or scoring_config()
    rule = resolved.calibration.separation_guards.floor_discrimination
    baseline = rule.commissioned

    if not baseline.commissioned():
        raise CalibrationError(
            "the floor-discrimination ratchet has never been commissioned. Its baselines "
            "must be MEASURED from real embeddings against the derived floor, once, and "
            "recorded.\nRun: make calibrate-commission"
        )

    by_number = {landing.number: landing for landing in landings}
    commissioned_unanswerable = {
        "2.7": baseline.unanswerable_2_7,
        "2.8": baseline.unanswerable_2_8,
        "3.5": baseline.unanswerable_3_5,
    }

    for number in UNANSWERABLE:
        landing = by_number.get(number)
        if landing is None:
            raise CalibrationError(f"probe {number} was not measured; cannot gate on it")
        commissioned = commissioned_unanswerable[number]
        if commissioned is None:  # pragma: no cover - commissioned() rules this out
            raise CalibrationError(f"probe {number} has no commissioned baseline")
        required = commissioned * rule.retention
        if landing.margin < required:
            raise CalibrationError(
                f"FLOOR NO LONGER DISCRIMINATES: unanswerable {number} sits "
                f"{landing.margin:+.4f} below the floor, under the required {required:.4f} "
                f"({rule.retention:.0%} of the commissioned {commissioned:.4f}). A question "
                f"the corpus cannot answer is approaching the match floor, so retrieval is "
                f"drifting towards answering it. This is a corpus or embedding-model "
                f"question, not a threshold to adjust."
            )

    answerable = [landing for landing in landings if landing.kind == "answerable"]
    if not answerable:
        raise CalibrationError("no answerable probes were measured; cannot gate on them")
    weakest = min(answerable, key=lambda landing: landing.margin)
    if baseline.min_answerable is None:  # pragma: no cover - commissioned() rules this out
        raise CalibrationError("no commissioned baseline for the weakest answerable")
    required_answerable = baseline.min_answerable * rule.retention
    if weakest.margin < required_answerable:
        raise CalibrationError(
            f"FLOOR NO LONGER DISCRIMINATES: answerable {weakest.number} clears the floor by "
            f"only {weakest.margin:+.4f}, under the required {required_answerable:.4f} "
            f"({rule.retention:.0%} of the commissioned {baseline.min_answerable:.4f}). A "
            f"question the corpus CAN answer is approaching the floor from above, so "
            f"retrieval is drifting towards refusing it."
        )


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
    """Every question the statistics are measured over, and which side it may take.

    `indexed` is the population rule in data form (D18). Originals are in the
    vector index and may appear on either side. Paraphrases are excluded from the
    index by amendment P — they carry no answer of their own — so they may be a
    QUERY and never a DOCUMENT.
    """
    rows: list[dict[str, Any]] = []
    with (FIXTURES / "qa_pairs.json").open(encoding="utf-8") as handle:
        for pair in json.load(handle):
            rows.append(
                {
                    "question_id": pair["question_id"],
                    "topic_family": pair["topic_family"],
                    "question": pair["question"],
                    "indexed": True,
                }
            )
    with (FIXTURES / "question_paraphrases.json").open(encoding="utf-8") as handle:
        for row in json.load(handle):
            rows.append(
                {
                    "question_id": row["question_id"],
                    "topic_family": row["topic_family"],
                    "question": row["question"],
                    "indexed": False,
                }
            )
    return sorted(rows, key=lambda row: row["question_id"])


def indexed_ids(rows: list[dict[str, Any]]) -> set[str]:
    """The document side: questions that are actually in the vector index."""
    return {row["question_id"] for row in rows if row["indexed"]}


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
