"""Calibration: the measurement the match floor is derived from.

This module had no behavioural tests until amendment J, which is how it came to
be shipped with a `topic_key` split that produced zero same-subject pairs on the
actual corpus — the function could not have succeeded, and nothing said so.
Every guard below exists because its absence produced a floor that meant
something other than what it claimed.

The geometry is the load-bearing part. Every score the floor will ever judge is
a QUERY-embedded question against a DOCUMENT-embedded one. Measuring
document-against-document gives a different distribution in a prefix-asymmetric
model, and it was what made the background p99 appear to sit above the lowest
real match.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pytest

from src.contracts.thresholds import scoring_config
from src.retrieval.calibration import (
    GEOMETRY,
    CalibrationArtifact,
    CalibrationError,
    calibration_corpus,
    compute,
    corpus_hash,
    cosine,
    load,
    save,
)

CONFIG = scoring_config()
COMPUTED_AT = "2026-08-13T00:00:00+00:00"
MODEL_TAG = "test-model:v1"


def artifact(**overrides: Any) -> CalibrationArtifact:
    """A well-formed artifact. Fields the test cares about are overridden."""
    defaults: dict[str, Any] = {
        "geometry": GEOMETRY,
        "embed_model_tag": MODEL_TAG,
        "corpus_hash": "0000000000000000",
        "computed_at": COMPUTED_AT,
        "background_pair_count": 12192,
        "same_topic_pair_count": 240,
        "bg_p50": 0.60,
        "bg_p95": 0.72,
        "bg_p99": 0.75,
        "same_topic_p05": 0.85,
        "same_topic_p50": 0.90,
    }
    return CalibrationArtifact(**{**defaults, **overrides})


def unit(seed: int, dimensions: int) -> list[float]:
    """A deterministic unit vector, varied by `seed`."""
    values = [math.sin(seed * 12.9898 + index * 78.233) for index in range(dimensions)]
    norm = math.sqrt(sum(v * v for v in values))
    return [v / norm for v in values]


def synthetic_corpus(
    families: int, per_family: int, *, tightness: float = 0.05
) -> tuple[dict[str, list[float]], dict[str, list[float]], dict[str, str]]:
    """Vectors where same-family questions are close and cross-family are not.

    Each family sits on its own basis vector, so cross-family similarity is 0 by
    construction and the two populations are cleanly separable. Members are then
    pulled off their anchor by `tightness`; raising it collapses the separation,
    which is how the separation guard is driven to fire.

    Built by hand rather than by embedding, so the guards can be pushed to their
    boundaries without needing a model.
    """
    dimensions = families + 4
    queries: dict[str, list[float]] = {}
    documents: dict[str, list[float]] = {}
    topics: dict[str, str] = {}
    for family in range(families):
        anchor = [0.0] * dimensions
        anchor[family] = 1.0
        for member in range(per_family):
            key = f"q-{family}-{member}"
            offset = unit(family * 97 + member * 13 + 7, dimensions)
            vector = [a + tightness * (member + 1) * o for a, o in zip(anchor, offset, strict=True)]
            norm = math.sqrt(sum(v * v for v in vector))
            normalised = [v / norm for v in vector]
            queries[key] = normalised
            documents[key] = normalised
            topics[key] = f"family-{family}"
    return queries, documents, topics


def measure(families: int = 40, per_family: int = 3, **kwargs: Any) -> CalibrationArtifact:
    queries, documents, topics = synthetic_corpus(families, per_family, **kwargs)
    return compute(
        query_vectors=queries,
        document_vectors=documents,
        topic_by_question=topics,
        embed_model_tag=MODEL_TAG,
        corpus_content_hash="deadbeefdeadbeef",
        computed_at=COMPUTED_AT,
    )


class TestCosine:
    def test_identical_unit_vectors_score_one(self) -> None:
        vector = unit(3, 8)
        assert cosine(vector, vector) == pytest.approx(1.0)

    def test_length_mismatch_is_refused(self) -> None:
        """Silent truncation would compare two different things and say nothing."""
        with pytest.raises(ValueError):
            cosine([1.0, 0.0], [1.0, 0.0, 0.0])


class TestCalibratedMapping:
    def test_the_background_median_maps_to_zero(self) -> None:
        """0 means 'indistinguishable from an unrelated question'."""
        assert artifact().calibrated(0.60) == pytest.approx(0.0)

    def test_the_same_topic_median_maps_to_one(self) -> None:
        assert artifact().calibrated(0.90) == pytest.approx(1.0)

    def test_the_midpoint_maps_to_a_half(self) -> None:
        assert artifact().calibrated(0.75) == pytest.approx(0.5)

    def test_it_is_clamped_at_both_ends(self) -> None:
        """A cosine outside the measured band is not a score outside [0, 1]."""
        assert artifact().calibrated(0.10) == 0.0
        assert artifact().calibrated(0.99) == 1.0

    def test_it_is_monotonic(self) -> None:
        subject = artifact()
        values = [subject.calibrated(raw) for raw in (0.60, 0.65, 0.70, 0.75, 0.80, 0.85)]
        assert values == sorted(values)

    def test_a_degenerate_corpus_is_refused(self) -> None:
        """If same-topic does not exceed background, there is nothing to map onto.

        Returning something anyway would produce a number in [0, 1] that carried
        no information, and every downstream comparison would look normal.
        """
        with pytest.raises(CalibrationError, match="degenerate"):
            artifact(bg_p50=0.90, same_topic_p50=0.90).calibrated(0.95)


class TestDerivedFloor:
    def test_it_is_the_midpoint_of_the_gap(self) -> None:
        """calibrated(0.75) = 0.5, calibrated(0.85) = 0.8333; midpoint 0.6667."""
        assert artifact().derived_floor() == pytest.approx(0.666667, abs=1e-6)

    def test_it_sits_between_the_two_anchors(self) -> None:
        subject = artifact()
        assert (
            subject.calibrated(subject.bg_p99)
            < subject.derived_floor()
            < subject.calibrated(subject.same_topic_p05)
        )

    def test_a_wider_gap_does_not_move_it_outside_the_anchors(self) -> None:
        subject = artifact(bg_p99=0.65, same_topic_p05=0.88)
        assert (
            subject.calibrated(subject.bg_p99)
            <= subject.derived_floor()
            <= subject.calibrated(subject.same_topic_p05)
        )

    def test_separation_is_the_raw_gap(self) -> None:
        assert artifact().separation == pytest.approx(0.85 - 0.75)


class TestComputeGeometry:
    def test_a_question_is_never_compared_with_itself(self) -> None:
        """Self-similarity is 1.0 by construction and says nothing about anything."""
        subject = measure(families=40, per_family=3)
        total = 120 * 119
        assert subject.background_pair_count + subject.same_topic_pair_count == total, (
            "every ordered pair is classified exactly once, and no self-pair is counted"
        )

    def test_the_split_matches_the_family_structure(self) -> None:
        """40 families of 3: same-topic = 40*3*2 = 240, background = the rest."""
        subject = measure(families=40, per_family=3)
        assert subject.same_topic_pair_count == 240
        assert subject.background_pair_count == 120 * 119 - 240

    def test_same_topic_scores_above_background(self) -> None:
        subject = measure()
        assert subject.same_topic_p05 > subject.bg_p99
        assert subject.same_topic_p50 > subject.bg_p50

    def test_the_geometry_is_recorded_on_the_artifact(self) -> None:
        """The numbers are meaningless in any other, so it is not left implicit."""
        assert measure().geometry == GEOMETRY


class TestGuards:
    def test_too_few_background_pairs_is_refused(self) -> None:
        """Percentiles over a handful of values are that handful, renamed."""
        with pytest.raises(CalibrationError, match="cross-topic pairs"):
            measure(families=5, per_family=3)

    def test_too_few_same_topic_pairs_is_refused(self) -> None:
        """The guard that the shipped corpus would once have failed.

        Grouping by a unique key gave every family one member and this
        population zero pairs; four byte-identical supersession chains would
        have given eight, which measures duplicates rather than paraphrases.
        """
        with pytest.raises(CalibrationError, match="same-subject pairs"):
            measure(families=120, per_family=1)

    def test_the_same_topic_minimum_is_the_configured_one(self) -> None:
        assert CONFIG.calibration.min_same_topic_pairs == 100

    def test_insufficient_separation_is_refused(self) -> None:
        """Loose families put genuine matches on top of unrelated ones.

        Failing loudly at calibration time is the point: the alternative is
        retrieval failing quietly, for the rest of the corpus's life.
        """
        with pytest.raises(CalibrationError, match="separation"):
            measure(tightness=8.0)

    def test_the_separation_guard_reports_both_anchors(self) -> None:
        """The message has to say what was measured, not just that it failed."""
        with pytest.raises(CalibrationError) as caught:
            measure(tightness=8.0)
        message = str(caught.value)
        assert "p05" in message and "p99" in message


class TestCorpusHash:
    def test_it_is_stable_across_orderings(self) -> None:
        rows = [
            {"question_id": "b", "topic_family": "f1", "question": "second"},
            {"question_id": "a", "topic_family": "f1", "question": "first"},
        ]
        assert corpus_hash(rows) == corpus_hash(list(reversed(rows)))

    def test_changed_question_text_changes_it(self) -> None:
        base = [{"question_id": "a", "topic_family": "f1", "question": "first"}]
        moved = [{"question_id": "a", "topic_family": "f1", "question": "FIRST"}]
        assert corpus_hash(base) != corpus_hash(moved)

    def test_regrouping_alone_changes_it(self) -> None:
        """The family partitions the pairs, so moving it moves both distributions.

        A hash over text alone would keep asserting statistics that no longer
        describe the corpus they were measured on.
        """
        base = [{"question_id": "a", "topic_family": "f1", "question": "first"}]
        regrouped = [{"question_id": "a", "topic_family": "f2", "question": "first"}]
        assert corpus_hash(base) != corpus_hash(regrouped)


class TestFailClosedLoading:
    def test_a_missing_artifact_names_the_fix(self, tmp_path: Path) -> None:
        with pytest.raises(CalibrationError, match="make calibrate"):
            load(tmp_path / "absent.json")

    def test_a_round_trip_preserves_every_field(self, tmp_path: Path) -> None:
        target = tmp_path / "calibration.json"
        original = artifact()
        save(original, target)
        assert load(target) == original

    def test_a_different_embed_model_is_refused(self, tmp_path: Path) -> None:
        """Anchors measured against another model describe another geometry."""
        target = tmp_path / "calibration.json"
        save(artifact(), target)
        with pytest.raises(CalibrationError, match="reembed"):
            load(target, expected_model_tag="nomic-embed-text:v1.5")

    def test_a_changed_corpus_is_refused(self, tmp_path: Path) -> None:
        target = tmp_path / "calibration.json"
        save(artifact(), target)
        with pytest.raises(CalibrationError, match="make calibrate"):
            load(target, expected_corpus_hash="ffffffffffffffff")

    def test_the_wrong_geometry_is_refused(self, tmp_path: Path) -> None:
        """A document-against-document measurement is a different distribution."""
        target = tmp_path / "calibration.json"
        target.write_text(
            json.dumps({**artifact().model_dump(), "geometry": "document_x_document"}),
            encoding="utf-8",
        )
        with pytest.raises(CalibrationError, match="geometry"):
            load(target)

    def test_matching_bindings_load_cleanly(self, tmp_path: Path) -> None:
        target = tmp_path / "calibration.json"
        subject = artifact()
        save(subject, target)
        loaded = load(
            target,
            expected_model_tag=MODEL_TAG,
            expected_corpus_hash=subject.corpus_hash,
        )
        assert loaded.derived_floor() == pytest.approx(subject.derived_floor())

    def test_an_unknown_field_is_refused(self, tmp_path: Path) -> None:
        """`extra="forbid"`: a hand-edited artifact fails rather than half-loads."""
        target = tmp_path / "calibration.json"
        target.write_text(
            json.dumps({**artifact().model_dump(), "hand_tuned_floor": 0.4}), encoding="utf-8"
        )
        with pytest.raises(Exception, match="hand_tuned_floor"):
            load(target)


class TestCalibrationCorpus:
    """What the statistics are actually measured over."""

    def test_it_covers_the_corpus_and_the_paraphrases(self) -> None:
        rows = calibration_corpus()
        assert len(rows) == 112, "40 corpus questions + 72 paraphrases"

    def test_every_row_carries_a_family(self) -> None:
        assert all(row["topic_family"] for row in calibration_corpus())

    def test_question_ids_are_unique(self) -> None:
        ids = [row["question_id"] for row in calibration_corpus()]
        assert len(set(ids)) == len(ids)

    def test_it_clears_both_minimum_pair_guards(self) -> None:
        """The shipped corpus must be able to calibrate itself.

        Computed from the family structure rather than by embedding, so this
        stays a fast unit test and still fails the moment the corpus shrinks
        past what the guards permit.
        """
        rows = calibration_corpus()
        sizes: dict[str, int] = {}
        for row in rows:
            sizes[row["topic_family"]] = sizes.get(row["topic_family"], 0) + 1
        same = sum(size * (size - 1) for size in sizes.values())
        total = len(rows) * (len(rows) - 1)
        assert same >= CONFIG.calibration.min_same_topic_pairs
        assert total - same >= CONFIG.calibration.min_background_pairs
