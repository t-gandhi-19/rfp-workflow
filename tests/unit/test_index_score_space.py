"""The vector index does not return a cosine, and the floor is a cosine floor.

Neo4j normalises a `cosine` index score into [0, 1] as `(1 + cos) / 2`.
Calibration measures its anchors with a plain dot product over unit vectors and
never touches the index, so the two numbers live in different spaces. Feeding
the index score into the calibrated mapping inflates every candidate.

This is the third instance of one fault — two numbers in different units
compared as though they were the same (amendment L: a calibrated-space floor
against raw cosine; the task-prefix bug: two differently conditioned embedding
spaces; this one). It is tested here at the arithmetic level and again against
the live index in the integration suite, because the conversion is only correct
if it matches what Neo4j actually does, and only Neo4j can settle that.
"""

from __future__ import annotations

import pytest

from src.graph.queries import index_score_to_cosine
from src.retrieval.calibration import CalibrationArtifact

#: The commissioned anchors, transcribed (amendment R). Enough of them to show
#: what the conversion changes about where a score lands.
BG_P50 = 0.4930148971911999
SAME_P50 = 0.7718081878021841
FLOOR = 0.5975344852115505


class TestTheConversion:
    @pytest.mark.parametrize(
        ("score", "cosine"),
        [
            (1.0, 1.0),  # identical vectors
            (0.5, 0.0),  # orthogonal
            (0.0, -1.0),  # opposed
            (0.75, 0.5),
            (0.8859040939010921, 0.7718081878021841),  # the same-subject median
            (0.74650744859560, 0.4930148971912),  # the background median
        ],
    )
    def test_it_inverts_neo4j_s_normalisation(self, score: float, cosine: float) -> None:
        assert index_score_to_cosine(score) == pytest.approx(cosine, abs=1e-12)

    def test_it_is_monotonic(self) -> None:
        """Ordering is preserved, which is why the fault was not obvious.

        The conversion is affine and increasing, so the RANKING within one
        question is unchanged by it. Only the absolute values move — and the
        floor is an absolute judgement, which is the half that broke.
        """
        scores = [0.1, 0.3, 0.5, 0.7, 0.9]
        converted = [index_score_to_cosine(s) for s in scores]
        assert converted == sorted(converted)

    def test_the_full_cosine_range_is_reachable(self) -> None:
        assert index_score_to_cosine(0.0) == -1.0
        assert index_score_to_cosine(1.0) == 1.0


class TestWhatTheFaultDidToTheFloor:
    """The measured consequence, pinned so the regression is legible."""

    @staticmethod
    def artifact() -> CalibrationArtifact:
        return CalibrationArtifact(
            geometry="query_x_document",
            population="query:any_family_member x document:indexed_originals",
            embed_model_tag="nomic-embed-text:v1.5",
            corpus_hash="a3d1eea3a24d6cb7",
            computed_at="2026-08-13T20:22:26+00:00",
            background_pair_count=5752,
            same_topic_pair_count=128,
            bg_p50=BG_P50,
            bg_p95=0.6192618897711936,
            bg_p99=0.6718287596779553,
            same_topic_p05=0.6473782454757817,
            same_topic_p50=SAME_P50,
            derived_floor=FLOOR,
        )

    def test_an_unconverted_background_score_saturates_the_mapping(self) -> None:
        """A perfectly average, unrelated pair looked like a perfect match.

        cos 0.4930 is the background MEDIAN — the middle of "unrelated". The
        index reports it as 0.7465, and calibrating that unconverted gives
        0.9092: near the top of the band, comfortably above the 0.5975 floor.
        Half of all unrelated pairs scored at least this well.
        """
        artifact = self.artifact()
        index_score = (1.0 + BG_P50) / 2.0
        assert artifact.calibrated(index_score) == pytest.approx(0.9092, abs=5e-5)
        assert artifact.calibrated(index_score) > artifact.derived_floor

    def test_conversion_puts_the_background_median_back_at_zero(self) -> None:
        """0 means "indistinguishable from an unrelated question", by definition."""
        artifact = self.artifact()
        index_score = (1.0 + BG_P50) / 2.0
        assert artifact.calibrated(index_score_to_cosine(index_score)) == pytest.approx(0.0)

    def test_the_same_subject_median_still_maps_to_one(self) -> None:
        artifact = self.artifact()
        index_score = (1.0 + SAME_P50) / 2.0
        assert artifact.calibrated(index_score_to_cosine(index_score)) == pytest.approx(1.0)

    def test_the_floor_rejects_background_only_after_conversion(self) -> None:
        """The property that matters: the floor has to reject something.

        Unconverted, a background-median pair cleared the floor — so the floor
        rejected nothing and every golden question MATCHED, including the three
        the corpus cannot answer.
        """
        artifact = self.artifact()
        index_score = (1.0 + BG_P50) / 2.0
        assert artifact.calibrated(index_score) >= artifact.derived_floor
        assert artifact.calibrated(index_score_to_cosine(index_score)) < artifact.derived_floor
