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

from src.contracts.thresholds import ScoringConfig, scoring_config
from src.retrieval.calibration import (
    GEOMETRY,
    POPULATION,
    CalibrationArtifact,
    CalibrationError,
    ProbeLanding,
    calibration_corpus,
    check_collapse,
    check_floor_discrimination,
    compute,
    corpus_hash,
    cosine,
    derive_floor,
    indexed_ids,
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
        "population": POPULATION,
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
        "derived_floor": 0.666667,
    }
    return CalibrationArtifact(**{**defaults, **overrides})


def ratcheted(**overrides: float | None) -> ScoringConfig:
    """CONFIG with given commissioned probe baselines, to drive the tier-2 gate."""
    data = CONFIG.model_dump()
    commissioned = data["calibration"]["separation_guards"]["floor_discrimination"]["commissioned"]
    commissioned.update(overrides)
    return ScoringConfig.model_validate(data)


def uncommissioned() -> ScoringConfig:
    return ratcheted(
        unanswerable_2_7=None,
        unanswerable_2_8=None,
        unanswerable_3_5=None,
        min_answerable=None,
    )


def landing(number: str, kind: str, margin: float) -> ProbeLanding:
    """A probe landing with only the fields the gate reads set meaningfully."""
    return ProbeLanding(number=number, kind=kind, raw=0.7, calibrated=0.5, margin=margin)


def healthy_landings(**margins: float) -> list[ProbeLanding]:
    """Three unanswerables and one answerable, all comfortably correct."""
    defaults = {"2.7": 0.30, "2.8": 0.28, "3.5": 0.17, "answerable": 0.24}
    defaults.update(margins)
    return [
        landing("2.7", "unanswerable", defaults["2.7"]),
        landing("2.8", "unanswerable", defaults["2.8"]),
        landing("3.5", "unanswerable", defaults["3.5"]),
        landing("1.4", "answerable", defaults["answerable"]),
    ]


def weighted(midpoint_weight: float) -> ScoringConfig:
    """CONFIG with a different midpoint weighting, to drive the rule's extremes."""
    data = CONFIG.model_dump()
    data["calibration"]["floor_derivation"]["midpoint_weight"] = midpoint_weight
    return ScoringConfig.model_validate(data)


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


def measure(
    families: int = 40,
    per_family: int = 3,
    *,
    indexed: set[str] | None = None,
    **kwargs: Any,
) -> CalibrationArtifact:
    queries, documents, topics = synthetic_corpus(families, per_family, **kwargs)
    return compute(
        query_vectors=queries,
        document_vectors=documents,
        topic_by_question=topics,
        indexed_ids=indexed if indexed is not None else set(documents),
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
    """Amendment O: the floor is derived at calibrate time and STORED.

    It used to live in scoring.yaml as a value someone was expected to overwrite
    once real statistics existed — which shipped as a known-wrong 0.50 that
    retrieval consumed at runtime. Config now holds the rule; the artifact holds
    the result, next to the anchors that justify it.
    """

    def test_it_is_the_midpoint_of_the_gap(self) -> None:
        """calibrated(0.75) = 0.5, calibrated(0.85) = 0.8333; midpoint 0.6667."""
        assert derive_floor(artifact()) == pytest.approx(0.666667, abs=1e-6)

    def test_it_sits_between_the_two_anchors(self) -> None:
        subject = artifact()
        assert (
            subject.calibrated(subject.bg_p99)
            < derive_floor(subject)
            < subject.calibrated(subject.same_topic_p05)
        )

    def test_a_wider_gap_does_not_move_it_outside_the_anchors(self) -> None:
        subject = artifact(bg_p99=0.65, same_topic_p05=0.88)
        assert (
            subject.calibrated(subject.bg_p99)
            <= derive_floor(subject)
            <= subject.calibrated(subject.same_topic_p05)
        )

    def test_midpoint_weight_zero_sits_on_the_background_anchor(self) -> None:
        """The permissive extreme: admit everything above background noise."""
        subject = artifact()
        config = weighted(0.0)
        assert derive_floor(subject, config=config) == pytest.approx(
            subject.calibrated(subject.bg_p99)
        )

    def test_midpoint_weight_one_sits_on_the_same_topic_anchor(self) -> None:
        """The strict extreme: reject anything weaker than the weakest match."""
        subject = artifact()
        assert derive_floor(subject, config=weighted(1.0)) == pytest.approx(
            subject.calibrated(subject.same_topic_p05)
        )

    def test_compute_stores_the_derived_floor_on_the_artifact(self) -> None:
        """Runtime reads it from here, so it has to actually be written."""
        measured = measure()
        assert measured.derived_floor == pytest.approx(derive_floor(measured), abs=1e-9)
        assert 0.0 <= measured.derived_floor <= 1.0

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


class TestMinimumPairGuards:
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


class TestCollapseDetector:
    """Tier 1. Median gap against an absolute floor.

    This is the statistic that moves when something breaks WHOLESALE. The
    missing-task-prefix bug compressed the entire similarity band; a median gap
    notices that at once, where a tail percentile can be dragged around by a
    handful of pairs — which is exactly what happened under the old population.

    Driven against hand-built artifacts rather than a synthetic corpus: collapse
    is a property of the DISTRIBUTIONS, and constructing them directly says so
    more clearly than tuning a generator until it misbehaves.
    """

    def test_a_healthy_body_passes(self) -> None:
        """Medians 0.90 and 0.60 — a 0.30 gap, well clear of the 0.13 floor."""
        check_collapse(artifact())

    def test_merged_medians_are_refused(self) -> None:
        with pytest.raises(CalibrationError, match="COLLAPSE"):
            check_collapse(artifact(bg_p50=0.60, same_topic_p50=0.66))

    def test_the_message_reports_both_medians(self) -> None:
        """The message has to say what was measured, not just that it failed."""
        with pytest.raises(CalibrationError) as caught:
            check_collapse(artifact(bg_p50=0.60, same_topic_p50=0.66))
        message = str(caught.value)
        assert "same_p50" in message and "bg_p50" in message

    def test_it_names_the_systemic_cause(self) -> None:
        """Collapse means a fault, not a marginal corpus — the message says so."""
        with pytest.raises(CalibrationError, match="task prefix"):
            check_collapse(artifact(bg_p50=0.60, same_topic_p50=0.66))

    def test_just_above_the_floor_passes(self) -> None:
        """Either side of the commissioned minimum, which is 0.1394.

        Not tested at exact equality: `same_p50 - bg_p50` is a float
        subtraction, so "exactly the floor" is not a state a caller can
        reliably construct or a guard can meaningfully promise.
        """
        floor = CONFIG.calibration.separation_guards.body_min
        check_collapse(artifact(bg_p50=0.60, same_topic_p50=0.60 + floor + 1e-6))

    def test_just_below_the_floor_is_refused(self) -> None:
        floor = CONFIG.calibration.separation_guards.body_min
        with pytest.raises(CalibrationError, match="COLLAPSE"):
            check_collapse(artifact(bg_p50=0.60, same_topic_p50=0.60 + floor - 1e-3))

    def test_the_commissioned_floor_is_half_the_measured_median_gap(self) -> None:
        """Recorded so a change to either side shows up as a failing test."""
        assert CONFIG.calibration.separation_guards.body_min == pytest.approx(0.1394)

    def test_body_separation_is_the_median_gap(self) -> None:
        assert artifact().body_separation == pytest.approx(0.90 - 0.60)


class TestPopulationRule:
    """D18. Only INDEXED questions may appear on the document side.

    Paraphrases are excluded from the vector index (amendment P) because they
    carry no answer of their own, so a paraphrase can never be the document half
    of a real comparison. Measuring para->para pairs measured something the
    production system is structurally incapable of doing — and on real
    embeddings those pairs set the same-subject p05 and failed the guard.
    """

    def test_the_population_is_recorded_on_the_artifact(self) -> None:
        assert measure().population == POPULATION

    def test_a_non_indexed_question_never_appears_as_a_document(self) -> None:
        """Only the first member of each family is indexed; the rest are queries."""
        queries, documents, topics = synthetic_corpus(40, 4)
        everything = sorted(documents)
        originals = {qid for qid in everything if qid.endswith("-0")}

        full = compute(
            query_vectors=queries,
            document_vectors=documents,
            topic_by_question=topics,
            indexed_ids=set(everything),
            embed_model_tag=MODEL_TAG,
            corpus_content_hash="x",
            computed_at=COMPUTED_AT,
        )
        restricted = compute(
            query_vectors=queries,
            document_vectors=documents,
            topic_by_question=topics,
            indexed_ids=originals,
            embed_model_tag=MODEL_TAG,
            corpus_content_hash="x",
            computed_at=COMPUTED_AT,
        )
        # 160 queries x 40 documents, minus the 40 self-pairs.
        assert restricted.background_pair_count + restricted.same_topic_pair_count == 160 * 40 - 40
        assert restricted.same_topic_pair_count == 40 * 3  # 4 members, 1 document, no self
        assert full.same_topic_pair_count > restricted.same_topic_pair_count

    def test_an_indexed_id_with_no_vector_is_refused(self) -> None:
        """A document side naming something unembeddable is a caller bug."""
        queries, documents, topics = synthetic_corpus(40, 3)
        with pytest.raises(CalibrationError, match="no document vector"):
            compute(
                query_vectors=queries,
                document_vectors=documents,
                topic_by_question=topics,
                indexed_ids={"not-a-real-id"},
                embed_model_tag=MODEL_TAG,
                corpus_content_hash="x",
                computed_at=COMPUTED_AT,
            )

    def test_the_shipped_corpus_marks_only_originals_as_indexed(self) -> None:
        rows = calibration_corpus()
        indexed = indexed_ids(rows)
        assert len(indexed) == 40
        assert all(not qid.startswith("HQP-") for qid in indexed)
        assert all(row["indexed"] is row["question_id"].startswith("HQ-") for row in rows)


class TestFloorDiscrimination:
    """Tier 2 (D19). THE PROBES GATE; THEY NEVER DERIVE.

    The floor's value comes only from corpus-internal statistics. This check is
    a canary verifying the derived floor still discriminates — the
    zero-tolerance eval core embedded into calibration, so a corpus or model
    change run through `make calibrate` alone cannot silently ship a
    non-discriminating floor.
    """

    def test_it_refuses_when_uncommissioned(self) -> None:
        """A ratchet with no baseline is not lenient, it is absent."""
        with pytest.raises(CalibrationError, match="never been commissioned"):
            check_floor_discrimination(healthy_landings(), config=uncommissioned())

    def test_the_uncommissioned_message_names_the_command(self) -> None:
        with pytest.raises(CalibrationError, match="make calibrate-commission"):
            check_floor_discrimination(healthy_landings(), config=uncommissioned())

    def test_probes_holding_their_commissioned_margins_pass(self) -> None:
        check_floor_discrimination(healthy_landings())

    def test_an_unanswerable_drifting_towards_the_floor_is_refused(self) -> None:
        """Commissioned 0.3055 at retention 0.6 requires 0.1833."""
        with pytest.raises(CalibrationError, match="FLOOR NO LONGER DISCRIMINATES"):
            check_floor_discrimination(healthy_landings(**{"2.7": 0.18}))

    def test_the_message_names_the_failing_probe(self) -> None:
        with pytest.raises(CalibrationError, match=r"unanswerable 3\.5"):
            check_floor_discrimination(healthy_landings(**{"3.5": 0.05}))

    def test_an_unanswerable_that_clears_the_floor_is_refused(self) -> None:
        """A negative margin means it now MATCHES, which is the whole failure."""
        with pytest.raises(CalibrationError, match="drifting towards answering"):
            check_floor_discrimination(healthy_landings(**{"2.8": -0.01}))

    def test_a_weakening_answerable_is_refused(self) -> None:
        """Commissioned 0.2405 at retention 0.6 requires 0.1443."""
        with pytest.raises(CalibrationError, match="drifting towards refusing"):
            check_floor_discrimination(healthy_landings(answerable=0.14))

    def test_just_above_the_retained_fraction_passes(self) -> None:
        check_floor_discrimination(healthy_landings(answerable=0.145))

    def test_a_missing_gated_probe_is_refused(self) -> None:
        """Skipping a probe would make the gate report on less than it claims."""
        partial = [item for item in healthy_landings() if item.number != "2.8"]
        with pytest.raises(CalibrationError, match=r"probe 2\.8 was not measured"):
            check_floor_discrimination(partial)

    def test_no_answerable_probes_is_refused(self) -> None:
        unanswerable_only = [item for item in healthy_landings() if item.kind == "unanswerable"]
        with pytest.raises(CalibrationError, match="no answerable probes"):
            check_floor_discrimination(unanswerable_only)

    def test_baits_are_never_gated(self) -> None:
        """3.4 is refused on legal grounds regardless of retrieval; 4.2 is
        Phase 4's pricing block. A bait anywhere must not fail the gate.
        """
        with_baits = [*healthy_landings(), landing("4.2", "bait", -0.5)]
        check_floor_discrimination(with_baits)

    def test_the_two_tiers_watch_different_things(self) -> None:
        """A corpus can pass one and fail the other, which is why there are two."""
        check_collapse(artifact())
        with pytest.raises(CalibrationError, match="FLOOR NO LONGER DISCRIMINATES"):
            check_floor_discrimination(healthy_landings(**{"2.7": 0.0}))


class TestTailSeparationIsDiagnosticOnly:
    """D19 demoted `same_p05 - bg_p99`. It is reported and gates nothing.

    It was a PROXY and it diverged from its target: open-ended downward (the
    same-subject tail is set by the hardest legitimate paraphrase, an authorship
    boundary with no crisp edge), non-convergent under data addition (two
    paraphrases per family gave +0.0237, three gave -0.0245), and correlated
    (version families contribute pairs against two byte-identical documents).
    It failed while every operational margin held or improved.
    """

    def test_it_is_still_computed_and_reported(self) -> None:
        assert artifact().separation == pytest.approx(0.85 - 0.75)

    def test_a_negative_tail_no_longer_blocks_calibration(self) -> None:
        """The exact shape of the real failure: tail negative, body healthy.

        Under D18 this refused. Under D19 tier 1 passes it and the probes
        decide whether the floor is acceptable.
        """
        subject = artifact(same_topic_p05=0.6400, bg_p99=0.7000)
        assert subject.separation < 0
        check_collapse(subject)
        check_floor_discrimination(healthy_landings())


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
        assert loaded.derived_floor == pytest.approx(subject.derived_floor)

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
        assert len(rows) == 148, "40 corpus questions + 108 paraphrases"

    def test_every_row_carries_a_family(self) -> None:
        assert all(row["topic_family"] for row in calibration_corpus())

    def test_question_ids_are_unique(self) -> None:
        ids = [row["question_id"] for row in calibration_corpus()]
        assert len(set(ids)) == len(ids)

    def test_it_clears_both_minimum_pair_guards(self) -> None:
        """The shipped corpus must be able to calibrate itself.

        Counted under the D18 POPULATION RULE, not over all pairs: queries are
        every family member, documents are indexed originals only. Computed from
        the family structure rather than by embedding, so this stays a fast unit
        test and still fails the moment the corpus shrinks past what the guards
        permit.
        """
        rows = calibration_corpus()
        indexed = indexed_ids(rows)
        members: dict[str, int] = {}
        documents: dict[str, int] = {}
        for row in rows:
            family = row["topic_family"]
            members[family] = members.get(family, 0) + 1
            if row["question_id"] in indexed:
                documents[family] = documents.get(family, 0) + 1

        same = sum(members[f] * documents[f] - documents[f] for f in documents)
        total = len(rows) * len(indexed) - len(indexed)
        assert same == 128, "96 from 32 singleton families + 32 from 4 version families"
        assert same >= CONFIG.calibration.min_same_topic_pairs
        assert total - same >= CONFIG.calibration.min_background_pairs
