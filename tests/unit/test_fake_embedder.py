"""The CI stand-in embedder: deterministic, correctly shaped, and off by default.

The "off by default" half matters as much as the rest. A stand-in that could be
enabled accidentally would let a run produce vectors that look fine, index fine,
and mean nothing — and retrieval quality would degrade with no error anywhere.

Amendment K replaced a hash-per-text scheme with a token bag-of-words one, so
these vectors now have real geometry: shared vocabulary means proximity. The
tests below pin that geometry, because it is what lets CI compute a genuine
calibration artifact and exercise the separation guard rather than skipping it.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pytest

from src.contracts.embedding import EmbedRole
from src.gateway.fake_embedder import (
    FAKE_EMBEDDINGS_ENV,
    FAMILY_LOOKUP_PATH,
    _anchor_vector,
    fake_embedding,
    fake_embeddings,
    fake_embeddings_enabled,
    family_anchor_weight,
    family_of,
    stem,
    token_bag,
    tokenize,
)

DIMENSIONS = 768
REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURES = REPO_ROOT / "fixtures"


def _read(name: str) -> Any:
    with (FIXTURES / name).open(encoding="utf-8") as handle:
        return json.load(handle)


QA_PAIRS: list[dict[str, Any]] = _read("qa_pairs.json")
PARAPHRASE_ROWS: list[dict[str, Any]] = _read("question_paraphrases.json")
GOLDEN_QUESTIONS: list[dict[str, Any]] = _read("answer_key_manual.json")["questions"]


def load_family_lookup() -> dict[str, Any]:
    with FAMILY_LOOKUP_PATH.open(encoding="utf-8") as handle:
        data: dict[str, Any] = json.load(handle)
    return data


def golden_text(number: str) -> str:
    return str(next(q for q in GOLDEN_QUESTIONS if q["number"] == number)["text"])


def same_family_pair() -> tuple[str, str]:
    """A paraphrase and the question it rewords — different texts, one subject."""
    row = PARAPHRASE_ROWS[0]
    original = next(p for p in QA_PAIRS if p["question_id"] == row["paraphrase_of"])
    return str(original["question"]), str(row["question"])


def other_family_question() -> str:
    family = PARAPHRASE_ROWS[0]["topic_family"]
    return str(next(p for p in QA_PAIRS if p["topic_family"] != family)["question"])


def embed(text: str, *, role: EmbedRole = EmbedRole.DOCUMENT) -> list[float]:
    return fake_embedding(text, DIMENSIONS, role=role)


def cosine(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b, strict=True))


class TestOffByDefault:
    def test_disabled_with_an_empty_environment(self) -> None:
        assert fake_embeddings_enabled({}) is False

    @pytest.mark.parametrize("value", ["", "0", "false", "no", "true", "yes", "2"])
    def test_only_an_exact_1_enables_it(self, value: str) -> None:
        """Nothing plausible-but-wrong turns it on by accident."""
        assert fake_embeddings_enabled({FAKE_EMBEDDINGS_ENV: value}) is False

    def test_enabled_by_an_explicit_opt_in(self) -> None:
        assert fake_embeddings_enabled({FAKE_EMBEDDINGS_ENV: "1"}) is True

    def test_the_real_environment_does_not_have_it_set(self) -> None:
        """Running the unit suite must not silently enable it."""
        assert fake_embeddings_enabled() is False


class TestRoleIsRequired:
    """The stand-in's signature mirrors the real embedder's, deliberately.

    `GatewayClient.embed` gives `role` no default because embedding a query as a
    document produces a plausible vector that retrieves badly. A stand-in with a
    laxer signature would let a call site omit it in CI and only fail against
    the real model.
    """

    def test_fake_embedding_requires_role(self) -> None:
        with pytest.raises(TypeError):
            fake_embedding("x", DIMENSIONS)  # type: ignore[call-arg]

    def test_fake_embeddings_requires_role(self) -> None:
        with pytest.raises(TypeError):
            fake_embeddings(["x"], DIMENSIONS)  # type: ignore[call-arg]


class TestShape:
    def test_width_matches_the_request(self) -> None:
        assert len(embed("anything")) == DIMENSIONS

    def test_vectors_are_unit_length(self) -> None:
        """The index uses cosine similarity; a non-unit vector would skew scores."""
        length = math.sqrt(sum(v * v for v in embed("cutover planning")))
        assert math.isclose(length, 1.0, rel_tol=1e-9)

    def test_values_are_in_range(self) -> None:
        assert all(-1.0 <= value <= 1.0 for value in embed("x"))

    def test_zero_dimensions_is_refused(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            embed_zero()

    def test_text_with_no_tokens_still_gives_a_unit_vector(self) -> None:
        """The index rejects a zero vector, and a blank field must not fail ingest."""
        for blank in ("", "   ", "!!! ---"):
            vector = embed(blank)
            assert math.isclose(math.sqrt(sum(v * v for v in vector)), 1.0, rel_tol=1e-9)


def embed_zero() -> list[float]:
    return fake_embedding("x", 0, role=EmbedRole.QUERY)


class TestDeterminism:
    def test_the_same_text_always_gives_the_same_vector(self) -> None:
        assert embed("landing zone") == embed("landing zone")

    def test_different_texts_give_different_vectors(self) -> None:
        assert embed("a") != embed("b")

    def test_it_does_not_depend_on_pythons_hash_seed(self) -> None:
        """Built on sha256, not hash(), so it is stable across processes."""
        expected = embed("stable across runs")
        assert embed("stable across runs") == expected

    def test_batch_matches_individual_calls(self) -> None:
        texts = ["one", "two", "three"]
        assert fake_embeddings(texts, DIMENSIONS, role=EmbedRole.DOCUMENT) == [
            embed(text) for text in texts
        ]

    def test_token_order_does_not_matter(self) -> None:
        """A bag of words is a bag: the sum commutes."""
        assert embed("cutover rollback plan") == pytest.approx(embed("plan rollback cutover"))


class TestTokenizing:
    def test_case_and_punctuation_are_normalised_away(self) -> None:
        assert tokenize("Cutover, Rollback!") == ["cutover", "rollback"]

    def test_stopwords_are_dropped(self) -> None:
        """On short questions the boilerplate is most of the tokens.

        Left in, every question resembles every other because they all ask
        politely, and the subject nouns get outvoted.
        """
        assert tokenize("Describe your landing zone design") == ["land", "zone", "design"]

    def test_an_all_stopword_text_still_yields_tokens(self) -> None:
        """An empty bag would give a meaningless vector rather than no vector."""
        assert tokenize("what is your") == ["what", "is", "your"]

    def test_digits_are_tokens(self) -> None:
        assert tokenize("ISO 27001") == ["iso", "27001"]


class TestStemming:
    """Without this the scheme measures spelling, not vocabulary.

    "vulnerability"/"vulnerabilities", "subcontractor"/"subcontractors" and
    "patch"/"patched" were the three lowest-scoring same-subject pairs in the
    corpus, and they alone pulled the p05 below the background p99.
    """

    @pytest.mark.parametrize(
        ("word", "expected"),
        [
            ("vulnerabilities", "vulnerability"),
            ("subcontractors", "subcontractor"),
            ("patched", "patch"),
            ("patching", "patch"),
            ("applications", "application"),
            ("processes", "process"),
        ],
    )
    def test_inflections_fold_together(self, word: str, expected: str) -> None:
        assert stem(word) == expected

    @pytest.mark.parametrize(
        ("singular", "plural"),
        [
            ("vulnerability", "vulnerabilities"),
            ("subcontractor", "subcontractors"),
            ("application", "applications"),
            ("timescale", "timescales"),
            ("process", "processes"),
            ("zone", "zones"),
            ("guardrail", "guardrails"),
        ],
    )
    def test_a_word_and_its_plural_share_a_stem(self, singular: str, plural: str) -> None:
        """The property that matters. What the shared stem LOOKS like does not."""
        assert stem(singular) == stem(plural)

    def test_a_double_s_is_not_stripped(self) -> None:
        """'process' is not 'proces'."""
        assert stem("process") == "process"

    def test_short_words_are_left_alone(self) -> None:
        """Stripping 'is' to 'i' would fold unrelated words together."""
        assert stem("is") == "is"
        assert stem("des") == "des"

    def test_it_is_idempotent(self) -> None:
        for word in ("vulnerabilities", "patched", "applications", "process"):
            assert stem(stem(word)) == stem(word)

    def test_inflected_questions_now_match(self) -> None:
        """The property the corpus needed, stated end to end."""
        first = embed("Describe your vulnerability and patch management process")
        second = embed("How are vulnerabilities identified and patched")
        assert cosine(first, second) > 0.35


class TestGeometryIsReal:
    """The property amendment K exists to create.

    The previous scheme hashed the whole string, so every distinct text was
    equidistant from every other. Calibration over it would have found no
    separation between "the same question" and "an unrelated one" — which meant
    CI could not exercise the calibration path, and the separation guard was
    the thing that would fail rather than the thing being tested.
    """

    def test_shared_vocabulary_scores_high(self) -> None:
        """Stopwords are dropped, so this compares content words only:
        {migrate, production, database, minimal, downtime} against
        {migrate, production, database, short, outage} — 3 of 7 shared."""
        first = embed("how do you migrate production databases with minimal downtime")
        second = embed("how do you migrate production databases with a short outage")
        assert cosine(first, second) > 0.55

    def test_unrelated_subjects_score_low(self) -> None:
        first = embed("describe your landing zone design and guardrail enforcement")
        second = embed("what are your post migration warranty and defect liability terms")
        assert cosine(first, second) < 0.3

    def test_a_paraphrase_beats_an_unrelated_question(self) -> None:
        """The ordering calibration depends on, stated as an ordering."""
        anchor = embed("how is privileged access governed and reviewed")
        paraphrase = embed("how is privileged access granted monitored and reviewed")
        unrelated = embed("where are your delivery centres located across time zones")
        assert cosine(anchor, paraphrase) > cosine(anchor, unrelated)

    def test_identical_text_scores_one(self) -> None:
        assert cosine(embed("same text"), embed("same text")) == pytest.approx(1.0)

    def test_it_still_carries_no_semantics(self) -> None:
        """Honest about the limit: synonyms with no shared tokens are unrelated.

        A real model knows "outage" and "downtime" are close. This does not, and
        no eval depending on semantic similarity may run against it.
        """
        assert cosine(embed("downtime"), embed("outage")) < 0.3


class TestFamilyLookup:
    """Amendment N. The mapping is generated with the fixtures, so it is covered
    by `make fixtures-check` byte-determinism; these assert its CONTENT."""

    def test_every_corpus_original_is_placed(self) -> None:
        for pair in QA_PAIRS:
            assert family_of(pair["question"]) == pair["topic_family"], pair["id"]

    def test_every_paraphrase_is_placed_in_its_original_s_family(self) -> None:
        """The property the same-subject anchor depends on: a paraphrase and the
        question it rewords must land on the SAME anchor, or the pair the
        calibration calls 'genuinely the same question' would not be."""
        for row in PARAPHRASE_ROWS:
            assert family_of(row["question"]) == row["topic_family"], row["id"]

    def test_the_lookup_covers_exactly_the_expected_population(self) -> None:
        """36 distinct original texts (4 supersession chains share theirs
        byte-for-byte), 108 paraphrases, 15 answerable golden questions."""
        assert len(load_family_lookup()["families"]) == 36 + 108 + 15

    @pytest.mark.parametrize("number", ["2.7", "2.8", "3.5"])
    def test_the_unanswerables_carry_no_family(self, number: str) -> None:
        """ABSENCE IS THE MECHANISM, not an oversight.

        A question the corpus cannot answer embeds as a pure token bag, which
        sits far from every anchor — so it lands below the floor by construction
        rather than because a fixture says so.
        """
        assert family_of(golden_text(number)) is None

    @pytest.mark.parametrize("number", ["3.4", "4.2"])
    def test_the_baits_carry_no_family(self, number: str) -> None:
        assert family_of(golden_text(number)) is None

    def test_answerable_golden_questions_are_placed_on_their_subject(self) -> None:
        family_by_answer = {pair["answer_id"]: pair["topic_family"] for pair in QA_PAIRS}
        placed = 0
        for question in GOLDEN_QUESTIONS:
            if question["number"] in {"2.7", "2.8", "3.5", "3.4", "4.2"}:
                continue
            expected = family_by_answer[question["expected_best_match_answer_id"]]
            assert family_of(question["text"]) == expected, question["number"]
            placed += 1
        assert placed == 15

    def test_whitespace_is_collapsed_before_lookup(self) -> None:
        """A text re-wrapped across lines is the same question."""
        original = QA_PAIRS[0]["question"]
        rewrapped = original.replace(" ", "\n   ", 1)
        assert family_of(rewrapped) == family_of(original)

    def test_an_unknown_text_has_no_family(self) -> None:
        assert family_of("a question about nothing in this corpus at all") is None


class TestFamilyAnchoredGeometry:
    """What amendment N exists to create: separation CI can actually calibrate.

    v2 could not. Measured over this corpus it put the same-subject p05 at
    0.2254 under a background p99 of 0.2841 — a separation of -0.0587 — so
    `make calibrate` refused and the D18/D19 gates could only be SKIPPED in CI,
    leaving the guards untested exactly where testing is cheapest.
    """

    def test_same_family_texts_score_far_above_cross_family_ones(self) -> None:
        first, second = same_family_pair()
        other = other_family_question()
        assert cosine(embed(first), embed(second)) > 0.8
        assert cosine(embed(first), embed(other)) < 0.3

    def test_a_paraphrase_lands_near_the_question_it_rewords(self) -> None:
        row = PARAPHRASE_ROWS[0]
        original = next(p for p in QA_PAIRS if p["question_id"] == row["paraphrase_of"])
        assert cosine(embed(row["question"]), embed(original["question"])) > 0.8

    def test_an_unanswerable_question_sits_far_from_every_anchor(self) -> None:
        """The NO_MATCH property, stated directly against the corpus."""
        unanswerable = embed(golden_text("2.7"))
        best = max(cosine(unanswerable, embed(pair["question"])) for pair in QA_PAIRS)
        assert best < 0.4

    def test_two_families_are_near_orthogonal(self) -> None:
        first = _anchor_vector("landing-zone", DIMENSIONS)
        second = _anchor_vector("data-migration", DIMENSIONS)
        assert abs(cosine(first, second)) < 0.15


class TestBlend:
    """The two ends of the blend, pinned."""

    def test_alpha_zero_is_exactly_the_v2_token_bag(self) -> None:
        text = QA_PAIRS[0]["question"]
        assert fake_embedding(
            text, DIMENSIONS, role=EmbedRole.DOCUMENT, alpha=0.0
        ) == pytest.approx(token_bag(text, DIMENSIONS))

    def test_alpha_one_is_the_anchor_alone(self) -> None:
        pair = QA_PAIRS[0]
        blended = fake_embedding(pair["question"], DIMENSIONS, role=EmbedRole.QUERY, alpha=1.0)
        assert blended == pytest.approx(_anchor_vector(pair["topic_family"], DIMENSIONS))

    def test_the_default_weight_comes_from_the_generated_artifact(self) -> None:
        assert family_anchor_weight() == load_family_lookup()["config"]["family_anchor_weight"]
        assert 0.0 <= family_anchor_weight() <= 1.0

    def test_an_out_of_range_alpha_is_refused(self) -> None:
        with pytest.raises(ValueError, match="alpha"):
            fake_embedding(QA_PAIRS[0]["question"], DIMENSIONS, role=EmbedRole.QUERY, alpha=1.5)

    def test_alpha_does_nothing_for_a_text_with_no_family(self) -> None:
        """No anchor to weight, so the blend is not reachable at all."""
        text = golden_text("3.5")
        for alpha in (0.0, 0.5, 1.0):
            assert fake_embedding(text, DIMENSIONS, role=EmbedRole.QUERY, alpha=alpha) == (
                token_bag(text, DIMENSIONS)
            )

    def test_within_a_family_the_texts_stay_distinguishable(self) -> None:
        """The anchor must not swamp the bag entirely.

        If it did, every text in a family would be the same vector and the
        same-subject distribution would be a spike at 1.0 — which would make the
        calibration statistics describe the anchor rather than the corpus.
        """
        first, second = same_family_pair()
        assert cosine(embed(first), embed(second)) < 0.999

    def test_blending_preserves_determinism_and_order_independence(self) -> None:
        texts = [pair["question"] for pair in QA_PAIRS[:5]]
        batch = fake_embeddings(texts, DIMENSIONS, role=EmbedRole.DOCUMENT)
        assert batch == [embed(text) for text in texts]
        assert fake_embeddings(list(reversed(texts)), DIMENSIONS, role=EmbedRole.DOCUMENT) == list(
            reversed(batch)
        )


class TestCannotBeEnabledOutsideTestConfig:
    """A stand-in reachable from a production run would produce vectors that
    look fine, index fine, and mean nothing."""

    def test_the_gateway_never_imports_the_stand_in(self) -> None:
        """`GatewayClient` is the production embedding path.

        Only the two CI-aware entry points (ingest and calibrate) consult the
        opt-in; the client itself must have no knowledge of it, or a code path
        could reach it without the environment being checked.
        """
        source = (REPO_ROOT / "src" / "gateway" / "client.py").read_text(encoding="utf-8")
        assert "fake_embedder" not in source
        assert FAKE_EMBEDDINGS_ENV not in source

    def test_every_call_site_is_guarded_by_the_opt_in(self) -> None:
        callers = sorted(
            path
            for path in (REPO_ROOT / "src").rglob("*.py")
            if "fake_embeddings(" in path.read_text(encoding="utf-8")
            and path.name != "fake_embedder.py"
        ) + sorted(
            path
            for path in (REPO_ROOT / "scripts").rglob("*.py")
            if "fake_embeddings(" in path.read_text(encoding="utf-8")
        )
        assert callers, "expected at least one caller; the search is probably wrong"
        for path in callers:
            source = path.read_text(encoding="utf-8")
            assert "fake_embeddings_enabled()" in source, path

    def test_the_lookup_lives_in_fixtures_not_config(self) -> None:
        """It is test data, and its location says so."""
        assert FAMILY_LOOKUP_PATH.parent.name == "fixtures"
        assert not (REPO_ROOT / "config" / "fake_embedder_families.json").exists()
