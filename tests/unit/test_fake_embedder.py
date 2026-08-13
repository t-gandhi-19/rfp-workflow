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

import math

import pytest

from src.contracts.embedding import EmbedRole
from src.gateway.fake_embedder import (
    FAKE_EMBEDDINGS_ENV,
    fake_embedding,
    fake_embeddings,
    fake_embeddings_enabled,
    stem,
    tokenize,
)

DIMENSIONS = 768


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
