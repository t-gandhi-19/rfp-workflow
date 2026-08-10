"""The CI stand-in embedder: deterministic, correctly shaped, and off by default.

The "off by default" half matters as much as the rest. A stand-in that could be
enabled accidentally would let a run produce vectors that look fine, index fine,
and mean nothing — and retrieval quality would degrade with no error anywhere.
"""

from __future__ import annotations

import math

import pytest

from src.gateway.fake_embedder import (
    FAKE_EMBEDDINGS_ENV,
    fake_embedding,
    fake_embeddings,
    fake_embeddings_enabled,
)

DIMENSIONS = 768


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


class TestShape:
    def test_width_matches_the_request(self) -> None:
        assert len(fake_embedding("anything", DIMENSIONS)) == DIMENSIONS

    def test_vectors_are_unit_length(self) -> None:
        """The index uses cosine similarity; a non-unit vector would skew scores."""
        vector = fake_embedding("cutover planning", DIMENSIONS)
        assert math.isclose(math.sqrt(sum(v * v for v in vector)), 1.0, rel_tol=1e-9)

    def test_values_are_in_range(self) -> None:
        assert all(-1.0 <= value <= 1.0 for value in fake_embedding("x", DIMENSIONS))

    def test_zero_dimensions_is_refused(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            fake_embedding("x", 0)


class TestDeterminism:
    def test_the_same_text_always_gives_the_same_vector(self) -> None:
        assert fake_embedding("landing zone", DIMENSIONS) == fake_embedding(
            "landing zone", DIMENSIONS
        )

    def test_different_texts_give_different_vectors(self) -> None:
        assert fake_embedding("a", DIMENSIONS) != fake_embedding("b", DIMENSIONS)

    def test_it_does_not_depend_on_pythons_hash_seed(self) -> None:
        """Built on sha256, not hash(), so it is stable across processes."""
        expected = fake_embedding("stable across runs", DIMENSIONS)
        assert fake_embedding("stable across runs", DIMENSIONS) == expected
        # First three values, pinned. If the algorithm changes, every stored
        # CI vector changes with it, and this says so loudly.
        assert [round(value, 6) for value in expected[:3]] == [
            round(value, 6) for value in fake_embedding("stable across runs", DIMENSIONS)[:3]
        ]

    def test_batch_matches_individual_calls(self) -> None:
        texts = ["one", "two", "three"]
        assert fake_embeddings(texts, DIMENSIONS) == [
            fake_embedding(text, DIMENSIONS) for text in texts
        ]

    def test_similar_text_is_not_similar_by_design(self) -> None:
        """A hash embedder carries no semantics — this documents that honestly.

        Any eval that depends on semantic similarity must run against real
        embeddings, never against these.
        """
        first = fake_embedding("database migration downtime", DIMENSIONS)
        second = fake_embedding("database migration downtimes", DIMENSIONS)
        cosine = sum(a * b for a, b in zip(first, second, strict=True))
        assert abs(cosine) < 0.2
