"""Stage C parsing, including the malformed response observed in the field.

The centrepiece is `test_the_observed_malformed_response_falls_back`: a real
reply from `llama3.1:8b`, committed verbatim, that returned seven scores for
eight candidates. It reproduced identically on a second call at temperature 0,
so it is a property of the model on this prompt rather than a one-off — which is
exactly why it belongs in the suite rather than in a commit message.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from src.retrieval.rerank import parse_rerank_response

RECORDED = Path(__file__).resolve().parents[2] / "fixtures" / "recorded" / "rerank"


def _recorded(name: str) -> dict[str, Any]:
    with (RECORDED / name).open(encoding="utf-8") as handle:
        return dict(json.load(handle))


class TestTheObservedFailure:
    def test_the_recording_is_what_we_think_it_is(self) -> None:
        """Guards the fixture itself — a 'fixed' recording would void the test."""
        recorded = _recorded("malformed_missing_index.json")
        assert recorded["candidate_count"] == 8
        assert recorded["score_count"] == 7
        assert recorded["scored_indices"] == [0, 2, 3, 4, 5, 6, 7]
        assert 1 not in recorded["scored_indices"]

    def test_the_observed_malformed_response_falls_back(self) -> None:
        recorded = _recorded("malformed_missing_index.json")
        outcome = parse_rerank_response(
            recorded["raw_content"], candidate_count=recorded["candidate_count"]
        )
        assert outcome.usable is False
        assert outcome.scores is None
        assert outcome.reason is not None
        assert "missing [1]" in outcome.reason

    def test_a_well_formed_response_still_reranks(self) -> None:
        """The malformed case only means something if the normal case works."""
        recorded = _recorded("well_formed.json")
        outcome = parse_rerank_response(
            recorded["raw_content"], candidate_count=recorded["candidate_count"]
        )
        assert outcome.usable is True
        assert outcome.scores is not None
        assert sorted(outcome.scores) == list(range(8))
        assert all(0.0 <= score <= 1.0 for score in outcome.scores.values())

    def test_both_recordings_name_the_prompt_version(self) -> None:
        """A prompt change must force re-recording, not silent comparison."""
        for name in ("malformed_missing_index.json", "well_formed.json"):
            recorded = _recorded(name)
            assert recorded["prompt_name"] == "rerank"
            assert recorded["prompt_version"] == 1
            assert recorded["model_tag"] == "llama3.1:8b"


class TestPartialResponsesAreNeverPartiallyApplied:
    """Scoring what came back and defaulting the rest to zero would be worse
    than not reranking: zero is an active claim of irrelevance, and the missing
    candidate may be the right answer."""

    def test_a_missing_index_rejects_the_whole_response(self) -> None:
        content = '{"scores": [{"index": 0, "score": 0.9}, {"index": 2, "score": 0.4}]}'
        outcome = parse_rerank_response(content, candidate_count=3)
        assert outcome.usable is False
        assert "missing [1]" in (outcome.reason or "")

    def test_an_extra_index_rejects_the_whole_response(self) -> None:
        content = (
            '{"scores": [{"index": 0, "score": 0.9}, {"index": 1, "score": 0.4}, '
            '{"index": 7, "score": 0.1}]}'
        )
        outcome = parse_rerank_response(content, candidate_count=2)
        assert outcome.usable is False
        assert "unexpected [7]" in (outcome.reason or "")

    def test_a_duplicated_index_is_rejected(self) -> None:
        content = '{"scores": [{"index": 0, "score": 0.9}, {"index": 0, "score": 0.2}]}'
        assert parse_rerank_response(content, candidate_count=1).usable is False


class TestMalformedShapes:
    @pytest.mark.parametrize(
        ("content", "fragment"),
        [
            ("", "empty response"),
            ("I cannot score these.", "no JSON object"),
            # Truncated before the closing brace — there is no object to extract.
            ('{"scores": [', "no JSON object"),
            # Braces present, contents invalid.
            ('{"scores": [0.5,]}', "not valid JSON"),
            ('{"result": []}', "no `scores` array"),
            ('{"scores": "high"}', "no `scores` array"),
            ('{"scores": ["0.9"]}', "not an object"),
            ('{"scores": [{"index": "a", "score": 0.9}]}', "non-integer index"),
            ('{"scores": [{"index": 0, "score": "high"}]}', "non-numeric score"),
            ('{"scores": [{"index": 0, "score": 1.4}]}', "outside [0, 1]"),
            ('{"scores": [{"index": 0, "score": -0.2}]}', "outside [0, 1]"),
        ],
    )
    def test_each_is_rejected_with_a_reason(self, content: str, fragment: str) -> None:
        outcome = parse_rerank_response(content, candidate_count=1)
        assert outcome.usable is False
        assert fragment in (outcome.reason or "")

    def test_nothing_raises(self) -> None:
        """A rerank failure degrades one question; it never crashes a run."""
        for content in ("", "garbage", '{"scores": null}', "[]", "{}"):
            assert parse_rerank_response(content, candidate_count=4).usable is False


class TestLenience:
    """Decoration around the JSON is tolerated; wrong content never is."""

    def test_a_code_fence_is_tolerated(self) -> None:
        content = '```json\n{"scores": [{"index": 0, "score": 0.5}]}\n```'
        assert parse_rerank_response(content, candidate_count=1).usable is True

    def test_surrounding_prose_is_tolerated(self) -> None:
        content = 'Here are the scores:\n{"scores": [{"index": 0, "score": 0.5}]}\nHope that helps.'
        assert parse_rerank_response(content, candidate_count=1).usable is True

    def test_integer_scores_are_accepted(self) -> None:
        """Models emit 1 rather than 1.0 constantly."""
        outcome = parse_rerank_response('{"scores": [{"index": 0, "score": 1}]}', candidate_count=1)
        assert outcome.usable is True
        assert outcome.scores == {0: 1.0}

    def test_booleans_are_not_numbers(self) -> None:
        """bool is a subclass of int in Python; True must not read as 1.0."""
        content = '{"scores": [{"index": 0, "score": true}]}'
        assert parse_rerank_response(content, candidate_count=1).usable is False

    def test_zero_candidates_is_rejected(self) -> None:
        assert parse_rerank_response('{"scores": []}', candidate_count=0).usable is False
