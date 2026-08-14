"""Stage C wiring: the fallback paths, driven without a graph or a model.

The centrepiece is the recorded malformed response from `llama3.1:8b`, fed
through the real Reranker to prove the all-or-nothing fallback fires for that
question while a well-formed response still reranks normally.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import httpx

from src.contracts import Outcome
from src.contracts.thresholds import scoring_config
from src.gateway.client import GatewayClient
from src.retrieval.retriever import Reranker, _age_days, build_rerank_prompt
from src.retrieval.scoring import CandidateInput

RECORDED = Path(__file__).resolve().parents[2] / "fixtures" / "recorded" / "rerank"
GATEWAY = GatewayClient(base_url="http://gateway.test", api_key="k")


def recorded(name: str) -> dict[str, object]:
    with (RECORDED / name).open(encoding="utf-8") as handle:
        return dict(json.load(handle))


def candidates(count: int) -> list[CandidateInput]:
    return [
        CandidateInput(
            question_id="q-1",
            matched_question_id="hq-1",
            answer_node_id=f"ANS-{i:04d}",
            tier1_summary=f"summary {i}",
            vector_score=0.7,
            outcome=Outcome.UNKNOWN,
            age_days=100,
            has_evidence=False,
        )
        for i in range(count)
    ]


def gateway_reply(content: str, *, status: int = 200) -> httpx.AsyncClient:
    def handler(request: httpx.Request) -> httpx.Response:
        if status != 200:
            return httpx.Response(status, text="upstream unavailable")
        return httpx.Response(200, json={"choices": [{"message": {"content": content}}]})

    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


class TestTheRecordedMalformedResponse:
    """A field-observed failure, exercised through the real Stage C path."""

    async def test_the_all_or_nothing_fallback_fires(self) -> None:
        entry = recorded("malformed_missing_index.json")
        reranker = Reranker(gateway=GATEWAY, recorded={"q-1": str(entry["raw_content"])})
        outcome = await reranker.score(
            question_id="q-1", question_text="anything", candidates=candidates(8)
        )
        assert outcome.usable is False
        assert outcome.scores is None
        assert "missing [1]" in (outcome.reason or "")

    async def test_a_well_formed_response_reranks_normally(self) -> None:
        """The malformed case only means something if the normal case works."""
        entry = recorded("well_formed.json")
        reranker = Reranker(gateway=GATEWAY, recorded={"q-2": str(entry["raw_content"])})
        outcome = await reranker.score(
            question_id="q-2", question_text="anything", candidates=candidates(8)
        )
        assert outcome.usable is True
        assert outcome.scores is not None
        assert sorted(outcome.scores) == list(range(8))

    async def test_one_bad_question_does_not_affect_another(self) -> None:
        """The requirement: the fallback is per question, not per run."""
        bad = str(recorded("malformed_missing_index.json")["raw_content"])
        good = str(recorded("well_formed.json")["raw_content"])
        reranker = Reranker(gateway=GATEWAY, recorded={"q-bad": bad, "q-good": good})

        bad_outcome = await reranker.score(
            question_id="q-bad", question_text="x", candidates=candidates(8)
        )
        good_outcome = await reranker.score(
            question_id="q-good", question_text="x", candidates=candidates(8)
        )
        assert bad_outcome.usable is False
        assert good_outcome.usable is True


class TestFailureModes:
    """Nothing here raises — a rerank failure degrades one question."""

    async def test_disabled_in_config_skips_cleanly(self) -> None:
        config = scoring_config().model_copy(
            update={"rerank": scoring_config().rerank.model_copy(update={"enabled": False})}
        )
        reranker = Reranker(gateway=GATEWAY, config=config)
        outcome = await reranker.score(
            question_id="q-1", question_text="x", candidates=candidates(4)
        )
        assert outcome.usable is False
        assert "disabled" in (outcome.reason or "")

    async def test_no_candidates_skips_cleanly(self) -> None:
        outcome = await Reranker(gateway=GATEWAY).score(
            question_id="q-1", question_text="x", candidates=[]
        )
        assert outcome.usable is False
        assert "no candidates" in (outcome.reason or "")

    async def test_a_gateway_error_falls_back(self) -> None:
        reranker = Reranker(gateway=GATEWAY)
        async with gateway_reply("", status=503) as client:
            outcome = await reranker.score(
                question_id="q-1",
                question_text="x",
                candidates=candidates(4),
                client=client,
            )
        assert outcome.usable is False
        assert "gateway error" in (outcome.reason or "")

    async def test_a_live_call_is_parsed(self) -> None:
        reply = '{"scores": [{"index": 0, "score": 0.9}, {"index": 1, "score": 0.1}]}'
        reranker = Reranker(gateway=GATEWAY)
        async with gateway_reply(reply) as client:
            outcome = await reranker.score(
                question_id="q-1",
                question_text="x",
                candidates=candidates(2),
                client=client,
            )
        assert outcome.usable is True
        assert outcome.scores == {0: 0.9, 1: 0.1}

    async def test_a_reply_without_content_falls_back(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"choices": []})

        reranker = Reranker(gateway=GATEWAY)
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            outcome = await reranker.score(
                question_id="q-1",
                question_text="x",
                candidates=candidates(2),
                client=client,
            )
        assert outcome.usable is False


class TestPromptShape:
    def test_it_matches_what_was_recorded(self) -> None:
        """The recordings were captured against this exact prompt."""
        prompt = build_rerank_prompt("A question?", candidates(2))
        assert "Score how relevant each candidate answer is" in prompt
        assert '"scores"' in prompt
        assert "QUESTION: A question?" in prompt
        assert "[0] summary 0" in prompt
        assert "[1] summary 1" in prompt

    def test_the_configured_timeout_is_the_measured_one(self) -> None:
        """120.7s measured on this CPU host; 300 leaves real headroom."""
        assert scoring_config().rerank.timeout_seconds == 300

    def test_temperature_is_zero(self) -> None:
        """A reranker that varies between runs makes every eval noise."""
        assert scoring_config().rerank.temperature == 0


class TestAgeInDays:
    def test_an_answer_written_today_is_zero(self) -> None:
        assert _age_days("2026-08-11", today=date(2026, 8, 11)) == 0

    def test_a_year_ago(self) -> None:
        assert _age_days("2025-08-11", today=date(2026, 8, 11)) == 365

    def test_a_future_date_is_floored_at_zero(self) -> None:
        assert _age_days("2027-01-01", today=date(2026, 8, 11)) == 0

    def test_an_unparseable_date_is_treated_as_ancient(self) -> None:
        """Never as 'written today' — that would hand a broken record the
        maximum recency bonus."""
        assert _age_days("not-a-date", today=date(2026, 8, 11)) == 10_000
        assert _age_days("", today=date(2026, 8, 11)) == 10_000
