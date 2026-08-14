"""Stages A and C wired to the graph and the gateway (build prompt §10).

Stage A embeds the question and asks the graph for in-domain candidates the
requesting customer is permitted to see. Stage C reranks the survivors in one
batched call. Stages B and D are the pure arithmetic in :mod:`src.retrieval.scoring`.

Two things are deliberate:

**`requesting_customer` is threaded, never defaulted.** It is required by
`find_similar_questions` and required here, so there is no code path that
retrieves without stating who is asking.

**A rerank failure degrades one question, never the run.** Timeouts, gateway
errors and unusable responses all resolve to "no rerank for this question", and
:func:`src.retrieval.scoring.blend` redistributes the weight. Nothing raises.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import date

import httpx
from neo4j import AsyncSession

from src.contracts import Outcome, RetrievalResult
from src.contracts.embedding import embedding_config
from src.contracts.thresholds import ScoringConfig, scoring_config
from src.gateway.client import GatewayClient, GatewayError
from src.graph import queries
from src.prompts import load_prompt
from src.retrieval.calibration import CalibrationArtifact, load_for_current_corpus
from src.retrieval.rerank import RerankOutcome, parse_rerank_response
from src.retrieval.scoring import CandidateInput, score_candidates, to_retrieval_result


@dataclass
class RetrievalTrace:
    """Why a question scored the way it did. Rendered into the eval report."""

    question_id: str
    candidates_considered: int = 0
    reranked: bool = False
    rerank_skip_reason: str | None = None
    notes: list[str] = field(default_factory=list)


def _age_days(answer_date: str, *, today: date) -> int:
    """Whole days between an answer's date and the run date, floored at zero."""
    try:
        written = date.fromisoformat(answer_date)
    except (TypeError, ValueError):
        # An unparseable date must not be read as "written today" — that would
        # hand a broken record the maximum recency bonus.
        return 10_000
    return max(0, (today - written).days)


def build_rerank_prompt(question: str, candidates: list[CandidateInput]) -> str:
    """The Stage C prompt, LOADED from config/prompts/rerank.md.

    It used to be built here, with a docstring saying it "mirrors" that file —
    two copies of one prompt kept in step by hand, with the version number
    living in a comment. That is the drift CLAUDE.md rule 17 forbids, and
    `tests/security/test_prompts_are_files.py` now makes the rule structural
    rather than remembered.

    The recorded CI fixtures were captured against this exact rendering, which
    is why they carry the prompt version: bumping `version:` in the file is what
    declares them stale.
    """
    listing = "\n".join(
        f"[{index}] {candidate.tier1_summary}" for index, candidate in enumerate(candidates)
    )
    return load_prompt("rerank").render(question=question, candidates=listing)


class Reranker:
    """Stage C. One batched call per question, or a recorded reply in CI."""

    def __init__(
        self,
        *,
        gateway: GatewayClient | None = None,
        config: ScoringConfig | None = None,
        recorded: dict[str, str] | None = None,
    ) -> None:
        self._config = config or scoring_config()
        self._gateway = gateway or GatewayClient.from_env()
        #: question_id -> raw response. Supplied in CI, where no model runs.
        self._recorded = recorded or {}

    @property
    def enabled(self) -> bool:
        return self._config.rerank.enabled

    async def score(
        self,
        *,
        question_id: str,
        question_text: str,
        candidates: list[CandidateInput],
        client: httpx.AsyncClient | None = None,
    ) -> RerankOutcome:
        if not self.enabled:
            return RerankOutcome(scores=None, reason="rerank disabled in config")
        if not candidates:
            return RerankOutcome(scores=None, reason="no candidates to rerank")

        if question_id in self._recorded:
            return parse_rerank_response(
                self._recorded[question_id], candidate_count=len(candidates)
            )

        prompt = build_rerank_prompt(question_text, candidates)
        try:
            content = await asyncio.wait_for(
                self._complete(prompt, client=client),
                timeout=self._config.rerank.timeout_seconds,
            )
        except TimeoutError:
            return RerankOutcome(
                scores=None,
                reason=f"rerank timed out after {self._config.rerank.timeout_seconds}s",
            )
        except GatewayError as exc:
            return RerankOutcome(scores=None, reason=f"gateway error: {exc}")

        return parse_rerank_response(content, candidate_count=len(candidates))

    async def _complete(self, prompt: str, *, client: httpx.AsyncClient | None) -> str:
        owned = client is None
        http = client or httpx.AsyncClient(timeout=self._config.rerank.timeout_seconds)
        try:
            response = await http.post(
                f"{self._gateway.base_url.rstrip('/')}/v1/chat/completions",
                json={
                    "model": self._config.rerank.alias,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": self._config.rerank.temperature,
                    "max_tokens": 400,
                },
                headers={"Authorization": f"Bearer {self._gateway.api_key}"},
            )
        except httpx.HTTPError as exc:
            raise GatewayError(f"rerank request failed: {exc}") from exc
        finally:
            if owned:
                await http.aclose()

        if response.status_code >= 300:
            raise GatewayError(f"rerank returned {response.status_code}: {response.text[:300]}")
        payload = response.json()
        try:
            return str(payload["choices"][0]["message"]["content"])
        except (KeyError, IndexError, TypeError) as exc:
            raise GatewayError(f"rerank response has no message content: {exc}") from exc


async def gather_candidates(
    session: AsyncSession,
    *,
    question_id: str,
    embedding: list[float],
    domain: str,
    requesting_customer: str,
    today: date,
    k: int | None = None,
) -> list[CandidateInput]:
    """Stage A. Vector search, then the graph facts stage B needs.

    Confidentiality, supersession and domain are all applied inside the query
    (Phase 2 amendment A), so nothing here can widen what the caller may see.
    """
    resolved = scoring_config()
    top_k = k if k is not None else resolved.retrieval.top_k

    hits = await queries.find_similar_questions(
        session,
        embedding=embedding,
        domain=domain,
        requesting_customer=requesting_customer,
        k=top_k,
    )

    candidates: list[CandidateInput] = []
    for hit in hits:
        for answer_id in hit.answer_ids:
            lineage = await queries.get_answer_with_lineage(session, answer_id=answer_id)
            if lineage is None or lineage.superseded:
                continue
            candidates.append(
                CandidateInput(
                    question_id=question_id,
                    matched_question_id=hit.question_id,
                    answer_node_id=lineage.answer_id,
                    tier1_summary=lineage.text[:400],
                    vector_score=hit.score,
                    outcome=Outcome(lineage.outcome),
                    age_days=_age_days(lineage.answer_date, today=today),
                    has_evidence=bool(lineage.evidence_codes),
                    superseded=lineage.superseded,
                    confidential=lineage.confidential,
                    sme_id=lineage.sme_id,
                )
            )
    return candidates


async def retrieve(
    session: AsyncSession,
    *,
    question_id: str,
    question_text: str,
    embedding: list[float],
    requesting_customer: str,
    calibration: CalibrationArtifact | None = None,
    domain: str = "cloud_migration",
    today: date | None = None,
    reranker: Reranker | None = None,
    config: ScoringConfig | None = None,
) -> tuple[RetrievalResult, RetrievalTrace]:
    """Stages A to D for one question.

    Returns the result and a trace explaining it, because a ranking nobody can
    account for is a ranking nobody should act on.

    `calibration` defaults to loading and verifying the artifact from disk. The
    default is a LOAD, not a fallback: if the artifact is absent, or was measured
    against a different embedding model or a different corpus, the load raises
    and retrieval refuses to run. Amendment J — there is no path from here to a
    ranking produced without calibration.
    """
    resolved = config or scoring_config()
    artifact = calibration if calibration is not None else load_for_current_corpus()
    run_date = today or date.today()
    trace = RetrievalTrace(question_id=question_id)

    expected_dim = embedding_config().model.dimensions
    if len(embedding) != expected_dim:
        raise ValueError(f"embedding has {len(embedding)} dimensions, expected {expected_dim}")

    candidates = await gather_candidates(
        session,
        question_id=question_id,
        embedding=embedding,
        domain=domain,
        requesting_customer=requesting_customer,
        today=run_date,
    )
    trace.candidates_considered = len(candidates)

    # Only the strongest few are worth a rerank call; ranking them by the
    # pre-rerank blend keeps the choice deterministic.
    prelim = score_candidates(candidates, calibration=artifact, rerank_scores=None, config=resolved)
    order = {c.answer_node_id: i for i, c in enumerate(prelim)}
    candidates.sort(key=lambda c: order.get(c.answer_node_id, len(order)))
    survivors = candidates[: resolved.retrieval.rerank_top_n]

    rerank_scores: dict[int, float] | None = None
    if survivors:
        active = reranker or Reranker(config=resolved)
        result = await active.score(
            question_id=question_id, question_text=question_text, candidates=survivors
        )
        if result.usable:
            rerank_scores = result.scores
            trace.reranked = True
        else:
            trace.rerank_skip_reason = result.reason
            trace.notes.append(f"rerank not applied ({result.reason}); rerank weight redistributed")

    scored = score_candidates(
        survivors, calibration=artifact, rerank_scores=rerank_scores, config=resolved
    )
    return to_retrieval_result(question_id, scored, calibration=artifact), trace
