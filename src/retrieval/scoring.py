"""Stages B to D: deterministic scoring arithmetic (build prompt §10).

No model is involved here and no configuration is hardcoded. Every weight,
multiplier and floor comes from `config/scoring.yaml`, so tuning retrieval is a
config change with a test-visible effect rather than a code change.

The ordering these functions produce is the thing the retrieval evals measure,
so two properties matter more than the exact numbers:

* **Determinism.** Identical inputs give identical scores and identical order.
  Ties break on `answer_node_id`, because Python's sort is stable and would
  otherwise preserve whatever arbitrary order the database happened to return.
* **Explainability.** Each candidate keeps its decomposition — vector score,
  multiplier, rerank score, final — so a ranking can be justified by pointing at
  the arithmetic instead of at a model's opinion.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from src.contracts import CandidateFlags, Outcome, RetrievalResult, RetrievalStatus, ScoredCandidate
from src.contracts.thresholds import ScoringConfig, scoring_config


@dataclass(frozen=True)
class CandidateInput:
    """One vector hit plus the graph facts stage B needs.

    Assembled by the retriever from `find_similar_questions` and the answer
    lineage; kept separate from `ScoredCandidate` so the arithmetic is testable
    without a database.
    """

    #: The question being ANSWERED — the one from the incoming RFP. Not the
    #: corpus question this answer was originally written for; the contract
    #: requires every candidate in a RetrievalResult to name the question it was
    #: scored for, and conflating the two makes a result claim it answers a
    #: historical question nobody asked.
    question_id: str
    #: The corpus question the vector search actually matched. Kept for the
    #: report, so a ranking can be traced back to what it resembled.
    matched_question_id: str | None
    answer_node_id: str
    tier1_summary: str
    vector_score: float
    outcome: Outcome
    age_days: int
    has_evidence: bool
    superseded: bool = False
    confidential: bool = False
    sme_id: str | None = None


def recency_multiplier(age_days: int, *, config: ScoringConfig | None = None) -> float:
    """exp(-age_days / decay_days): 1.0 today, decaying smoothly with age.

    Smooth rather than banded on purpose — a threshold would make one day's
    difference flip a ranking, and "why did this drop out overnight?" is a
    question with no satisfying answer.
    """
    resolved = config or scoring_config()
    if age_days < 0:
        raise ValueError(f"age_days must not be negative, got {age_days}")
    return math.exp(-age_days / resolved.graph_multiplier.recency.decay_days)


def graph_multiplier(
    *,
    outcome: Outcome,
    age_days: int,
    has_evidence: bool,
    config: ScoringConfig | None = None,
) -> float:
    """What the graph knows that embedding similarity cannot.

    Outcome, recency and evidence, multiplied. An answer that won a real bid,
    was written recently, and is backed by a case study should outrank an
    equally-similar answer that lost two years ago and cites nothing.
    """
    resolved = config or scoring_config()
    multipliers = resolved.graph_multiplier

    outcome_factor = {
        Outcome.WON: multipliers.outcome.won,
        Outcome.LOST: multipliers.outcome.lost,
        Outcome.UNKNOWN: multipliers.outcome.unknown,
    }[outcome]

    evidence_factor = multipliers.evidence_bonus if has_evidence else 1.0
    return outcome_factor * recency_multiplier(age_days, config=resolved) * evidence_factor


def blend(
    *,
    vector_score: float,
    multiplier: float,
    rerank_score: float | None,
    config: ScoringConfig | None = None,
) -> float:
    """Stage D. Combine the boosted similarity with the rerank score.

    The boosted similarity is clamped to [0, 1] before blending. The multiplier
    can exceed 1 (won x recent x evidenced reaches ~1.27), and an unclamped
    product would push `final_score` outside the range the contract permits —
    and, worse, would let the multiplier rather than the similarity dominate.

    When there is no rerank score — disabled in config, or the response was
    unusable for this question — the rerank weight is **redistributed** onto the
    boosted similarity rather than treated as a zero. A missing opinion is not
    an opinion of zero: scoring it as one would drag every candidate down
    uniformly and make the match floor unreachable.
    """
    resolved = config or scoring_config()
    weights = resolved.final_score.weights

    boosted = min(1.0, max(0.0, vector_score * multiplier))
    if rerank_score is None:
        return boosted

    total = weights.vector_graph + weights.rerank
    if total <= 0:
        raise ValueError("final_score weights must sum to something positive")
    return (weights.vector_graph * boosted + weights.rerank * rerank_score) / total


def score_candidates(
    candidates: list[CandidateInput],
    *,
    rerank_scores: dict[int, float] | None = None,
    config: ScoringConfig | None = None,
) -> list[ScoredCandidate]:
    """Apply stages B and D to every candidate and rank the result.

    `rerank_scores` is keyed by position in `candidates`. None means no rerank
    for this question at all — the weight redistribution in :func:`blend`
    applies uniformly, so candidates stay comparable with each other even though
    they are no longer comparable with a reranked question's scores.
    """
    resolved = config or scoring_config()

    scored: list[ScoredCandidate] = []
    for index, candidate in enumerate(candidates):
        multiplier = graph_multiplier(
            outcome=candidate.outcome,
            age_days=candidate.age_days,
            has_evidence=candidate.has_evidence,
            config=resolved,
        )
        rerank_score = None if rerank_scores is None else rerank_scores.get(index)
        final = blend(
            vector_score=candidate.vector_score,
            multiplier=multiplier,
            rerank_score=rerank_score,
            config=resolved,
        )
        scored.append(
            ScoredCandidate(
                question_id=candidate.question_id,
                answer_node_id=candidate.answer_node_id,
                tier1_summary=candidate.tier1_summary,
                vector_score=candidate.vector_score,
                graph_multiplier=multiplier,
                rerank_score=rerank_score,
                final_score=final,
                flags=CandidateFlags(
                    superseded=candidate.superseded,
                    outcome=candidate.outcome,
                    confidential=candidate.confidential,
                    recency_days=candidate.age_days,
                    sme_id=candidate.sme_id,
                ),
            )
        )

    # Descending score, then answer id — a total order, so identical inputs
    # always produce an identical ranking.
    return sorted(scored, key=lambda c: (-c.final_score, c.answer_node_id))


def to_retrieval_result(
    question_id: str,
    scored: list[ScoredCandidate],
    *,
    config: ScoringConfig | None = None,
) -> RetrievalResult:
    """Wrap ranked candidates with the MATCHED / NO_MATCH verdict.

    NO_MATCH is a legitimate outcome, not a failure. It is what obliges the
    drafter to escalate rather than stretch a weak candidate into an answer, and
    the eval scores it at zero tolerance in both directions.
    """
    resolved = config or scoring_config()
    floor = resolved.retrieval.match_floor
    cleared = [candidate for candidate in scored if candidate.final_score >= floor]
    return RetrievalResult(
        question_id=question_id,
        candidates=scored,
        status=RetrievalStatus.MATCHED if cleared else RetrievalStatus.NO_MATCH,
        floor_used=floor,
    )
