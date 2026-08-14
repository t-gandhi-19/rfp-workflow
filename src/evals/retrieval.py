"""The retrieval eval category (build prompt §3, Phase 3 step 5).

Runs the twenty golden questions through the real pipeline — embed with the
query prefix, vector search, graph facts, calibrate, rerank if enabled, score —
and judges the result against the hand-written answer key.

WHAT IS ZERO TOLERANCE, and why each one is not a percentage:

    refusal gate          every unanswerable must NO_MATCH and no answerable
                          may. Both directions are failures: the first makes the
                          drafter answer without support, the second wastes an
                          SME on a question the corpus can already answer.
                          Baits are ADVISORY — see `_check_no_match_set` for why
                          the literal set {2.7, 2.8, 3.5} could not be the gate.
    staleness             a superseded answer in any candidate list is a wrong
                          answer presented as a current one. The chain heads
                          must be retrieved and their predecessors never.
    confidentiality       ANS-0014 in any Meridian candidate list is a leak.
    paraphrase exclusion  amendment P. A paraphrase carries no answer of its
                          own, so it can only duplicate a candidate already
                          present.

Recall@5 is gated at 0.8, rank-1 accuracy is gated by ratchet, and MRR is
REPORTED. The two gates answer different questions and neither subsumes the
other:

    Recall@5           did the right answer reach the drafter AT ALL?
    rank1_accuracy     was it the drafter's PRIMARY SOURCE — the one cited, and
                       the one the confidence formula keys on?

Recall@5 is structurally blind to the second. On golden 1.1 with rerank off it
reported 1.0000 while the primary source was a WON answer on a neighbouring
topic, ahead of the one the key calls the only direct answer. MRR describes how
comfortably the set ranked, which is worth watching and not worth failing a
build over.

ATTRIBUTION IS PART OF THE RESULT, not a debugging aid. Every candidate carries
raw -> calibrated -> preference -> final, because a ranking nobody can account
for is a ranking nobody should act on, and because the three-column comparison
in the PR is only meaningful if each column can be traced to the arithmetic that
produced it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

from neo4j import AsyncSession
from pydantic import BaseModel, ConfigDict, Field

from src.contracts import RetrievalStatus, ScoredCandidate
from src.contracts.embedding import EmbedRole, embedding_config
from src.contracts.thresholds import ScoringConfig, scoring_config
from src.evals.contracts import (
    CategoryResult,
    CategoryStatus,
    EvalMetric,
    MetricDirection,
    Violation,
)
from src.gateway.client import GatewayClient
from src.gateway.fake_embedder import fake_embeddings, fake_embeddings_enabled
from src.retrieval.calibration import BAITS, UNANSWERABLE, CalibrationArtifact
from src.retrieval.retriever import Reranker, retrieve

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURES = REPO_ROOT / "fixtures"

#: The golden RFP's issuer. Every question is retrieved as this customer, which
#: is what makes the Bluepine confidentiality check meaningful.
MERIDIAN = "Meridian Insurance Group"

#: Recall@5 must clear this over the expected matches. Zero tolerance in the
#: sense that it does not move: it is a floor on the eval, not a tuning knob.
RECALL_AT_5_THRESHOLD = 0.8

#: How deep "Recall@5" looks.
RECALL_K = 5

#: RANK-1 PRIMARY-SOURCE ACCURACY. Commissioned, and gated by ratchet.
#:
#: WHY IT EXISTS. Recall@5 is structurally blind to a rank-1 error whenever the
#: right answer is anywhere in the top five — and rank 1 is not one candidate
#: among five, it is the DRAFTER'S PRIMARY SOURCE, the one the citation names
#: and the one the confidence formula keys on.
#:
#: That blindness was not hypothetical. On golden 1.1, with rerank off, ANS-0032
#: (a WON RACI answer, preference x1.1952) outranked ANS-0037 — which the
#: hand-written key calls "the only direct team-model answer" — because
#: preference overturned a relevance gap of 1.0000 vs 0.8247. Recall@5 reported
#: 1.0000 throughout. The flip was noticed only by the preference-decisive
#: diagnostic, which is a diagnostic and gates nothing.
#:
#: COMMISSIONED at 1.0 — 15 of 15 — measured in the SHIPPED configuration
#: (rerank enabled) on 2026-08-13, corpus a3d1eea3a24d6cb7, against
#: nomic-embed-text:v1.5. This commissions a gate the system currently passes,
#: which is the legitimate case for a ratchet: it defends a property that holds
#: now against a future that erodes it.
#:
#: A NON-SHIPPED ARM MAY FAIL THIS, and that is the gate working rather than a
#: false positive: the rerank-off arm scores 14/15 precisely because of the 1.1
#: inversion. The category's verdict is a claim about the shipped configuration.
COMMISSIONED_RANK1_ACCURACY = 1.0

#: The confidential answer. Reachable to Bluepine, never to Meridian.
CONFIDENTIAL_ANSWER_ID = "ANS-0014"

#: Supersession chains: the head that must be retrieved, and the predecessor
#: that must never be. Transcribed from the corpus rather than derived at run
#: time on purpose — an eval that computes its own expectations from the data it
#: is judging cannot fail when the data is wrong.
CHAIN_HEADS = ("ANS-0025", "ANS-0006", "ANS-0010", "ANS-0022")
CHAIN_SUPERSEDED = ("ANS-0024", "ANS-0005", "ANS-0009", "ANS-0021")


class CandidateAttribution(BaseModel):
    """One candidate's full decomposition. Rendered per question in the report."""

    model_config = ConfigDict(extra="forbid")

    rank: int = Field(ge=1)
    answer_id: str
    matched_question_id: str | None = None
    raw_cosine: float
    calibrated: float
    preference: float
    final: float
    relevance: float
    cleared_floor: bool


class QuestionOutcome(BaseModel):
    """What the pipeline did with one golden question."""

    model_config = ConfigDict(extra="forbid")

    number: str
    kind: str
    expected_answer_id: str | None
    status: RetrievalStatus
    #: 1-based position of the expected answer, or None if it never appeared.
    rank_of_expected: int | None
    candidates: list[CandidateAttribution]
    floor_used: float
    #: The preference-decisive diagnostic: does rank 1 change with preference
    #: switched off? Reported from day one because D17's whole design rests on
    #: preference being a nudge, and nobody had measured how often it decides.
    rank1_with_preference: str | None
    rank1_without_preference: str | None

    @property
    def preference_decisive(self) -> bool:
        return (
            self.rank1_with_preference is not None
            and self.rank1_with_preference != self.rank1_without_preference
        )

    @property
    def hit_at_k(self) -> bool:
        return self.rank_of_expected is not None and self.rank_of_expected <= RECALL_K


@dataclass
class RetrievalRun:
    """Everything measured, before it is turned into metrics."""

    outcomes: list[QuestionOutcome] = field(default_factory=list)
    violations: list[Violation] = field(default_factory=list)

    @property
    def expected_matches(self) -> list[QuestionOutcome]:
        return [o for o in self.outcomes if o.expected_answer_id and o.kind == "answerable"]

    @property
    def recall_at_5(self) -> float:
        expected = self.expected_matches
        if not expected:
            return 0.0
        return sum(1 for o in expected if o.hit_at_k) / len(expected)

    @property
    def mrr(self) -> float:
        """Mean reciprocal rank over the expected matches. Reported, not gated."""
        expected = self.expected_matches
        if not expected:
            return 0.0
        total = sum(1.0 / o.rank_of_expected if o.rank_of_expected else 0.0 for o in expected)
        return total / len(expected)

    @property
    def rank1_accuracy(self) -> float:
        """How often the DRAFTER'S PRIMARY SOURCE is the expected answer.

        Not a softer Recall@1: it is the metric that sees what Recall@5 cannot,
        which is a wrong answer at rank 1 with the right one just behind it.
        """
        expected = self.expected_matches
        if not expected:
            return 0.0
        return sum(1 for o in expected if o.rank_of_expected == 1) / len(expected)

    @property
    def rank1_misses(self) -> list[QuestionOutcome]:
        """The questions whose primary source is not the expected answer."""
        return [o for o in self.expected_matches if o.rank_of_expected != 1]

    @property
    def no_match_numbers(self) -> set[str]:
        return {o.number for o in self.outcomes if o.status is RetrievalStatus.NO_MATCH}

    @property
    def preference_decisive_rate(self) -> float:
        answerable = [o for o in self.outcomes if o.kind == "answerable"]
        if not answerable:
            return 0.0
        return sum(1 for o in answerable if o.preference_decisive) / len(answerable)


def golden_questions() -> list[dict[str, Any]]:
    """The twenty questions, from the BUILDER-IMMUTABLE hand-written key (D16).

    Read, never written. It is the independent ground truth this eval is judged
    against, so an eval that could edit it would be grading its own homework.
    """
    with (FIXTURES / "answer_key_manual.json").open(encoding="utf-8") as handle:
        key: dict[str, Any] = json.load(handle)
    return list(key["questions"])


def classify(number: str, expected_answer_id: str | None) -> str:
    if number in UNANSWERABLE:
        return "unanswerable"
    if number in BAITS:
        return "bait"
    return "answerable" if expected_answer_id else "other"


def paraphrase_question_ids() -> set[str]:
    with (FIXTURES / "question_paraphrases.json").open(encoding="utf-8") as handle:
        return {row["question_id"] for row in json.load(handle)}


async def embed_queries(texts: list[str]) -> list[list[float]]:
    """Query-side embeddings, through the gateway or the CI stand-in.

    The QUERY prefix, never the document one. They are not interchangeable: the
    calibration artifact's geometry is `query_x_document`, so embedding these as
    documents would compare a distribution the floor was never derived over.
    """
    config = embedding_config()
    if fake_embeddings_enabled():
        return fake_embeddings(texts, config.model.dimensions, role=EmbedRole.QUERY)

    client = GatewayClient.from_env()
    vectors: list[list[float]] = []
    batch = 16
    for start in range(0, len(texts), batch):
        vectors.extend(
            await client.embed(
                texts[start : start + batch], alias=config.model.alias, role=EmbedRole.QUERY
            )
        )
    return vectors


def _rank1_without_preference(candidates: list[ScoredCandidate]) -> str | None:
    """Who would win on relevance alone.

    The same total order the scorer uses — descending score, then answer id —
    so the comparison isolates preference and not a tie-breaking difference.
    """
    if not candidates:
        return None
    ordered = sorted(candidates, key=lambda c: (-c.relevance, c.answer_node_id))
    return ordered[0].answer_node_id


def _attributions(candidates: list[ScoredCandidate], *, floor: float) -> list[CandidateAttribution]:
    return [
        CandidateAttribution(
            rank=index + 1,
            answer_id=candidate.answer_node_id,
            matched_question_id=None,
            raw_cosine=candidate.vector_score,
            calibrated=candidate.calibrated_similarity,
            preference=candidate.preference,
            final=candidate.final_score,
            relevance=candidate.relevance,
            cleared_floor=candidate.relevance >= floor,
        )
        for index, candidate in enumerate(candidates)
    ]


async def run_retrieval_eval(
    session: AsyncSession,
    *,
    calibration: CalibrationArtifact,
    reranker: Reranker | None = None,
    today: date | None = None,
    config: ScoringConfig | None = None,
) -> RetrievalRun:
    """Every golden question through the full pipeline, judged against the key."""
    resolved = config or scoring_config()
    questions = golden_questions()
    vectors = await embed_queries([q["text"] for q in questions])
    paraphrases = paraphrase_question_ids()

    run = RetrievalRun()
    for question, embedding in zip(questions, vectors, strict=True):
        number = question["number"]
        expected = question["expected_best_match_answer_id"] or None
        kind = classify(number, expected)

        result, _trace = await retrieve(
            session,
            question_id=f"GQ-{number}",
            question_text=question["text"],
            embedding=embedding,
            requesting_customer=MERIDIAN,
            calibration=calibration,
            today=today,
            reranker=reranker,
            config=resolved,
        )

        candidates = result.candidates
        answer_ids = [c.answer_node_id for c in candidates]
        rank = answer_ids.index(expected) + 1 if expected and expected in answer_ids else None

        run.outcomes.append(
            QuestionOutcome(
                number=number,
                kind=kind,
                expected_answer_id=expected,
                status=result.status,
                rank_of_expected=rank,
                candidates=_attributions(candidates, floor=result.floor_used),
                floor_used=result.floor_used,
                rank1_with_preference=answer_ids[0] if answer_ids else None,
                rank1_without_preference=_rank1_without_preference(candidates),
            )
        )

        # Zero-tolerance checks that are properties of a CANDIDATE LIST rather
        # than of the ranking, so they are evaluated per question as it runs.
        for stale in CHAIN_SUPERSEDED:
            if stale in answer_ids:
                run.violations.append(
                    Violation(
                        rule="staleness",
                        question_number=number,
                        detail=f"superseded answer {stale} appeared in the candidate list",
                    )
                )
        if CONFIDENTIAL_ANSWER_ID in answer_ids:
            run.violations.append(
                Violation(
                    rule="confidentiality",
                    question_number=number,
                    detail=(
                        f"{CONFIDENTIAL_ANSWER_ID} is confidential to Bluepine and reached a "
                        f"candidate list retrieved as {MERIDIAN}"
                    ),
                )
            )
        leaked = sorted(paraphrases.intersection(answer_ids))
        if leaked:
            run.violations.append(
                Violation(
                    rule="paraphrase_exclusion",
                    question_number=number,
                    detail=f"paraphrase ids in the candidate list: {leaked}",
                )
            )

    _check_no_match_set(run)
    _check_chain_heads(run)
    return run


def _check_no_match_set(run: RetrievalRun) -> None:
    """The refusal gate, over the two populations it is a claim about.

    GATED, both directions, because both are real harms:

        an unanswerable that MATCHED    the drafter answers a question the
                                        corpus cannot support, instead of
                                        escalating it
        an answerable that NO_MATCHED   the drafter escalates a question the
                                        corpus can answer, wasting an SME

    BAITS ARE NOT GATED, and that is a reconciliation rather than a loophole.
    "NO_MATCH exactly {2.7, 2.8, 3.5}" and the commissioned bait margins cannot
    both hold: `check_floor_discrimination` commissioned 3.4 at 0.3750 and 4.2
    at 0.0440 BELOW the floor, and a candidate below the floor is a refusal by
    definition. Three places already record baits as advisory — `BAITS` in
    `src.retrieval.calibration`, the commissioning baselines, and
    `probe_report`'s "ADVISORY ONLY, never gated" — and the only way to make the
    literal set pass would be to drop the floor below 0.5394 to admit 4.2, which
    is a derived constant moving to fit a test.

    So the baits are MEASURED and REPORTED against their commissioned margins,
    and they gate nothing. 3.4 is refused on legal grounds regardless of what
    retrieval does, and 4.2 belongs to Phase 4's pricing block.
    """
    by_number = {outcome.number: outcome for outcome in run.outcomes}

    for number in UNANSWERABLE:
        outcome = by_number.get(number)
        if outcome is None:
            run.violations.append(
                Violation(
                    rule="no_match_set",
                    question_number=number,
                    detail="expected an unanswerable probe, but the key does not contain it",
                )
            )
        elif outcome.status is not RetrievalStatus.NO_MATCH:
            run.violations.append(
                Violation(
                    rule="no_match_set",
                    question_number=number,
                    detail=(
                        "the corpus cannot answer this question, but retrieval MATCHED it — "
                        "the drafter would answer instead of escalating"
                    ),
                )
            )

    for outcome in run.outcomes:
        if outcome.kind == "answerable" and outcome.status is RetrievalStatus.NO_MATCH:
            run.violations.append(
                Violation(
                    rule="no_match_set",
                    question_number=outcome.number,
                    detail=(
                        f"retrieval refused a question the corpus can answer "
                        f"(expected {outcome.expected_answer_id}) — the drafter would escalate "
                        f"instead of answering"
                    ),
                )
            )


def _check_chain_heads(run: RetrievalRun) -> None:
    """Every supersession head reaches at least one candidate list.

    The mirror of the staleness rule. Suppressing the superseded answer is only
    half the requirement: if the head never surfaces either, retrieval has lost
    the answer rather than kept the current one.
    """
    seen = {attribution.answer_id for outcome in run.outcomes for attribution in outcome.candidates}
    for head in CHAIN_HEADS:
        if head not in seen:
            run.violations.append(
                Violation(
                    rule="staleness",
                    detail=(
                        f"supersession head {head} never appeared in any candidate list; "
                        "the current answer was lost, not merely the stale one suppressed"
                    ),
                )
            )


def _refused_baits(run: RetrievalRun) -> list[str]:
    return sorted(
        outcome.number
        for outcome in run.outcomes
        if outcome.kind == "bait" and outcome.status is RetrievalStatus.NO_MATCH
    )


def to_category_result(run: RetrievalRun) -> CategoryResult:
    """Turn the measurements into the harness's common shape."""
    no_match = sorted(run.no_match_numbers)
    metrics = [
        EvalMetric(
            key="recall_at_5",
            label="Recall@5 over the expected matches",
            value=round(run.recall_at_5, 4),
            direction=MetricDirection.HIGHER_IS_BETTER,
            threshold=RECALL_AT_5_THRESHOLD,
            passed=run.recall_at_5 >= RECALL_AT_5_THRESHOLD,
            detail=f"{sum(1 for o in run.expected_matches if o.hit_at_k)}/"
            f"{len(run.expected_matches)} expected answers in the top {RECALL_K}",
        ),
        EvalMetric(
            key="rank1_accuracy",
            label="Rank-1 primary-source accuracy",
            value=round(run.rank1_accuracy, 4),
            direction=MetricDirection.HIGHER_IS_BETTER,
            threshold=COMMISSIONED_RANK1_ACCURACY,
            passed=run.rank1_accuracy >= COMMISSIONED_RANK1_ACCURACY,
            detail=(
                f"{sum(1 for o in run.expected_matches if o.rank_of_expected == 1)}/"
                f"{len(run.expected_matches)} — rank 1 is the drafter's primary source and "
                f"what the confidence formula keys on, which Recall@5 cannot see. "
                f"Ratchet: commissioned at {COMMISSIONED_RANK1_ACCURACY} in the shipped "
                f"configuration; any regression fails."
                + (
                    f" MISSES: {
                        ', '.join(
                            f'{o.number} (expected {o.expected_answer_id}, got '
                            f'{o.rank1_with_preference} at rank 1)'
                            for o in run.rank1_misses
                        )
                    }"
                    if run.rank1_misses
                    else ""
                )
            ),
        ),
        EvalMetric(
            key="mrr",
            label="Mean reciprocal rank",
            value=round(run.mrr, 4),
            direction=MetricDirection.HIGHER_IS_BETTER,
            detail="reported, not gated — Recall@5 and rank-1 accuracy carry the pass/fail",
        ),
        EvalMetric(
            key="no_match_gate",
            label="Unanswerables refuse; answerables do not",
            value=float(len([v for v in run.violations if v.rule == "no_match_set"])),
            direction=MetricDirection.MUST_BE_ZERO,
            threshold=0.0,
            passed=not any(v.rule == "no_match_set" for v in run.violations),
            detail=(
                f"observed NO_MATCH: {{{', '.join(no_match) or 'none'}}}; "
                f"gated over {', '.join(UNANSWERABLE)} and the answerables. "
                f"Baits {', '.join(BAITS)} are advisory — see the row below."
            ),
        ),
        EvalMetric(
            key="baits_refused",
            label=f"Baits refused (advisory, never gated): {', '.join(BAITS)}",
            value=float(len(_refused_baits(run))),
            direction=MetricDirection.HIGHER_IS_BETTER,
            detail=(
                f"refused: {{{', '.join(_refused_baits(run)) or 'none'}}}. "
                "Commissioned below the floor at 3.4 +0.3750 and 4.2 +0.0440, so refusal is "
                "the expected landing. 3.4 is refused on legal grounds regardless of "
                "retrieval; 4.2 belongs to Phase 4's pricing block."
            ),
        ),
        EvalMetric(
            key="staleness",
            label="Superseded answers retrieved",
            value=float(len([v for v in run.violations if v.rule == "staleness"])),
            direction=MetricDirection.MUST_BE_ZERO,
            threshold=0.0,
            passed=not any(v.rule == "staleness" for v in run.violations),
            detail=f"heads {', '.join(CHAIN_HEADS)} must appear; "
            f"{', '.join(CHAIN_SUPERSEDED)} must not",
        ),
        EvalMetric(
            key="confidentiality",
            label=f"{CONFIDENTIAL_ANSWER_ID} in a {MERIDIAN} candidate list",
            value=float(len([v for v in run.violations if v.rule == "confidentiality"])),
            direction=MetricDirection.MUST_BE_ZERO,
            threshold=0.0,
            passed=not any(v.rule == "confidentiality" for v in run.violations),
        ),
        EvalMetric(
            key="paraphrase_exclusion",
            label="Paraphrase ids in any candidate list (amendment P)",
            value=float(len([v for v in run.violations if v.rule == "paraphrase_exclusion"])),
            direction=MetricDirection.MUST_BE_ZERO,
            threshold=0.0,
            passed=not any(v.rule == "paraphrase_exclusion" for v in run.violations),
        ),
        EvalMetric(
            key="preference_decisive_rate",
            label="Questions where preference changes rank 1",
            value=round(run.preference_decisive_rate, 4),
            direction=MetricDirection.LOWER_IS_BETTER,
            detail=(
                "diagnostic. D17 intends preference as a nudge among qualifying "
                "candidates; this is how often it actually decides the winner"
            ),
        ),
    ]
    passed = all(metric.passed for metric in metrics if metric.passed is not None)
    return CategoryResult(
        key="retrieval",
        label="Retrieval",
        status=CategoryStatus.PASS if passed else CategoryStatus.FAIL,
        metrics=metrics,
        violations=run.violations,
        notes=[
            "Judged against fixtures/answer_key_manual.json, which this eval reads "
            "and never writes (D16).",
            "Every candidate carries raw -> calibrated -> preference -> final; the "
            "per-question attribution is rendered below the summary.",
        ],
    )


# ---------------------------------------------------------------------------
# Presentation
#
# LIVES HERE, not in a script, because two entry points render the same run:
# `scripts.eval_retrieval` (the step 5 runner) and `scripts.evals` (the step 8
# harness). It was written in the first and, when the harness arrived, the
# harness simply did not print it — so `make evals` reported category verdicts
# with no attribution behind them, and the numbers a PR quotes came from a
# different command than the one the Makefile documents.
#
# Attribution is part of the result, not a debugging aid: a ranking nobody can
# account for is a ranking nobody should act on.
# ---------------------------------------------------------------------------


def format_run(run: RetrievalRun) -> str:
    """The full per-question rendering: metrics, misses, three columns, attribution."""
    result = to_category_result(run)
    lines = ["", "retrieval eval", ""]
    for metric in result.metrics:
        verdict = "" if metric.passed is None else ("  PASS" if metric.passed else "  FAIL")
        gate = "" if metric.threshold is None else f"  (threshold {metric.threshold})"
        lines.append(f"  {metric.label:<52} {metric.value:>8.4f}{gate}{verdict}")
        if metric.detail:
            lines.append(f"      {metric.detail}")
    lines.append("")

    if run.rank1_misses:
        lines.append("  RANK-1 PRIMARY-SOURCE MISSES (what Recall@5 cannot see):")
        for miss in run.rank1_misses:
            lines.append(
                f"    {miss.number}  expected {miss.expected_answer_id} at rank 1, "
                f"got {miss.rank1_with_preference} (rank {miss.rank_of_expected} without "
                f"preference: {miss.rank1_without_preference})"
            )
        lines.append("")

    lines.append("  per question:")
    lines.append(
        f"    {'no.':<5} {'kind':<13} {'status':<9} {'expected':<10} {'rank':<5} "
        f"{'cands':<6} pref-decisive"
    )
    for outcome in run.outcomes:
        rank = "-" if outcome.rank_of_expected is None else str(outcome.rank_of_expected)
        lines.append(
            f"    {outcome.number:<5} {outcome.kind:<13} {outcome.status.value:<9} "
            f"{outcome.expected_answer_id or '-':<10} {rank:<5} "
            f"{len(outcome.candidates):<6} {'YES' if outcome.preference_decisive else 'no'}"
        )
    lines.append("")

    # THE THREE COLUMNS: relevance, preference, final — for the rank-1 candidate
    # of every question. This is where D17's split is legible per question: the
    # first column is what qualifies a candidate, the second is what reorders
    # among qualifying ones, and the third is what the drafter actually sees.
    lines.append("  rank 1 — the drafter's primary source (relevance x preference = final):")
    lines.append(f"    {'no.':<5} {'answer':<10} {'relevance':>10} {'preference':>11} {'final':>9}")
    for outcome in run.outcomes:
        if not outcome.candidates:
            lines.append(f"    {outcome.number:<5} {'(none)':<10}")
            continue
        top = outcome.candidates[0]
        lines.append(
            f"    {outcome.number:<5} {top.answer_id:<10} {top.relevance:>10.4f} "
            f"{top.preference:>11.4f} {top.final:>9.4f}"
        )
    lines.append("")

    lines.append("  attribution — top 3 per question (raw -> calibrated -> pref -> final):")
    for outcome in run.outcomes:
        lines.append(f"    {outcome.number}  floor {outcome.floor_used:.4f}")
        if not outcome.candidates:
            lines.append("      (no candidates)")
        for attribution in outcome.candidates[:3]:
            flag = "clears" if attribution.cleared_floor else "BELOW "
            lines.append(
                f"      {attribution.rank}. {attribution.answer_id:<10} "
                f"raw {attribution.raw_cosine:.4f}  cal {attribution.calibrated:.4f}  "
                f"rel {attribution.relevance:.4f}  pref {attribution.preference:.4f}  "
                f"final {attribution.final:.4f}  {flag}"
            )
    lines.append("")

    if result.violations:
        lines.append("  VIOLATIONS:")
        for violation in result.violations:
            where = f" [{violation.question_number}]" if violation.question_number else ""
            lines.append(f"    {violation.rule}{where}: {violation.detail}")
        lines.append("")
    else:
        lines.append("  no zero-tolerance violations.")
        lines.append("")
    return "\n".join(lines)
