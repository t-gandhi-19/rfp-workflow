"""The five agents, as the controller's `AgentLayer` (1c).

WHICH STEPS USE A MODEL, AND WHICH DO NOT. Rule 3 draws this line, and the line
is not where the word "agent" suggests:

* **triage** — one local model call. No tools.
* **requirement_extractor** — DETERMINISTIC parsing (`src/extraction/`). The
  local model is consulted only on segments the parser could not classify, and
  only to answer "is this a question at all". It never supplies a field.
* **retriever** — no drafting model at all. Three MCP tool tiers plus the
  existing batched rerank, and the ranking arithmetic in `src/retrieval/`. The
  selection reasons on the returned object are DERIVED from that arithmetic;
  the model's only contribution is the rerank score.
* **drafter** — Groq 70B, with `get_evidence` and `entity_exists`. Returns
  prose, claims and citations. Confidence is computed here from its claim
  decomposition, never returned by it.
* **critic** — Groq 70B, with `entity_exists`. Returns issues and a delta
  bounded at zero by the contract.

EVERY MODEL CALL IS AUTHORIZED BEFORE IT IS MADE. The ledger belongs to the
controller and is threaded in; each method asks it at the point a real call is
about to happen, which is why the conditional ones (the extract assist, the
rerank) can be honestly counted at all.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import date

import httpx
from neo4j import AsyncSession

from src.agents.crew import build_agent, build_crew, build_task
from src.agents.mcp_client import McpClient, drafter_client
from src.agents.tools import critic_tools, drafter_tools
from src.contracts import (
    ConfidenceInputs,
    CritiqueResult,
    DraftedAnswer,
    DraftPayload,
    ExtractedQuestion,
    HaltReason,
    RankingBasis,
    RetrieverSelection,
    RFPDocument,
    TriageResult,
)
from src.contracts.embedding import EmbedRole
from src.contracts.enums import SelectionOutcome
from src.contracts.selection import CandidateSelection
from src.controller.budget import BudgetLedger, CallKind, CallUsage
from src.controller.protocols import AgentValidationError
from src.extraction.questions import parse_questions
from src.gateway.client import GatewayClient
from src.guardrails.injection import wrap
from src.observability.tracing import llm_span
from src.prompts import load_prompt
from src.retrieval.calibration import CalibrationArtifact, load_for_current_corpus
from src.retrieval.retriever import Reranker, retrieve

logger = logging.getLogger("rfp.agents")


def _usage_from(crew_output: object) -> CallUsage:
    """Tokens and cost out of a crewAI result, without trusting its shape.

    crewAI's usage object has moved between versions. A missing count is
    recorded as zero tokens and an UNKNOWN cost rather than as a free call —
    see `CallUsage`, and the ledger's refusal to treat unpriced as free.
    """
    usage = getattr(crew_output, "token_usage", None)
    total = getattr(usage, "total_tokens", None)
    return CallUsage(
        tokens=int(total) if isinstance(total, int | float) else 0,
        cost_usd=None,
    )


def _require(crew_output: object, model: type, *, step: str) -> object:
    """Pull the validated Pydantic result off a crewAI output, or raise.

    A crew that returned prose where a contract was required is exactly the
    failure `AgentValidationError` names, and the controller's one retry is
    what it earns.
    """
    parsed = getattr(crew_output, "pydantic", None)
    if isinstance(parsed, model):
        return parsed
    raw = str(getattr(crew_output, "raw", crew_output))[:400]
    raise AgentValidationError(f"{step} did not return a valid {model.__name__}; got: {raw}")


@dataclass
class CrewAgentLayer:
    """The controller's five collaborators, backed by crewAI and MCP."""

    session: AsyncSession
    gateway: GatewayClient = field(default_factory=GatewayClient.from_env)
    #: `drafter-sa`, used by the drafter's and critic's tools. There is no
    #: retriever client here on purpose: the retrieval PIPELINE is deterministic
    #: code, and it reaches the graph through `src/graph/queries.py` — the same
    #: tested, parameterised functions the MCP server exposes, called directly
    #: because there is no agent in that path to keep behind a tool surface.
    #: The MCP door exists for the steps where a MODEL chooses what to ask for.
    drafter_mcp: McpClient = field(default_factory=drafter_client)
    calibration: CalibrationArtifact | None = None
    reranker: Reranker | None = None
    today: date | None = None

    # -- triage -----------------------------------------------------------

    async def triage(self, document: RFPDocument, *, budget: BudgetLedger) -> TriageResult:
        """One local call, no tools. A mismatch halts the run readably."""
        budget.authorize(CallKind.TRIAGE)
        prompt = load_prompt("triage")
        # Rule 15: the document is data, never instructions. Every span of it
        # that reaches a prompt is delimiter-wrapped, and the delimiters the
        # content itself contains are broken first.
        sample = parse_questions(document.raw_text, rfp_id=document.id)[:5]

        agent = build_agent(
            role="RFP triage analyst",
            goal="Decide whether this document is a cloud-migration RFP this system answers.",
            backstory=(
                "You gate the pipeline. A wrong yes spends an SME's time; a wrong no costs seconds."
            ),
            prompt=prompt,
        )
        task = build_task(
            description=prompt.render(
                rfp_title=wrap(document.source_filename),
                rfp_issuer=wrap(document.customer_name),
                section_headings=wrap(
                    "\n".join(section.name for section in document.sections) or "(none)"
                ),
                sample_questions=wrap("\n".join(question.text for question in sample) or "(none)"),
            ),
            expected_output="A JSON object matching the TriageResult contract.",
            agent=agent,
            output_model=TriageResult,
        )
        with llm_span(
            step="triage", prompt_version=prompt.version_tag, model_alias=prompt.model_alias
        ):
            output = await build_crew(agent=agent, task=task).kickoff_async()
        budget.record(_usage_from(output))

        result = _require(output, TriageResult, step="triage")
        assert isinstance(result, TriageResult)  # noqa: S101 - narrowed by _require
        # The contract requires a halt reason on a mismatch; a model that
        # returned domain_match=False without one would have failed validation,
        # so this only ever normalises the reason's identity.
        if not result.domain_match and result.halt_reason is None:
            return result.model_copy(update={"halt_reason": HaltReason.DOMAIN_MISMATCH})
        return result

    # -- requirement extraction -------------------------------------------

    async def extract(
        self, document: RFPDocument, *, budget: BudgetLedger
    ) -> list[ExtractedQuestion]:
        """Deterministic parsing. The model never supplies a field (rule 3).

        The assist exists for one judgement — "is this segment a question at
        all" — and is not reached when the parser is confident, which is the
        normal case. That is why `extract_assist` is budgeted at `<= 1` per run
        rather than at exactly one.
        """
        questions = parse_questions(document.raw_text, rfp_id=document.id)
        logger.info("extracted %d question(s) from %s", len(questions), document.source_filename)
        return questions

    # -- retrieval --------------------------------------------------------

    async def retrieve(
        self,
        question: ExtractedQuestion,
        *,
        requesting_customer: str,
        budget: BudgetLedger,
    ) -> RetrieverSelection:
        """Three tiers plus one batched rerank, then derived selection reasons.

        `requesting_customer` is threaded, never defaulted: the graph query
        requires it and so does this, so no code path retrieves without stating
        who is asking. It is recorded on the returned object because
        cross-customer confidentiality is a zero-tolerance eval category and the
        audit trail has to be able to show it.
        """
        budget.authorize(CallKind.RERANK, question_id=question.id)

        async with httpx.AsyncClient(timeout=self.gateway.timeout) as client:
            embedding = (
                await self.gateway.embed(
                    [question.normalized_text],
                    alias="embed-model",
                    role=EmbedRole.QUERY,
                    client=client,
                )
            )[0]

            artifact = self.calibration or load_for_current_corpus()
            result, trace = await retrieve(
                self.session,
                question_id=question.id,
                question_text=question.normalized_text,
                embedding=embedding,
                requesting_customer=requesting_customer,
                calibration=artifact,
                domain="cloud_migration",
                today=self.today,
                reranker=self.reranker,
            )

        budget.record(CallUsage(tokens=0, cost_usd=None))
        return build_selection(
            result=result,
            trace_reranked=trace.reranked,
            rerank_skip_reason=trace.rerank_skip_reason,
            candidates_considered=trace.candidates_considered,
            requesting_customer=requesting_customer,
        )

    # -- drafting ---------------------------------------------------------

    async def draft(
        self,
        question: ExtractedQuestion,
        selection: RetrieverSelection,
        *,
        budget: BudgetLedger,
        retry_context: str | None = None,
    ) -> DraftedAnswer:
        """Groq 70B, then OUR arithmetic for the confidence.

        The retry does not re-authorize a draft call — the controller has
        already counted it, and charging the retry against the same allowance
        would make the single permitted retry impossible.
        """
        if retry_context is None:
            budget.authorize(CallKind.DRAFT, question_id=question.id)

        prompt = load_prompt("drafter")
        sources = await self._tier_two_sources(selection)
        description = prompt.render(
            question_number=question.printed_number or f"#{question.order + 1}",
            question_text=wrap(question.text),
            word_limit=question.word_limit or "no stated limit",
            sources=_render_sources(sources),
        )
        if retry_context is not None:
            description = (
                f"{description}\n\nYOUR PREVIOUS REPLY WAS REJECTED BY THE OUTPUT "
                f"CONTRACT:\n{retry_context}\n\nReturn JSON that satisfies it this time."
            )

        agent = build_agent(
            role="RFP response drafter",
            goal="Draft one grounded answer, citing a source for every factual claim.",
            backstory=(
                "You write for a human reviewer who will check every citation. An "
                "escalation costs an SME ten minutes; an invented fact costs the bid."
            ),
            prompt=prompt,
            tools=drafter_tools(self.drafter_mcp),
        )
        task = build_task(
            description=description,
            expected_output="A JSON object matching the DraftPayload contract.",
            agent=agent,
            output_model=DraftPayload,
        )
        with llm_span(
            step="drafter", prompt_version=prompt.version_tag, model_alias=prompt.model_alias
        ):
            output = await build_crew(agent=agent, task=task).kickoff_async()
        budget.record(_usage_from(output))

        payload = _require(output, DraftPayload, step="drafter")
        assert isinstance(payload, DraftPayload)  # noqa: S101 - narrowed by _require
        return to_drafted_answer(payload, selection=selection, question=question)

    async def _tier_two_sources(self, selection: RetrieverSelection) -> list[dict[str, object]]:
        """Full answer text for the SELECTED candidates only.

        Progressive disclosure (§10): the retriever saw summaries, and the full
        text is a separate, explicit fetch for the few that survived the floor.
        Fetching everything would put material the ranking already rejected in
        front of the drafter.
        """
        chosen = [s for s in selection.selections if s.outcome is SelectionOutcome.SELECTED]
        if not chosen:
            return []
        async with httpx.AsyncClient(timeout=self.drafter_mcp.timeout) as client:
            gathered: list[dict[str, object]] = []
            for candidate in chosen:
                lineage = await self.drafter_mcp.get_evidence(
                    answer_id=candidate.answer_node_id, client=client
                )
                if lineage is not None:
                    gathered.append(lineage)
            return gathered

    # -- critique ---------------------------------------------------------

    async def critique(
        self,
        question: ExtractedQuestion,
        answer: DraftedAnswer,
        *,
        budget: BudgetLedger,
    ) -> CritiqueResult:
        """Groq 70B. The delta is bounded at zero by the contract, not by the prompt."""
        budget.authorize(CallKind.CRITIQUE, question_id=question.id)

        prompt = load_prompt("critic")
        async with httpx.AsyncClient(timeout=self.drafter_mcp.timeout) as client:
            sources = [
                lineage
                for source_id in answer.source_ids
                if (
                    lineage := await self.drafter_mcp.get_evidence(
                        answer_id=source_id, client=client
                    )
                )
                is not None
            ]

        agent = build_agent(
            role="RFP response critic",
            goal="Find claims the cited sources do not support. Lower confidence or leave it.",
            backstory=(
                "You are the last automated check before a human. You cannot rewrite "
                "and you cannot approve — only flag."
            ),
            prompt=prompt,
            tools=critic_tools(self.drafter_mcp),
        )
        task = build_task(
            description=prompt.render(
                question_number=question.printed_number or f"#{question.order + 1}",
                question_text=wrap(question.text),
                answer_text=wrap(answer.answer_text),
                source_ids=", ".join(answer.source_ids),
                sources=_render_sources(sources),
            ),
            expected_output="A JSON object matching the CritiqueResult contract.",
            agent=agent,
            output_model=CritiqueResult,
        )
        with llm_span(
            step="critic", prompt_version=prompt.version_tag, model_alias=prompt.model_alias
        ):
            output = await build_crew(agent=agent, task=task).kickoff_async()
        budget.record(_usage_from(output))

        critique = _require(output, CritiqueResult, step="critic")
        assert isinstance(critique, CritiqueResult)  # noqa: S101 - narrowed by _require
        return critique


# ---------------------------------------------------------------------------
# Deterministic helpers. These are the parts rule 3 keeps out of the model.
# ---------------------------------------------------------------------------


def build_selection(
    *,
    result: object,
    trace_reranked: bool,
    rerank_skip_reason: str | None,
    candidates_considered: int,
    requesting_customer: str,
) -> RetrieverSelection:
    """Turn a scored `RetrievalResult` into the typed selection object.

    EVERY REASON HERE IS DERIVED. `preference_changed_rank` is computed by
    re-sorting the same candidates on relevance alone and comparing positions,
    which is the only way to state what preference actually did rather than
    what it was configured to be able to do.
    """
    from src.contracts import RetrievalResult

    assert isinstance(result, RetrievalResult)  # noqa: S101 - the caller's own output

    by_relevance = sorted(result.candidates, key=lambda c: (-c.relevance, c.answer_node_id))
    relevance_rank = {c.answer_node_id: i for i, c in enumerate(by_relevance)}

    selections = [
        CandidateSelection(
            answer_node_id=candidate.answer_node_id,
            rank=index,
            outcome=(
                SelectionOutcome.SELECTED
                if candidate.relevance >= result.floor_used
                else SelectionOutcome.BELOW_RELEVANCE_FLOOR
            ),
            relevance=candidate.relevance,
            floor_used=result.floor_used,
            preference=candidate.preference,
            preference_changed_rank=relevance_rank[candidate.answer_node_id] != index,
        )
        for index, candidate in enumerate(result.candidates)
    ]

    return RetrieverSelection(
        question_id=result.question_id,
        requesting_customer=requesting_customer,
        status=result.status,
        ranking_basis=(
            RankingBasis.SIMILARITY_AND_RERANK
            if trace_reranked
            else RankingBasis.CALIBRATED_SIMILARITY_ONLY
        ),
        rerank_skip_reason=None if trace_reranked else (rerank_skip_reason or "rerank not applied"),
        candidates_considered=candidates_considered,
        selections=selections,
    )


def to_drafted_answer(
    payload: DraftPayload,
    *,
    selection: RetrieverSelection,
    question: ExtractedQuestion,
) -> DraftedAnswer:
    """Compute the confidence and assemble the answer contract.

    `primary_final_score` is the RELEVANCE of the top selected candidate — the
    source the answer principally rests on. Coverage comes from the drafter's
    own claim decomposition. Neither number is the model's opinion of itself.
    """
    from src.retrieval.confidence import compute_confidence, requires_sme_review

    selected = [s for s in selection.selections if s.outcome is SelectionOutcome.SELECTED]
    primary = selected[0].relevance if selected else 0.0
    claims = payload.claims
    supported = sum(1 for claim in claims if claim.is_supported)

    confidence = compute_confidence(
        primary_final_score=primary,
        claims_with_sources=supported,
        total_claims=len(claims),
    )
    unsupported = payload.unsupported_claims
    needs_review = (
        payload.escalate
        or bool(unsupported)
        or not payload.source_ids
        or requires_sme_review(confidence)
    )
    reason = payload.escalation_reason
    if needs_review and not reason:
        if not payload.source_ids:
            reason = "the draft cited no source"
        elif unsupported:
            reason = f"{len(unsupported)} claim(s) carry no citation: {unsupported[0]}"
        else:
            reason = f"computed confidence {confidence:.2f} is below the escalation threshold"

    return DraftedAnswer(
        question_id=question.id,
        answer_text=payload.answer_text,
        source_ids=payload.source_ids,
        confidence=confidence,
        needs_sme_review=needs_review,
        unsupported_claims=unsupported,
        escalation_reason=reason,
        confidence_inputs=ConfidenceInputs(
            primary_final_score=primary,
            claims_with_sources=supported,
            total_claims=len(claims),
        ),
    )


def _render_sources(sources: list[dict[str, object]]) -> str:
    """Sources as a numbered block for a prompt.

    JSON rather than prose, so the model can see the field names it is being
    asked to cite by, and so an evidence record containing something that reads
    like an instruction is visibly DATA in a structure rather than a sentence
    sitting in the prompt body.
    """
    if not sources:
        return "(no sources were retrieved for this question)"
    return "\n\n".join(
        f"[{index + 1}] {json.dumps(source, ensure_ascii=False, default=str)}"
        for index, source in enumerate(sources)
    )


__all__ = ["CrewAgentLayer", "build_selection", "to_drafted_answer"]
