"""The six read tools, and the contracts that define their boundary (§2).

EVERY TOOL WRAPS A TESTED FUNCTION IN `src.graph.queries`. That is CLAUDE.md
rule 4, and it is the whole design: an agent cannot phrase a graph question, only
choose among questions that have already been written, parameterised and tested.

**There is no raw-query tool, and there is no way to add one by accident.** A
tool is a `ToolSpec` naming a handler; the handler receives a validated input
contract and returns a Pydantic model. Nothing in this module accepts Cypher, and
`tests/unit/test_mcp_tools.py` asserts that no tool's input contract has a field
that could carry one.

WHY THE CONTRACTS LIVE HERE rather than being reused from `queries`. The query
functions take keyword arguments; a tool boundary takes a JSON document. Those
are different shapes with different failure modes, and the contract is what turns
"the caller sent nonsense" into a readable 422 rather than a TypeError from three
frames down. The output models are the query functions' own return types, so the
boundary adds no second definition of what an answer is.

`retrieve_candidates` REQUIRES `requesting_customer`. It has no default and is
not optional, for the same reason `find_similar_questions` has none: an optional
visibility parameter is one forgotten argument away from a leak, and a tool
boundary is exactly where arguments get forgotten.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from neo4j import AsyncSession
from pydantic import BaseModel, ConfigDict, Field

from src.contracts import EntityCheckResult
from src.contracts.embedding import embedding_config
from src.contracts.enums import EntityType
from src.graph import queries

#: The realm role every tool requires. Read-only access to the graph is a single
#: grant: there is no tool here that writes, so there is nothing to subdivide.
KG_READER = "kg-reader"


# ---------------------------------------------------------------------------
# Input contracts. One per tool, all `extra="forbid"`.
#
# forbid rather than ignore, deliberately: a caller sending `requesting_custmer`
# should be told, not silently served a result computed without it.
# ---------------------------------------------------------------------------


class RetrieveCandidatesInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    embedding: list[float] = Field(
        min_length=1,
        description="Query-side embedding. Width must match the configured index.",
    )
    #: REQUIRED. No default, not optional — see the module docstring.
    requesting_customer: str = Field(
        min_length=1,
        description="Customer this retrieval is executed for. Confidential material "
        "belonging to any other customer is excluded inside the query.",
    )
    domain: str = Field(default="cloud_migration", min_length=1)
    k: int = Field(default=10, gt=0, le=50)


class GetFullAnswersInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question_id: str = Field(min_length=1)
    requesting_customer: str = Field(min_length=1)


class GetEvidenceInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    answer_id: str = Field(min_length=1)


class EntityExistsInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    entity_type: EntityType


class GetSmeForCapabilityInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    capability_id: str = Field(min_length=1)


class CoverageGapsInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    rfp_id: str = Field(min_length=1)


# ---------------------------------------------------------------------------
# Output envelopes. A list result is wrapped rather than returned bare, so every
# response is a JSON object and a future field can be added without changing the
# shape callers parse.
# ---------------------------------------------------------------------------


class CandidatesOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    candidates: list[queries.SimilarQuestion]


class AnswersOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    answers: list[queries.AnswerRecord]


class EvidenceOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    #: None when the answer id is unknown — distinct from "known, no evidence".
    lineage: queries.AnswerLineage | None


class EntityExistsOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    resolution: EntityCheckResult


class SmesOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    smes: list[queries.SMERecord]


class CoverageGapsOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    gaps: list[queries.CoverageGap]


# ---------------------------------------------------------------------------
# Handlers
# ---------------------------------------------------------------------------


async def _retrieve_candidates(
    session: AsyncSession, payload: RetrieveCandidatesInput
) -> CandidatesOutput:
    expected = embedding_config().model.dimensions
    if len(payload.embedding) != expected:
        raise ValueError(
            f"embedding has {len(payload.embedding)} dimensions, the index expects {expected}"
        )
    hits = await queries.find_similar_questions(
        session,
        embedding=payload.embedding,
        domain=payload.domain,
        requesting_customer=payload.requesting_customer,
        k=payload.k,
    )
    return CandidatesOutput(candidates=hits)


async def _get_full_answers(session: AsyncSession, payload: GetFullAnswersInput) -> AnswersOutput:
    answers = await queries.answers_for_question(
        session,
        question_id=payload.question_id,
        requesting_customer=payload.requesting_customer,
    )
    return AnswersOutput(answers=answers)


async def _get_evidence(session: AsyncSession, payload: GetEvidenceInput) -> EvidenceOutput:
    lineage = await queries.get_answer_with_lineage(session, answer_id=payload.answer_id)
    return EvidenceOutput(lineage=lineage)


async def _entity_exists(session: AsyncSession, payload: EntityExistsInput) -> EntityExistsOutput:
    resolution = await queries.entity_exists(
        session, name_or_code=payload.name, entity_type=payload.entity_type
    )
    return EntityExistsOutput(resolution=resolution)


async def _get_sme_for_capability(
    session: AsyncSession, payload: GetSmeForCapabilityInput
) -> SmesOutput:
    smes = await queries.get_sme_for_capability(session, capability_id=payload.capability_id)
    return SmesOutput(smes=smes)


async def _coverage_gaps(session: AsyncSession, payload: CoverageGapsInput) -> CoverageGapsOutput:
    gaps = await queries.coverage_gaps(session, rfp_id=payload.rfp_id)
    return CoverageGapsOutput(gaps=gaps)


# ---------------------------------------------------------------------------
# The registry
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ToolSpec:
    """One tool: a name, a contract pair, and the tested function behind it."""

    name: str
    description: str
    input_model: type[BaseModel]
    output_model: type[BaseModel]
    handler: Callable[[AsyncSession, Any], Awaitable[BaseModel]]

    def schema(self) -> dict[str, Any]:
        """The manifest entry. JSON Schema derived from the contracts, never
        hand-written — a hand-written schema is a second definition that drifts."""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_model.model_json_schema(),
            "output_schema": self.output_model.model_json_schema(),
        }


TOOLS: tuple[ToolSpec, ...] = (
    ToolSpec(
        name="retrieve_candidates",
        description=(
            "Top-k in-domain questions similar to a query embedding, filtered to what "
            "the requesting customer may see. Confidentiality, supersession and domain "
            "are applied inside the query."
        ),
        input_model=RetrieveCandidatesInput,
        output_model=CandidatesOutput,
        handler=_retrieve_candidates,
    ),
    ToolSpec(
        name="get_full_answers",
        description="Live answers for a question, excluding any this customer may not see.",
        input_model=GetFullAnswersInput,
        output_model=AnswersOutput,
        handler=_get_full_answers,
    ),
    ToolSpec(
        name="get_evidence",
        description="One answer's full provenance: author, customer, outcome, evidence codes.",
        input_model=GetEvidenceInput,
        output_model=EvidenceOutput,
        handler=_get_evidence,
    ),
    ToolSpec(
        name="entity_exists",
        description=(
            "Resolve a named entity against the registry. Grounding depends on this "
            "answering 'no' honestly."
        ),
        input_model=EntityExistsInput,
        output_model=EntityExistsOutput,
        handler=_entity_exists,
    ),
    ToolSpec(
        name="get_sme_for_capability",
        description="The SMEs who own a capability, for escalation routing.",
        input_model=GetSmeForCapabilityInput,
        output_model=SmesOutput,
        handler=_get_sme_for_capability,
    ),
    ToolSpec(
        name="coverage_gaps",
        description="Questions in an RFP with no live answer, and why.",
        input_model=CoverageGapsInput,
        output_model=CoverageGapsOutput,
        handler=_coverage_gaps,
    ),
)

TOOLS_BY_NAME: dict[str, ToolSpec] = {tool.name: tool for tool in TOOLS}
