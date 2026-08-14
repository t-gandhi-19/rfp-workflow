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

from pydantic import BaseModel, ConfigDict, Field

from src.contracts import DraftedAnswer, EntityCheckResult, RunState
from src.contracts.embedding import embedding_config
from src.contracts.enums import EntityType
from src.graph import queries
from src.mcp_server.context import ToolContext

#: Read-only access to the graph. The six original tools all require exactly
#: this, because none of them writes and there was nothing to subdivide.
KG_READER = "kg-reader"

#: Phase 4. `save_draft` writes, and writing is not reading — a caller that may
#: retrieve is not thereby a caller that may persist an answer. Same role name
#: and same semantics as the write-api's, because it IS the write-api's: the
#: tool proxies there with the caller's own token rather than writing Postgres
#: itself (rule 4).
DRAFT_WRITER = "draft-writer"

#: Phase 4. `get_run_state` reads a run's progress. A third role rather than
#: reusing `kg-reader`, because the graph and the run record are different
#: worlds: a component entitled to read the corpus is not thereby entitled to
#: read what any given customer's live run is doing.
RFP_READER = "rfp-reader"


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


class SaveDraftInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_id: str = Field(min_length=1, description="The run this answer belongs to.")
    #: The whole contract, not a loose set of fields. `DraftedAnswer` refuses an
    #: uncited, unescalated answer, so nothing ungrounded can reach the table
    #: through this tool — the validation happens before a connection is opened.
    answer: DraftedAnswer


class SaveDraftOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    written: str
    key: str
    #: The JWT subject write-api recorded. Returned so a caller can see whose
    #: identity was actually stamped on the row rather than assuming its own.
    written_by: str


class GetRunStateInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_id: str = Field(min_length=1)


class RunStateOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    #: None for a run id that does not exist — a legitimate answer rather than
    #: an error, so an agent is not told to retry something that cannot succeed.
    state: RunState | None


# ---------------------------------------------------------------------------
# Handlers
# ---------------------------------------------------------------------------


async def _retrieve_candidates(
    ctx: ToolContext, payload: RetrieveCandidatesInput
) -> CandidatesOutput:
    expected = embedding_config().model.dimensions
    if len(payload.embedding) != expected:
        raise ValueError(
            f"embedding has {len(payload.embedding)} dimensions, the index expects {expected}"
        )
    hits = await queries.find_similar_questions(
        ctx.require_session(),
        embedding=payload.embedding,
        domain=payload.domain,
        requesting_customer=payload.requesting_customer,
        k=payload.k,
    )
    return CandidatesOutput(candidates=hits)


async def _get_full_answers(ctx: ToolContext, payload: GetFullAnswersInput) -> AnswersOutput:
    answers = await queries.answers_for_question(
        ctx.require_session(),
        question_id=payload.question_id,
        requesting_customer=payload.requesting_customer,
    )
    return AnswersOutput(answers=answers)


async def _get_evidence(ctx: ToolContext, payload: GetEvidenceInput) -> EvidenceOutput:
    lineage = await queries.get_answer_with_lineage(
        ctx.require_session(), answer_id=payload.answer_id
    )
    return EvidenceOutput(lineage=lineage)


async def _entity_exists(ctx: ToolContext, payload: EntityExistsInput) -> EntityExistsOutput:
    resolution = await queries.entity_exists(
        ctx.require_session(), name_or_code=payload.name, entity_type=payload.entity_type
    )
    return EntityExistsOutput(resolution=resolution)


async def _get_sme_for_capability(
    ctx: ToolContext, payload: GetSmeForCapabilityInput
) -> SmesOutput:
    smes = await queries.get_sme_for_capability(
        ctx.require_session(), capability_id=payload.capability_id
    )
    return SmesOutput(smes=smes)


async def _save_draft(ctx: ToolContext, payload: SaveDraftInput) -> SaveDraftOutput:
    """Persist one drafted answer — THROUGH write-api, never to Postgres here.

    Rule 4 has no exception for "the service is on our side of the network".
    The row's `written_by` therefore records the ORIGINAL caller, because this
    forwards their token rather than minting one of its own: an audit trail that
    said every draft was written by mcp-server would be an audit trail that
    answered no question anybody asks of it.
    """
    from src.mcp_server.write_client import put_draft

    ack = await put_draft(run_id=payload.run_id, answer=payload.answer, token=ctx.require_token())
    return SaveDraftOutput(written=ack["written"], key=ack["key"], written_by=ack["written_by"])


async def _get_run_state(ctx: ToolContext, payload: GetRunStateInput) -> RunStateOutput:
    """Read one run's checkpointed progress.

    Reads go directly through the SELECT-only identity, the same asymmetry as
    `src/evals/store.py` and `src/controller/checkpoint.py` — and for the same
    reason: adding a read endpoint to write-api would put a query surface on the
    only thing allowed to mutate state.
    """
    from src.controller.checkpoint import CheckpointError, load_run

    try:
        resumed = await load_run(payload.run_id)
    except CheckpointError:
        # Distinct from an error: "no such run" is a legitimate answer, and a
        # 500 would tell an agent to retry something that will never succeed.
        return RunStateOutput(state=None)
    return RunStateOutput(state=resumed.state)


async def _coverage_gaps(ctx: ToolContext, payload: CoverageGapsInput) -> CoverageGapsOutput:
    gaps = await queries.coverage_gaps(ctx.require_session(), rfp_id=payload.rfp_id)
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
    handler: Callable[[ToolContext, Any], Awaitable[BaseModel]]
    #: The realm role this tool requires. Defaults to `kg-reader` because six of
    #: the eight are graph reads — but it is a FIELD, so a write tool cannot be
    #: added without stating what it needs, and `tests/unit/test_mcp_tools.py`
    #: asserts that the two non-read tools do not sit behind the read role.
    required_role: str = KG_READER
    #: Whether the invocation opens a graph session.
    needs_graph: bool = True
    #: Whether the caller's own bearer token is forwarded to the handler. True
    #: only for tools that write through write-api, so a read tool never holds
    #: a credential it has no use for.
    needs_token: bool = False

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
    # Phase 4. The first two tools that are not graph reads, and the reason
    # `required_role` is a field rather than a constant.
    ToolSpec(
        name="save_draft",
        description=(
            "Persist one drafted answer for a run. Writes through write-api with the "
            "CALLER's token, so the row records who asked. Requires 'draft-writer'."
        ),
        input_model=SaveDraftInput,
        output_model=SaveDraftOutput,
        handler=_save_draft,
        required_role=DRAFT_WRITER,
        needs_graph=False,
        needs_token=True,
    ),
    ToolSpec(
        name="get_run_state",
        description=(
            "One run's checkpointed stage, per-question statuses and spend. "
            "Returns null for an unknown run id. Requires 'rfp-reader'."
        ),
        input_model=GetRunStateInput,
        output_model=RunStateOutput,
        handler=_get_run_state,
        required_role=RFP_READER,
        needs_graph=False,
    ),
)

TOOLS_BY_NAME: dict[str, ToolSpec] = {tool.name: tool for tool in TOOLS}
