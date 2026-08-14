"""The tools the drafter and critic may reach for.

TWO TOOLS, AND NO MORE. `get_evidence` reads one answer's provenance;
`entity_exists` resolves a name against the closed-world registry. Neither takes
a query, neither takes a filter, and there is nothing here that accepts Cypher —
rule 4 is enforced by the MCP server's surface, and this module cannot widen it
because it has no way to phrase a request the server does not already implement.

WHY THE DRAFTER GETS TOOLS AT ALL, given that its sources are already rendered
into its prompt. Because the prompt carries what the RANKING selected, and the
drafter may need to check something it is about to write: whether a vendor it is
naming actually exists, or what a source's evidence codes really say. The
alternative is a model that writes the name and hopes — which is the failure the
entity guardrail exists to catch, and catching it later costs an escalation
that a tool call could have avoided.

A TOOL FAILURE IS NOT A RUN FAILURE. Each `_run` returns a readable string on
error rather than raising, because a raised exception inside a crewAI tool call
becomes an agent-loop failure whose message the model never sees. The model
being told "that vendor does not resolve" is the entire point of the tool.
"""

from __future__ import annotations

import json
from typing import Protocol, runtime_checkable

from crewai.tools import BaseTool
from pydantic import BaseModel, ConfigDict, Field

from src.agents.mcp_client import McpError
from src.contracts import EntityType


@runtime_checkable
class ToolCaller(Protocol):
    """What a tool needs from an MCP client, and nothing else.

    `BaseTool` is a Pydantic model, so this field is validated at construction.
    Typing it as the concrete `McpClient` would make the client a dependency of
    every test of what these tools SAY — and what they say to the model is the
    whole of their behaviour. Naming the one method they use keeps that
    testable without loosening the field to `Any`.
    """

    def call_sync(self, tool: str, payload: dict[str, object]) -> dict[str, object]: ...


class GetEvidenceInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    answer_id: str = Field(description="The answer node id to fetch provenance for, e.g. ANS-0007.")


class EntityExistsInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(description="The exact name or code as you intend to write it.")
    entity_type: str = Field(
        description=("One of: vendor, product, certification, client, tool, location.")
    )


class GetEvidenceTool(BaseTool):
    """Tier 3 of progressive disclosure: one answer's full provenance."""

    name: str = "get_evidence"
    description: str = (
        "Fetch one previous answer's provenance — author, customer, commercial outcome, "
        "evidence codes, and whether it has been superseded. Use this before citing a "
        "source id you have not already been shown in full. Takes an answer id."
    )
    args_schema: type[BaseModel] = GetEvidenceInput
    model_config = ConfigDict(arbitrary_types_allowed=True)
    client: ToolCaller

    def _run(self, answer_id: str) -> str:
        try:
            body = self.client.call_sync("get_evidence", {"answer_id": answer_id})
        except McpError as exc:
            return f"get_evidence failed for {answer_id}: {exc}"
        lineage = body.get("lineage")
        if lineage is None:
            # A distinct answer from "no evidence recorded": this id is not in
            # the graph at all, which means citing it would be a fabricated
            # citation.
            return (
                f"No answer with id {answer_id} exists. Do not cite it. "
                f"If you have no other source for the claim, escalate."
            )
        return json.dumps(lineage, ensure_ascii=False, default=str)


class EntityExistsTool(BaseTool):
    """The closed-world grounding check (build prompt §15)."""

    name: str = "entity_exists"
    description: str = (
        "Check whether a vendor, product, certification, client, tool or location exists "
        "in the registry before naming it. The world is CLOSED: if it does not resolve "
        "here, it may not appear in the answer. Takes a name and an entity type."
    )
    args_schema: type[BaseModel] = EntityExistsInput
    model_config = ConfigDict(arbitrary_types_allowed=True)
    client: ToolCaller

    def _run(self, name: str, entity_type: str) -> str:
        try:
            resolved = EntityType(entity_type)
        except ValueError:
            return (
                f"'{entity_type}' is not an entity type. Use one of: "
                f"{', '.join(t.value for t in EntityType)}."
            )
        try:
            body = self.client.call_sync(
                "entity_exists", {"name": name, "entity_type": resolved.value}
            )
        except McpError as exc:
            return f"entity_exists failed for {name}: {exc}"

        resolution = body.get("resolution")
        if isinstance(resolution, dict) and resolution.get("passed"):
            return f"'{name}' resolves to {resolution.get('resolved_node_id')}. Safe to name."
        return (
            f"'{name}' does NOT resolve as a {resolved.value}. Do not name it in the answer. "
            f"Use a term that does resolve, or escalate."
        )


def drafter_tools(client: ToolCaller) -> list[BaseTool]:
    """Build prompt §11: the drafter gets `get_evidence` and `entity_exists`."""
    return [GetEvidenceTool(client=client), EntityExistsTool(client=client)]


def critic_tools(client: ToolCaller) -> list[BaseTool]:
    """The critic gets `entity_exists` only.

    It is checking a written answer against sources it has already been given,
    so it needs the registry to test names the drafter used — and nothing that
    would let it go looking for material the drafter never saw. A critic that
    could fetch new evidence would be re-drafting, which rule 14 forbids it.
    """
    return [EntityExistsTool(client=client)]
