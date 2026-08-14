"""The drafter's and critic's tools.

Driven against a fake MCP client, because what is under test is what the tool
SAYS BACK TO THE MODEL. A tool that resolves nothing and a tool that resolves
something must produce visibly different instructions, and a tool that fails
must produce an instruction rather than an exception the model never sees.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from src.agents.mcp_client import McpError
from src.agents.tools import EntityExistsTool, GetEvidenceTool, critic_tools, drafter_tools


@dataclass
class FakeMcp:
    """Answers `call_sync` from a script."""

    responses: dict[str, dict[str, object]] = field(default_factory=dict)
    raises: set[str] = field(default_factory=set)
    calls: list[tuple[str, dict[str, object]]] = field(default_factory=list)
    timeout: float = 5.0

    def call_sync(self, tool: str, payload: dict[str, object]) -> dict[str, object]:
        self.calls.append((tool, payload))
        if tool in self.raises:
            raise McpError(f"mcp-server unreachable for '{tool}'")
        return self.responses.get(tool, {})


class TestGetEvidence:
    def test_a_known_answer_returns_its_provenance(self) -> None:
        mcp = FakeMcp(responses={"get_evidence": {"lineage": {"answer_id": "ANS-0007"}}})
        tool = GetEvidenceTool(client=mcp)
        assert "ANS-0007" in tool._run(answer_id="ANS-0007")

    def test_an_unknown_answer_tells_the_model_not_to_cite_it(self) -> None:
        """Distinct from "known, no evidence". Citing an id that is not in the
        graph is a fabricated citation, and the tool has to say so in words the
        model will act on."""
        mcp = FakeMcp(responses={"get_evidence": {"lineage": None}})
        tool = GetEvidenceTool(client=mcp)
        reply = tool._run(answer_id="ANS-9999")
        assert "Do not cite it" in reply
        assert "escalate" in reply

    def test_a_transport_failure_is_reported_not_raised(self) -> None:
        """A raised exception inside a crewAI tool becomes an agent-loop failure
        whose message the model never sees."""
        mcp = FakeMcp(raises={"get_evidence"})
        tool = GetEvidenceTool(client=mcp)
        assert "get_evidence failed" in tool._run(answer_id="ANS-0007")


class TestEntityExists:
    def test_a_resolving_entity_is_safe_to_name(self) -> None:
        mcp = FakeMcp(
            responses={
                "entity_exists": {
                    "resolution": {
                        "entity_text": "Kubernetes",
                        "entity_type": "tool",
                        "resolved_node_id": "TOOL-0004",
                        "passed": True,
                    }
                }
            }
        )
        tool = EntityExistsTool(client=mcp)
        reply = tool._run(name="Kubernetes", entity_type="tool")
        assert "TOOL-0004" in reply
        assert "Safe to name" in reply

    def test_an_unresolved_entity_is_forbidden_in_the_answer(self) -> None:
        """The world is CLOSED (§15). The tool states the consequence, not just
        the fact."""
        mcp = FakeMcp(
            responses={
                "entity_exists": {
                    "resolution": {
                        "entity_text": "Northwind Cloud",
                        "entity_type": "vendor",
                        "resolved_node_id": None,
                        "passed": False,
                    }
                }
            }
        )
        tool = EntityExistsTool(client=mcp)
        reply = tool._run(name="Northwind Cloud", entity_type="vendor")
        assert "does NOT resolve" in reply
        assert "Do not name it" in reply

    def test_an_unknown_entity_type_lists_the_valid_ones(self) -> None:
        """And costs no round trip — the payload would have been refused by the
        server's schema anyway, and a 422 says less to a model than this does."""
        mcp = FakeMcp()
        tool = EntityExistsTool(client=mcp)
        reply = tool._run(name="Kubernetes", entity_type="framework")
        assert "is not an entity type" in reply
        assert "vendor" in reply
        assert mcp.calls == []

    def test_a_transport_failure_is_reported_not_raised(self) -> None:
        mcp = FakeMcp(raises={"entity_exists"})
        tool = EntityExistsTool(client=mcp)
        assert "entity_exists failed" in tool._run(name="Kubernetes", entity_type="tool")


class TestWhoGetsWhichTools:
    def test_the_drafter_gets_both(self) -> None:
        names = {tool.name for tool in drafter_tools(FakeMcp())}
        assert names == {"get_evidence", "entity_exists"}

    def test_the_critic_gets_only_the_registry(self) -> None:
        """A critic that could fetch new evidence would be re-drafting, which
        rule 14 forbids it."""
        names = {tool.name for tool in critic_tools(FakeMcp())}
        assert names == {"entity_exists"}

    @pytest.mark.parametrize("tool_name", ["get_evidence", "entity_exists"])
    def test_no_tool_accepts_a_query(self, tool_name: str) -> None:
        """Rule 4: agents never write free Cypher. The tool surface has no way
        to phrase one — asserted on the schema rather than trusted to the
        server, so a future tool that added a passthrough fails here."""
        tool = next(t for t in drafter_tools(FakeMcp()) if t.name == tool_name)
        fields = set(tool.args_schema.model_fields)
        assert not (fields & {"query", "cypher", "sql", "filter", "where"})
