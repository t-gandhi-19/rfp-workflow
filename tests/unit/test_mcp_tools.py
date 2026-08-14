"""The MCP tool boundary: what exists, what it accepts, and what it refuses.

The server is an agent's ONLY route to the graph, so the interesting assertions
are about what is absent — no Cypher, no writes, no tool outside the six — and
those cannot be checked by reading a diff.
"""

from __future__ import annotations

import ast
import inspect
import textwrap

import pytest
from pydantic import ValidationError

from src.contracts.enums import EntityType
from src.graph import queries
from src.mcp_server.tools import (
    DRAFT_WRITER,
    KG_READER,
    RFP_READER,
    TOOLS,
    TOOLS_BY_NAME,
    EntityExistsInput,
    RetrieveCandidatesInput,
    ToolSpec,
)

EXPECTED_TOOLS = {
    "retrieve_candidates",
    "get_full_answers",
    "get_evidence",
    "entity_exists",
    "get_sme_for_capability",
    "coverage_gaps",
    # Phase 4. Neither is a graph read, which is why `required_role` became a
    # field on ToolSpec rather than staying a module constant.
    "save_draft",
    "get_run_state",
}

#: The six that wrap `src.graph.queries`. Kept separate from the set above
#: because the "delegates to a tested query function" rule is about THEM: the
#: two Phase 4 tools delegate to write-api and to the checkpoint reader, and
#: asserting they call `queries.` would be asserting the wrong thing.
GRAPH_TOOLS = {
    "retrieve_candidates",
    "get_full_answers",
    "get_evidence",
    "entity_exists",
    "get_sme_for_capability",
    "coverage_gaps",
}

#: Substrings that would betray a query-language escape hatch in a field name.
_QUERY_SMELLS = ("cypher", "query_string", "raw", "statement", "sql", "script", "eval", "exec")


class TestTheToolSet:
    def test_exactly_the_specified_tools_exist(self) -> None:
        assert set(TOOLS_BY_NAME) == EXPECTED_TOOLS

    def test_names_are_unique(self) -> None:
        assert len(TOOLS) == len(TOOLS_BY_NAME) == len(EXPECTED_TOOLS)

    def test_every_tool_describes_itself(self) -> None:
        """The description is what an agent chooses on."""
        for tool in TOOLS:
            assert len(tool.description) > 40, tool.name


class TestRolesAreNotOneGrant:
    """A retrieval grant is not a write grant.

    Every tool sat behind `kg-reader` until Phase 4, and adding a WRITE tool to
    that route would have made every existing retriever able to persist an
    answer — without any test failing, because the route-level dependency was
    the only thing that named a role.
    """

    def test_the_graph_reads_need_kg_reader(self) -> None:
        for name in GRAPH_TOOLS:
            assert TOOLS_BY_NAME[name].required_role == KG_READER, name

    def test_saving_a_draft_needs_the_write_role(self) -> None:
        assert TOOLS_BY_NAME["save_draft"].required_role == DRAFT_WRITER

    def test_reading_a_run_needs_the_run_reader_role(self) -> None:
        """The graph and the run record are different worlds: reading the corpus
        does not entitle a caller to watch a customer's live run."""
        assert TOOLS_BY_NAME["get_run_state"].required_role == RFP_READER

    def test_no_write_tool_hides_behind_the_read_role(self) -> None:
        for tool in TOOLS:
            if not tool.needs_graph and tool.name != "get_run_state":
                assert tool.required_role != KG_READER, tool.name

    def test_only_the_write_tool_is_handed_a_token(self) -> None:
        """A read tool holding a forwarded credential is a credential with no
        use and one more place to leak it from."""
        assert [tool.name for tool in TOOLS if tool.needs_token] == ["save_draft"]

    def test_the_non_graph_tools_get_no_session(self) -> None:
        assert {tool.name for tool in TOOLS if not tool.needs_graph} == {
            "save_draft",
            "get_run_state",
        }


class TestNoRawQueryToolExists:
    """CLAUDE.md rule 4: agents never write free Cypher or SQL.

    Asserted structurally rather than by inspection, because 'nobody added one'
    is a claim about the future as much as the present.
    """

    def test_no_tool_is_named_like_a_raw_query(self) -> None:
        for name in TOOLS_BY_NAME:
            assert not any(smell in name.lower() for smell in _QUERY_SMELLS), name

    def test_no_input_field_could_carry_a_query(self) -> None:
        """A field called `cypher` would be a raw-query tool wearing a hat."""
        for tool in TOOLS:
            for field in tool.input_model.model_fields:
                assert not any(smell in field.lower() for smell in _QUERY_SMELLS), (
                    f"{tool.name}.{field}"
                )

    def test_every_input_contract_forbids_extra_fields(self) -> None:
        """`ignore` would let an unknown field through in silence.

        That matters more here than usual: a caller sending `requesting_custmer`
        must be told, not quietly served a result computed without it.
        """
        for tool in TOOLS:
            assert tool.input_model.model_config.get("extra") == "forbid", tool.name
            assert tool.output_model.model_config.get("extra") == "forbid", tool.name

    def test_every_handler_delegates_to_a_tested_query_function(self) -> None:
        """The handler bodies name functions from `src.graph.queries` and run no
        Cypher of their own."""
        exported = {name for name in dir(queries) if not name.startswith("_")}
        for tool in (TOOLS_BY_NAME[name] for name in sorted(GRAPH_TOOLS)):
            source = inspect.getsource(tool.handler)
            assert "queries." in source, tool.name
            called = {name for name in exported if f"queries.{name}(" in source}
            assert called, f"{tool.name} does not call a query function"
            # No inline Cypher: these keywords never appear in a handler.
            lowered = source.lower()
            for keyword in ("match (", "merge ", "create ", "delete ", "set "):
                assert keyword not in lowered, f"{tool.name} contains inline Cypher: {keyword}"

    def test_the_non_graph_handlers_do_not_touch_a_database_directly(self) -> None:
        """Rule 4 for the two tools that are not graph reads.

        `save_draft` must go through write-api and `get_run_state` through the
        checkpoint reader. A handler that opened its own engine, or wrote SQL,
        would satisfy every other assertion in this file — none of them look at
        these two, since neither calls `queries.`.
        """
        for name in ("save_draft", "get_run_state"):
            source = inspect.getsource(TOOLS_BY_NAME[name].handler).lower()
            for smell in ("insert into", "update ", "create_async_engine", "text(", "execute("):
                assert smell not in source, f"{name} touches a database directly: {smell}"

    def test_save_draft_delegates_to_the_write_api_client(self) -> None:
        source = inspect.getsource(TOOLS_BY_NAME["save_draft"].handler)
        assert "put_draft" in source

    def test_get_run_state_delegates_to_the_checkpoint_reader(self) -> None:
        source = inspect.getsource(TOOLS_BY_NAME["get_run_state"].handler)
        assert "load_run" in source

    def test_every_handler_calls_its_query_function_with_valid_keywords(self) -> None:
        """Wrapping a function is only safe if the call actually binds.

        Caught a real one: `_entity_exists` passed `name=`, but
        `queries.entity_exists` takes `name_or_code=`. Nothing in the contract
        tests could see it — they never invoke a handler — and it would have
        surfaced as a TypeError on the first live call.

        Reads the keyword names out of each handler's AST and checks them
        against the target's real signature, so the binding is verified without
        a database.
        """
        for tool in (TOOLS_BY_NAME[name] for name in sorted(GRAPH_TOOLS)):
            tree = ast.parse(textwrap.dedent(inspect.getsource(tool.handler)))
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                func = node.func
                if not (
                    isinstance(func, ast.Attribute)
                    and isinstance(func.value, ast.Name)
                    and func.value.id == "queries"
                ):
                    continue
                target = getattr(queries, func.attr)
                accepted = set(inspect.signature(target).parameters)
                passed = {kw.arg for kw in node.keywords if kw.arg}
                unknown = passed - accepted
                assert not unknown, (
                    f"{tool.name} calls queries.{func.attr} with {sorted(unknown)}, "
                    f"which it does not accept; it takes {sorted(accepted)}"
                )

    def test_no_tool_name_suggests_a_write(self) -> None:
        for name in TOOLS_BY_NAME:
            assert not any(
                verb in name for verb in ("write", "create", "update", "delete", "merge", "set")
            ), name


class TestRequestingCustomerIsRequired:
    """An optional visibility parameter is one forgotten argument away from a leak.

    A tool boundary is exactly where an argument gets forgotten, so the contract
    refuses rather than defaulting.
    """

    def test_retrieve_candidates_refuses_without_it(self) -> None:
        with pytest.raises(ValidationError) as caught:
            # mypy sees the omission too, which is the point — the runtime
            # refusal is what protects a JSON caller mypy never looks at.
            RetrieveCandidatesInput(embedding=[0.1] * 768, k=5)  # type: ignore[call-arg]
        assert "requesting_customer" in str(caught.value)

    def test_it_has_no_default(self) -> None:
        field = RetrieveCandidatesInput.model_fields["requesting_customer"]
        assert field.is_required()

    def test_an_empty_string_is_not_a_customer(self) -> None:
        with pytest.raises(ValidationError):
            RetrieveCandidatesInput(embedding=[0.1] * 768, requesting_customer="   "[:0], k=5)

    def test_both_customer_scoped_tools_require_it(self) -> None:
        """`get_full_answers` reads the same confidential material."""
        for name in ("retrieve_candidates", "get_full_answers"):
            field = TOOLS_BY_NAME[name].input_model.model_fields["requesting_customer"]
            assert field.is_required(), name


class TestSchemasAreDerived:
    def test_every_tool_publishes_both_schemas(self) -> None:
        for tool in TOOLS:
            schema = tool.schema()
            assert schema["name"] == tool.name
            assert schema["input_schema"]["type"] == "object"
            assert "properties" in schema["input_schema"]
            assert "properties" in schema["output_schema"]

    def test_the_schema_names_the_required_fields(self) -> None:
        schema = TOOLS_BY_NAME["retrieve_candidates"].schema()["input_schema"]
        assert "requesting_customer" in schema["required"]
        assert "embedding" in schema["required"]
        # Defaulted fields are not required.
        assert "domain" not in schema.get("required", [])

    def test_schemas_are_json_serialisable(self) -> None:
        import json

        for tool in TOOLS:
            json.dumps(tool.schema())


class TestInputValidation:
    def test_k_is_bounded(self) -> None:
        """An unbounded k is a way to pull the whole corpus through one call."""
        with pytest.raises(ValidationError):
            RetrieveCandidatesInput(embedding=[0.1] * 768, requesting_customer="Meridian", k=1000)
        with pytest.raises(ValidationError):
            RetrieveCandidatesInput(embedding=[0.1] * 768, requesting_customer="Meridian", k=0)

    def test_an_empty_embedding_is_refused(self) -> None:
        with pytest.raises(ValidationError):
            RetrieveCandidatesInput(embedding=[], requesting_customer="Meridian")

    def test_unknown_fields_are_refused(self) -> None:
        with pytest.raises(ValidationError) as caught:
            RetrieveCandidatesInput(
                embedding=[0.1] * 768,
                requesting_customer="Meridian",
                # The field `answers_for_question` used to carry. A caller
                # reaching for it must be told it is gone, not served a result
                # computed without it.
                exclude_confidential=False,  # type: ignore[call-arg]
            )
        assert "exclude_confidential" in str(caught.value)

    def test_entity_type_must_be_a_known_kind(self) -> None:
        with pytest.raises(ValidationError):
            EntityExistsInput.model_validate({"name": "x", "entity_type": "not-a-kind"})
        parsed = EntityExistsInput.model_validate(
            {"name": "x", "entity_type": EntityType.VENDOR.value}
        )
        assert parsed.entity_type is EntityType.VENDOR


class TestSpecShape:
    def test_a_toolspec_is_frozen(self) -> None:
        """The registry is not something a request can edit."""
        tool: ToolSpec = TOOLS[0]
        with pytest.raises(Exception):  # noqa: B017 - dataclasses raise FrozenInstanceError
            tool.name = "other"  # type: ignore[misc]
