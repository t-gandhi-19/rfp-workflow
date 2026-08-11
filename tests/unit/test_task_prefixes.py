"""Task prefixes: applied in one place, required at the call site.

nomic-embed-text is trained with them. Sending raw text is not a style choice —
it measurably degrades retrieval, and it was the root cause of Recall@5 = 0.333
with unanswerable questions matching. These tests pin the mechanism so the
regression cannot return quietly.
"""

from __future__ import annotations

import httpx
import pytest

from src.contracts.embedding import (
    EmbedRole,
    apply_task_prefix,
    embedding_config,
    strip_task_prefix,
    task_prefix,
)
from src.gateway.client import GatewayClient
from src.gateway.fake_embedder import fake_embedding


class TestConfig:
    def test_both_prefixes_are_configured(self) -> None:
        prefixes = embedding_config().model.task_prefixes
        assert prefixes.document == "search_document: "
        assert prefixes.query == "search_query: "

    def test_the_two_roles_differ(self) -> None:
        """If they were equal the whole mechanism would be a no-op."""
        assert task_prefix(EmbedRole.DOCUMENT) != task_prefix(EmbedRole.QUERY)


class TestApplication:
    def test_documents_get_the_document_prefix(self) -> None:
        assert apply_task_prefix(["abc"], EmbedRole.DOCUMENT) == ["search_document: abc"]

    def test_queries_get_the_query_prefix(self) -> None:
        assert apply_task_prefix(["abc"], EmbedRole.QUERY) == ["search_query: abc"]

    def test_every_text_in_a_batch_is_prefixed(self) -> None:
        assert apply_task_prefix(["a", "b"], EmbedRole.DOCUMENT) == [
            "search_document: a",
            "search_document: b",
        ]

    def test_stripping_reverses_either_prefix(self) -> None:
        assert strip_task_prefix("search_document: abc") == "abc"
        assert strip_task_prefix("search_query: abc") == "abc"

    def test_stripping_leaves_unprefixed_text_alone(self) -> None:
        assert strip_task_prefix("abc") == "abc"


class TestTheGatewayAppliesThem:
    """The one place prefixes are applied — never a caller building strings."""

    async def test_the_document_prefix_reaches_the_wire(self) -> None:
        seen: dict[str, object] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            import json

            seen.update(json.loads(request.content))
            return httpx.Response(200, json={"data": [{"index": 0, "embedding": [0.1] * 768}]})

        client = GatewayClient(base_url="http://gw", api_key="k")
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
            await client.embed(
                ["a question"], alias="embed-model", role=EmbedRole.DOCUMENT, client=http
            )
        assert seen["input"] == ["search_document: a question"]

    async def test_the_query_prefix_reaches_the_wire(self) -> None:
        seen: dict[str, object] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            import json

            seen.update(json.loads(request.content))
            return httpx.Response(200, json={"data": [{"index": 0, "embedding": [0.1] * 768}]})

        client = GatewayClient(base_url="http://gw", api_key="k")
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
            await client.embed(
                ["a question"], alias="embed-model", role=EmbedRole.QUERY, client=http
            )
        assert seen["input"] == ["search_query: a question"]

    async def test_role_has_no_default(self) -> None:
        """Embedding a query as a document retrieves badly and says nothing."""
        client = GatewayClient(base_url="http://gw", api_key="k")
        with pytest.raises(TypeError):
            await client.embed(["x"], alias="embed-model")  # type: ignore[call-arg]


class TestTheFakeEmbedderIsPrefixInsensitive:
    """Documented choice: the CI stand-in strips prefixes before hashing.

    The real model embeds the two roles differently. If the stand-in did too, a
    CI query would never match its own corpus entry and the vector-index tests
    would assert nothing. Stripping makes CI a clean plumbing oracle and claims
    nothing about semantics.
    """

    def test_both_roles_hash_identically(self) -> None:
        document = fake_embedding("search_document: landing zones", 768)
        query = fake_embedding("search_query: landing zones", 768)
        assert document == query

    def test_it_still_matches_the_unprefixed_form(self) -> None:
        assert fake_embedding("search_query: abc", 768) == fake_embedding("abc", 768)

    def test_different_texts_still_differ(self) -> None:
        assert fake_embedding("search_query: a", 768) != fake_embedding("search_query: b", 768)
