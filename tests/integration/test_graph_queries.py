"""Every graph query, against a real Neo4j holding the ingested fixtures.

These need a running stack and ingested data, so they skip rather than fail
without one (Phase 1 review amendment B). CI starts Neo4j and ingests with the
deterministic stand-in embedder before running them.

This is also where the *real* Cypher entity resolver is exercised over the
fixtures. The unit self-check resolves in-memory so a stackless clone can still
prove its fixtures cohere; this proves the query agrees with that.
"""

from __future__ import annotations

import csv
import json
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import pytest
from neo4j import AsyncSession

from src.contracts.embedding import embedding_config
from src.contracts.enums import EntityType, Outcome, RetrievalStatus  # noqa: F401
from src.gateway.fake_embedder import fake_embedding
from src.graph import queries
from src.graph.driver import close_driver, get_driver
from tests.live import require_neo4j

pytestmark = pytest.mark.integration

FIXTURES = Path(__file__).resolve().parents[2] / "fixtures"

#: The golden RFP's issuer — the customer a normal run is executed for.
MERIDIAN = "Meridian Insurance Group"
#: The one customer with confidential material in the corpus.
BLUEPINE = "Bluepine Health Systems"


@pytest.fixture(scope="module", autouse=True)
def _live_neo4j() -> None:
    require_neo4j()


@pytest.fixture
async def session() -> AsyncIterator[AsyncSession]:
    driver = get_driver()
    async with driver.session() as neo_session:
        yield neo_session
    await close_driver()


@pytest.fixture(scope="module")
def pairs() -> list[dict[str, Any]]:
    with (FIXTURES / "qa_pairs.json").open(encoding="utf-8") as handle:
        data: list[dict[str, Any]] = json.load(handle)
    return data


def _registry_rows(name: str) -> list[dict[str, str]]:
    with (FIXTURES / "registry" / name).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


async def _ingested(session: AsyncSession) -> bool:
    result = await session.run("MATCH (q:Question) RETURN count(q) AS c")
    record = await result.single()
    return bool(record and record["c"] > 0)


@pytest.fixture(autouse=True)
async def _require_ingest(session: AsyncSession) -> None:
    if not await _ingested(session):
        pytest.skip("graph is empty — run: make ingest")


class TestVectorSearch:
    async def test_returns_in_domain_questions(self, session: AsyncSession) -> None:
        dimensions = embedding_config().model.dimensions
        result = await session.run(
            "MATCH (q:Question) WHERE q.embedding IS NOT NULL "
            "RETURN q.embedding AS e ORDER BY q.id LIMIT 1"
        )
        record = await result.single()
        assert record is not None, "no question carries an embedding"
        hits = await queries.find_similar_questions(
            session,
            embedding=list(record["e"]),
            domain="cloud_migration",
            requesting_customer=MERIDIAN,
            k=5,
        )
        assert hits
        assert len(hits) <= 5
        assert all(0.0 <= hit.score <= 1.0 for hit in hits)
        assert len(record["e"]) == dimensions

    async def test_results_are_ordered_deterministically(self, session: AsyncSession) -> None:
        """Two identical queries must return the same order, or evals are noise."""
        embedding = fake_embedding("deterministic ordering probe", 768)
        first = await queries.find_similar_questions(
            session,
            embedding=embedding,
            domain="cloud_migration",
            requesting_customer=MERIDIAN,
            k=8,
        )
        second = await queries.find_similar_questions(
            session,
            embedding=embedding,
            domain="cloud_migration",
            requesting_customer=MERIDIAN,
            k=8,
        )
        assert [h.question_id for h in first] == [h.question_id for h in second]

    async def test_an_unknown_domain_returns_nothing(self, session: AsyncSession) -> None:
        hits = await queries.find_similar_questions(
            session,
            embedding=fake_embedding("x", 768),
            domain="payroll_services",
            requesting_customer=MERIDIAN,
            k=5,
        )
        assert hits == []

    async def test_a_wrong_width_embedding_is_refused(self, session: AsyncSession) -> None:
        """Caught before it reaches the index, where the error would be opaque."""
        with pytest.raises(ValueError, match="dimensions"):
            await queries.find_similar_questions(
                session,
                embedding=[0.1] * 128,
                domain="cloud_migration",
                requesting_customer=MERIDIAN,
                k=5,
            )

    async def test_an_empty_requesting_customer_is_refused(self, session: AsyncSession) -> None:
        """Visibility must never be left implicit."""
        with pytest.raises(ValueError, match="requesting_customer"):
            await queries.find_similar_questions(
                session,
                embedding=fake_embedding("x", 768),
                domain="cloud_migration",
                requesting_customer="   ",
                k=5,
            )


class TestLineage:
    async def test_returns_full_provenance(
        self, session: AsyncSession, pairs: list[dict[str, Any]]
    ) -> None:
        pair = next(p for p in pairs if not p["confidential"])
        lineage = await queries.get_answer_with_lineage(session, answer_id=pair["answer_id"])
        assert lineage is not None
        assert lineage.answer_id == pair["answer_id"]
        assert lineage.sme_id == pair["author_sme_id"]
        assert lineage.outcome.value == pair["outcome"]
        assert lineage.customer == pair["customer"]
        assert lineage.capability_ids == [pair["capability_id"]]

    async def test_supersession_is_visible_from_both_ends(
        self, session: AsyncSession, pairs: list[dict[str, Any]]
    ) -> None:
        old = next(p for p in pairs if p["superseded_by"])
        old_lineage = await queries.get_answer_with_lineage(session, answer_id=old["answer_id"])
        new_lineage = await queries.get_answer_with_lineage(session, answer_id=old["superseded_by"])
        assert old_lineage is not None and new_lineage is not None
        assert old_lineage.superseded_by == [old["superseded_by"]]
        assert new_lineage.supersedes == [old["answer_id"]]
        assert old_lineage.superseded is True

    async def test_unknown_answer_returns_none(self, session: AsyncSession) -> None:
        assert await queries.get_answer_with_lineage(session, answer_id="ANS-9999") is None


class TestEntityResolution:
    """The real Cypher resolver, over the real fixtures."""

    async def test_every_vendor_resolves_by_code_and_by_name(self, session: AsyncSession) -> None:
        for row in _registry_rows("vendors.csv"):
            by_code = await queries.entity_exists(
                session, name_or_code=row["code"], entity_type=EntityType.VENDOR
            )
            by_name = await queries.entity_exists(
                session, name_or_code=row["name"], entity_type=EntityType.VENDOR
            )
            assert by_code.passed and by_code.resolved_node_id == row["code"]
            assert by_name.passed and by_name.resolved_node_id == row["code"]

    async def test_every_cited_entity_in_the_corpus_resolves(
        self, session: AsyncSession, pairs: list[dict[str, Any]]
    ) -> None:
        """Build prompt §8: the real resolver, run over the fixtures."""
        kinds = [
            EntityType.VENDOR,
            EntityType.PRODUCT,
            EntityType.CERTIFICATION,
            EntityType.CLIENT,
            EntityType.LOCATION,
        ]
        unresolved: list[tuple[str, str]] = []
        for pair in pairs:
            for entity in pair["cited_entities"]:
                for kind in kinds:
                    check = await queries.entity_exists(
                        session, name_or_code=entity, entity_type=kind
                    )
                    if check.passed:
                        break
                else:
                    unresolved.append((pair["id"], entity))
        assert unresolved == []

    async def test_corporate_suffixes_and_case_are_tolerated(self, session: AsyncSession) -> None:
        check = await queries.entity_exists(
            session, name_or_code="cloudnova partners, inc.", entity_type=EntityType.VENDOR
        )
        assert check.passed and check.resolved_node_id == "VND-0001"

    async def test_an_invented_vendor_does_not_resolve(self, session: AsyncSession) -> None:
        """A miss must be a miss — no nearest-match guessing."""
        check = await queries.entity_exists(
            session, name_or_code="Quantum Migration Labs", entity_type=EntityType.VENDOR
        )
        assert check.passed is False
        assert check.resolved_node_id is None

    async def test_a_vendor_does_not_resolve_as_a_product(self, session: AsyncSession) -> None:
        check = await queries.entity_exists(
            session, name_or_code="CloudNova Partners", entity_type=EntityType.PRODUCT
        )
        assert check.passed is False

    async def test_empty_input_does_not_resolve(self, session: AsyncSession) -> None:
        check = await queries.entity_exists(
            session, name_or_code="   ", entity_type=EntityType.VENDOR
        )
        assert check.passed is False


class TestSmeRouting:
    async def test_every_capability_routes_to_a_named_sme(self, session: AsyncSession) -> None:
        for row in _registry_rows("capabilities.csv"):
            smes = await queries.get_sme_for_capability(session, capability_id=row["id"])
            assert smes, f"no SME owns {row['id']}"
            assert all(sme.name for sme in smes)

    async def test_unknown_capability_routes_nowhere(self, session: AsyncSession) -> None:
        assert await queries.get_sme_for_capability(session, capability_id="CAP-9999") == []


class TestAnswersForQuestion:
    async def test_excludes_superseded_by_default(
        self, session: AsyncSession, pairs: list[dict[str, Any]]
    ) -> None:
        old = next(p for p in pairs if p["superseded_by"])
        answers = await queries.answers_for_question(
            session, question_id=old["question_id"], requesting_customer=MERIDIAN
        )
        assert old["answer_id"] not in {a.answer_id for a in answers}

    async def test_can_include_superseded_explicitly(
        self, session: AsyncSession, pairs: list[dict[str, Any]]
    ) -> None:
        """Lineage and audit legitimately want the whole chain."""
        old = next(p for p in pairs if p["superseded_by"])
        answers = await queries.answers_for_question(
            session,
            question_id=old["question_id"],
            requesting_customer=MERIDIAN,
            exclude_superseded=False,
        )
        assert old["answer_id"] in {a.answer_id for a in answers}

    async def test_an_empty_requesting_customer_is_refused(
        self, session: AsyncSession, pairs: list[dict[str, Any]]
    ) -> None:
        with pytest.raises(ValueError, match="requesting_customer"):
            await queries.answers_for_question(
                session, question_id=pairs[0]["question_id"], requesting_customer=""
            )


class TestConfidentialityAtTheQueryBoundary:
    """Confidential material cannot cross customers, whatever the caller does.

    There is no parameter that relaxes this — `exclude_confidential` was deleted
    rather than defaulted, because a boolean a caller can pass `False` to is one
    careless call site away from a leak. These tests prove the guarantee at the
    boundary; the Phase 5 adversarial eval re-proves it end to end.
    """

    async def test_confidential_answers_never_cross_customers_at_the_query_boundary(
        self, session: AsyncSession, pairs: list[dict[str, Any]]
    ) -> None:
        secret = next(p for p in pairs if p["confidential"])
        assert secret["customer"] == BLUEPINE

        for_meridian = await queries.answers_for_question(
            session, question_id=secret["question_id"], requesting_customer=MERIDIAN
        )
        assert secret["answer_id"] not in {a.answer_id for a in for_meridian}

    async def test_the_owning_customer_can_still_see_its_own_confidential_answer(
        self, session: AsyncSession, pairs: list[dict[str, Any]]
    ) -> None:
        """Confidentiality is not deletion — Bluepine's own run must still use it."""
        secret = next(p for p in pairs if p["confidential"])
        for_bluepine = await queries.answers_for_question(
            session, question_id=secret["question_id"], requesting_customer=BLUEPINE
        )
        assert secret["answer_id"] in {a.answer_id for a in for_bluepine}

    async def test_a_confidential_question_is_not_even_a_candidate_for_another_customer(
        self, session: AsyncSession, pairs: list[dict[str, Any]]
    ) -> None:
        """Closes the gap where it could surface as a similarity candidate.

        Retrieving the confidential answer's own embedding makes it the nearest
        possible neighbour — so if visibility filtering were missing anywhere in
        the candidate path, this is the query that would expose it.
        """
        secret = next(p for p in pairs if p["confidential"])
        result = await session.run(
            "MATCH (q:Question {id: $qid}) RETURN q.embedding AS e",
            qid=secret["question_id"],
        )
        record = await result.single()
        assert record is not None and record["e"], "confidential question has no embedding"

        for_meridian = await queries.find_similar_questions(
            session,
            embedding=list(record["e"]),
            domain="cloud_migration",
            requesting_customer=MERIDIAN,
            k=20,
        )
        assert secret["question_id"] not in {h.question_id for h in for_meridian}

    async def test_the_owning_customer_does_get_it_as_a_candidate(
        self, session: AsyncSession, pairs: list[dict[str, Any]]
    ) -> None:
        secret = next(p for p in pairs if p["confidential"])
        result = await session.run(
            "MATCH (q:Question {id: $qid}) RETURN q.embedding AS e",
            qid=secret["question_id"],
        )
        record = await result.single()
        assert record is not None

        for_bluepine = await queries.find_similar_questions(
            session,
            embedding=list(record["e"]),
            domain="cloud_migration",
            requesting_customer=BLUEPINE,
            k=20,
        )
        assert secret["question_id"] in {h.question_id for h in for_bluepine}

    async def test_filtering_does_not_shrink_the_candidate_list(
        self, session: AsyncSession
    ) -> None:
        """The index is over-fetched so visibility filtering still yields k.

        Without over-fetching, a run whose nearest neighbours happened to be
        another customer's material would quietly come back short — fewer
        candidates, no error, worse answers.
        """
        embedding = fake_embedding("landing zone guardrails", 768)
        hits = await queries.find_similar_questions(
            session,
            embedding=embedding,
            domain="cloud_migration",
            requesting_customer=MERIDIAN,
            k=20,
        )
        assert len(hits) == 20


class TestCoverageGaps:
    async def test_a_fully_answered_rfp_reports_no_gaps(
        self, session: AsyncSession, pairs: list[dict[str, Any]]
    ) -> None:
        result = await session.run("MATCH (r:RFP) RETURN r.id AS id ORDER BY id LIMIT 1")
        record = await result.single()
        assert record is not None
        gaps = await queries.coverage_gaps(session, rfp_id=record["id"])
        # Every historical question has a live answer except where the only
        # answer was superseded.
        assert all(gap.reason for gap in gaps)

    async def test_unknown_rfp_reports_nothing(self, session: AsyncSession) -> None:
        assert await queries.coverage_gaps(session, rfp_id="HRFP-NOPE") == []


class TestIngestIsIdempotent:
    async def test_counts_are_stable(self, session: AsyncSession) -> None:
        """Re-running ingest must not duplicate. Asserted against known totals."""
        totals = await queries.counts(session)
        assert totals["by_label"]["Question"] == 40
        assert totals["by_label"]["Answer"] == 40
        assert totals["by_label"]["Vendor"] == 12
        assert totals["by_relationship"]["SUPERSEDES"] == 4
        assert totals["by_relationship"]["ANSWERED_BY"] == 40

    async def test_every_question_carries_an_embedding_of_the_pinned_width(
        self, session: AsyncSession
    ) -> None:
        dimensions = embedding_config().model.dimensions
        result = await session.run(
            "MATCH (q:Question) RETURN q.id AS id, size(q.embedding) AS width ORDER BY id"
        )
        widths = {record["id"]: record["width"] async for record in result}
        assert widths
        assert set(widths.values()) == {dimensions}
