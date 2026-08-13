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

from src.contracts.embedding import EmbedRole, embedding_config
from src.contracts.enums import EntityType, Outcome, RetrievalStatus  # noqa: F401
from src.evals.retrieval import golden_questions
from src.gateway.client import GatewayClient
from src.gateway.fake_embedder import fake_embedding, fake_embeddings, fake_embeddings_enabled
from src.graph import queries
from src.graph.driver import close_driver, get_driver
from src.retrieval.calibration import (
    calibration_corpus,
    cosine,
    indexed_ids,
    load_for_current_corpus,
)
from src.retrieval.units import EPSILON as UNITS_EPSILON
from src.retrieval.units import check_units_agreement
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


async def embed_texts(texts: list[str], role: EmbedRole) -> list[list[float]]:
    """Embed for one side of the geometry, matching how calibration does it.

    Uses the stand-in when CI has opted in, so the probe-agreement test runs in
    both worlds — the units question is about scales, not semantics, and the
    stand-in's vectors have a real geometry to disagree about.
    """
    config = embedding_config()
    if fake_embeddings_enabled():
        return fake_embeddings(texts, config.model.dimensions, role=role)
    client = GatewayClient.from_env()
    vectors: list[list[float]] = []
    for start in range(0, len(texts), 16):
        vectors.extend(
            await client.embed(texts[start : start + 16], alias=config.model.alias, role=role)
        )
    return vectors


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
        embedding = fake_embedding("deterministic ordering probe", 768, role=EmbedRole.QUERY)
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

    async def test_no_paraphrase_is_ever_a_candidate_for_any_golden_question(
        self, session: AsyncSession
    ) -> None:
        """AMENDMENT P, ZERO TOLERANCE. Not 'few'. None, for all twenty.

        A paraphrase carries no answer of its own — it points at the one its
        original already has — so a paraphrase in a candidate list is a
        duplicate of a result already there, competing for a rank it cannot
        deserve. It is also the population the calibration artifact says is not
        on the document side, so one appearing would mean the floor was derived
        over a distribution retrieval does not actually sample.
        """
        with (FIXTURES / "answer_key_manual.json").open(encoding="utf-8") as handle:
            golden = json.load(handle)["questions"]
        with (FIXTURES / "question_paraphrases.json").open(encoding="utf-8") as handle:
            paraphrase_ids = {row["question_id"] for row in json.load(handle)}

        offenders: list[tuple[str, str]] = []
        for question in golden:
            hits = await queries.find_similar_questions(
                session,
                embedding=fake_embedding(question["text"], 768, role=EmbedRole.QUERY),
                domain="cloud_migration",
                requesting_customer=MERIDIAN,
                # Deliberately far more than retrieval ever asks for: a
                # paraphrase absent from the top 5 but present at rank 19 is
                # still in the candidate space this forbids.
                k=40,
            )
            offenders.extend(
                (question["number"], hit.question_id)
                for hit in hits
                if hit.question_id in paraphrase_ids
            )
        assert offenders == [], f"paraphrases surfaced as candidates: {offenders}"

    async def test_paraphrases_hold_no_embedding_at_all(self, session: AsyncSession) -> None:
        """The primary mechanism, checked directly rather than through a query.

        The filter in `find_similar_questions` is belt and braces; this is the
        braces. A paraphrase with a vector is in the index whatever any query
        says, and the calibration artifact's population would be wrong.
        """
        result = await session.run(
            "MATCH (q:Question:Paraphrase) RETURN count(q) AS total, count(q.embedding) AS embedded"
        )
        record = await result.single()
        assert record is not None
        assert record["total"] == 108
        assert record["embedded"] == 0

    async def test_exactly_the_originals_are_indexed(self, session: AsyncSession) -> None:
        """The D18 document side, as a count: 40 indexed originals, no more."""
        result = await session.run(
            "MATCH (q:Question) WHERE q.embedding IS NOT NULL "
            "RETURN count(q) AS embedded, count(CASE WHEN q:Paraphrase THEN 1 END) AS paraphrases"
        )
        record = await result.single()
        assert record is not None
        assert record["embedded"] == 40
        assert record["paraphrases"] == 0

    async def test_the_returned_score_is_a_cosine_not_the_index_score(
        self, session: AsyncSession
    ) -> None:
        """The conversion, settled by the only authority on it.

        A unit test can prove `2 * score - 1` inverts `(1 + cos) / 2`. It cannot
        prove Neo4j normalises that way — and that assumption is what the floor
        rests on, so it is pinned against the live index.

        Compares the score `find_similar_questions` returns with the cosine
        computed directly from the two stored vectors.

        THE TOLERANCE IS SET FROM WHAT THIS HAS TO DISCRIMINATE, not from an
        observed difference. The two hypotheses are "the score is cos" and "the
        score is (1 + cos) / 2"; at a cosine of 0.09 those are 0.09 and 0.545,
        so they are separated by roughly 0.45. Anything comfortably under that
        distinguishes them.

        0.01 is therefore generous for the float32 noise — Neo4j stores vectors
        as float32 while this dot product is float64, and the conversion doubles
        that absolute error — while still being ~45x tighter than the fault it
        exists to catch. An earlier 1e-3 was fitted to a single observation on
        the real corpus and failed in CI at 1.4e-3 on the stand-in's vectors,
        which is a tolerance describing one measurement rather than a claim.

        That same rounding is why retrieval's cosine differs slightly from
        calibration's; it is far below the 0.2788 width of the band.
        """
        # The gap between the two hypotheses, so the assertion below is known to
        # be capable of telling them apart.
        discrimination = 0.01
        result = await session.run(
            "MATCH (q:Question) WHERE q.embedding IS NOT NULL "
            "RETURN q.id AS id, q.embedding AS e ORDER BY q.id LIMIT 1"
        )
        record = await result.single()
        assert record is not None
        probe = list(record["e"])

        hits = await queries.find_similar_questions(
            session,
            embedding=probe,
            domain="cloud_migration",
            requesting_customer=MERIDIAN,
            k=5,
        )
        assert hits

        vectors = await session.run(
            "MATCH (q:Question) WHERE q.id IN $ids RETURN q.id AS id, q.embedding AS e",
            ids=[hit.question_id for hit in hits],
        )
        stored = {row["id"]: list(row["e"]) async for row in vectors}

        for hit in hits:
            expected = sum(a * b for a, b in zip(probe, stored[hit.question_id], strict=True))
            assert hit.score == pytest.approx(expected, abs=discrimination), (
                f"{hit.question_id}: returned {hit.score:.6f}, true cosine {expected:.6f}. "
                "If these differ by roughly (1+cos)/2, the index normalisation changed."
            )
            # The other hypothesis, ruled out explicitly rather than by
            # implication — except where the two coincide, at cos == 1.0.
            unconverted = (1.0 + expected) / 2.0
            if abs(unconverted - expected) > discrimination:
                assert abs(hit.score - unconverted) > discrimination, (
                    f"{hit.question_id}: the score matches the UNCONVERTED index value "
                    f"{unconverted:.6f}, so the conversion is not being applied"
                )

    async def test_the_scores_span_the_band_calibration_measured(
        self, session: AsyncSession
    ) -> None:
        """A sanity check that the converted numbers are in the right place.

        Unconverted index scores on this corpus sat around 0.81-0.84 — above the
        same-subject median of 0.7718, which is what made every candidate
        saturate. Converted, an unrelated pair should land near the background
        median, well under that.
        """
        result = await session.run(
            "MATCH (q:Question) WHERE q.embedding IS NOT NULL "
            "RETURN q.embedding AS e ORDER BY q.id LIMIT 1"
        )
        record = await result.single()
        assert record is not None

        hits = await queries.find_similar_questions(
            session,
            embedding=list(record["e"]),
            domain="cloud_migration",
            requesting_customer=MERIDIAN,
            k=20,
        )
        # Near-duplicates are excluded, not assumed absent: the four
        # supersession chains hold two records with BYTE-IDENTICAL question
        # text, so a probe that happens to be a chain member has a legitimate
        # twin at cosine ~1.0 in both the real and the stand-in geometry.
        # Everything else is a different subject and must sit in the background
        # band rather than above it.
        others = [hit.score for hit in hits if hit.score < 0.99]
        assert others, "expected hits beyond the probe and its duplicates"
        assert max(others) < 0.75, f"non-duplicate hits reach {max(others):.4f}; band looks wrong"

    async def test_the_two_similarity_paths_agree_on_units(self, session: AsyncSession) -> None:
        """Amendment S, at the level preflight runs it."""
        samples = await check_units_agreement(session)
        assert samples, "the agreement check compared nothing"
        worst = max(sample.delta for sample in samples)
        assert worst <= UNITS_EPSILON, f"worst disagreement {worst:.2e} exceeds {UNITS_EPSILON}"

    async def test_a_d19_probe_lands_the_same_through_both_paths(
        self, session: AsyncSession
    ) -> None:
        """The agreement that actually matters, on the number that gates.

        The units check above compares raw similarities. This compares what the
        two paths CONCLUDE: a D19 probe's margin against the derived floor,
        computed the way `scripts/calibrate` computes it (embed the corpus
        directly, dot product, calibrate) and the way retrieval computes it
        (through the vector index). Those were the two subsystems that disagreed,
        and the margin is what the tier-2 ratchet gates on — so agreeing on
        cosines but not on margins would still be a defect.

        Probe 2.7 is used because it is an unanswerable: it must land BELOW the
        floor, so a units error shows up as a sign change rather than as a small
        numeric drift.

        THE TOLERANCE IS THE CORROBORATION BAND, measured rather than chosen.
        When the units fix landed, the eval's probe margins reproduced the
        commissioning run's across all five probes with deltas of 0.0026, 0.0030,
        0.0060, 0.0141 and 0.0246 — a spread of 0.003 to 0.025 arising from
        float32 storage, the graph path's extra hop, and the two paths embedding
        from the same text at different times. 0.05 is twice the largest of those
        and still far below the ~0.3 margin the probe carries, so it accepts the
        known spread while failing any real divergence.
        """
        artifact = load_for_current_corpus()
        probe = next(q for q in golden_questions() if q["number"] == "2.7")

        # Calibration's path: embed both sides directly, dot product, calibrate.
        documents = sorted(indexed_ids(calibration_corpus()))
        text_by_id = {row["question_id"]: row["question"] for row in calibration_corpus()}
        document_vectors = await embed_texts(
            [text_by_id[qid] for qid in documents], EmbedRole.DOCUMENT
        )
        query_vector = (await embed_texts([probe["text"]], EmbedRole.QUERY))[0]
        best_direct = max(cosine(query_vector, vector) for vector in document_vectors)
        direct_margin = artifact.derived_floor - artifact.calibrated(best_direct)

        # Retrieval's path: through the vector index.
        hits = await queries.find_similar_questions(
            session,
            embedding=query_vector,
            domain="cloud_migration",
            requesting_customer=MERIDIAN,
            k=20,
        )
        assert hits, "the probe returned no candidates at all"
        best_index = max(hit.score for hit in hits)
        index_margin = artifact.derived_floor - artifact.calibrated(best_index)

        # Both must agree that it is below the floor — the sign is the gate.
        assert direct_margin > 0, f"probe 2.7 cleared the floor on the direct path: {direct_margin}"
        assert index_margin > 0, f"probe 2.7 cleared the floor on the index path: {index_margin}"
        assert abs(direct_margin - index_margin) <= 0.05, (
            f"the two paths disagree on probe 2.7's margin: direct {direct_margin:+.4f} "
            f"vs index {index_margin:+.4f}. The tier-2 ratchet gates on this number, so "
            "the path that derives it and the path that applies it must agree."
        )

    async def test_an_unknown_domain_returns_nothing(self, session: AsyncSession) -> None:
        hits = await queries.find_similar_questions(
            session,
            embedding=fake_embedding("x", 768, role=EmbedRole.QUERY),
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
                embedding=fake_embedding("x", 768, role=EmbedRole.QUERY),
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

    async def test_every_paraphrase_inherits_its_original_s_confidentiality_owner(
        self, session: AsyncSession
    ) -> None:
        """AMENDMENT P's other half, proved through the graph, not the fixtures.

        Excluding paraphrases from retrieval does NOT make their ownership
        irrelevant. `find_similar_questions` decides confidentiality by walking
        (question)-[:ASKED_IN]->(:RFP)-[:ISSUED_BY]->(owner), and `answers_for_
        question` walks the same path — so a paraphrase hung off a different
        customer's RFP would make its original's answer reachable through a
        synonym, whether or not the paraphrase itself can be a candidate.

        Asserted as an EQUALITY over all 108, resolved by the graph. The fixture
        self-check proves the JSON agrees with itself; this proves ingest built
        the edges that agreement depends on.
        """
        result = await session.run(
            """
            MATCH (p:Question:Paraphrase)-[:PARAPHRASE_OF]->(original:Question)
            MATCH (p)-[:ASKED_IN]->(:RFP)-[:ISSUED_BY]->(pOwner:Customer)
            MATCH (original)-[:ASKED_IN]->(:RFP)-[:ISSUED_BY]->(oOwner:Customer)
            RETURN p.id AS paraphrase_id, pOwner.name AS paraphrase_owner,
                   oOwner.name AS original_owner
            ORDER BY paraphrase_id
            """
        )
        rows = [record.data() async for record in result]
        assert len(rows) == 108, "every paraphrase must resolve an owner on both sides"
        mismatched = [r for r in rows if r["paraphrase_owner"] != r["original_owner"]]
        assert mismatched == [], f"paraphrases owned by another customer: {mismatched}"

    async def test_the_bluepine_paraphrases_are_the_named_case(
        self, session: AsyncSession, pairs: list[dict[str, Any]]
    ) -> None:
        """The one family where inheritance failing would be an actual leak.

        Bluepine owns the corpus's only confidential answer, and its family now
        carries three paraphrases. If any of them were asked in another
        customer's RFP, ANS-0014 would become reachable to that customer through
        a reworded question.
        """
        secret = next(p for p in pairs if p["confidential"])
        assert secret["customer"] == BLUEPINE

        result = await session.run(
            """
            MATCH (p:Question:Paraphrase)-[:PARAPHRASE_OF]->(:Question {id: $qid})
            MATCH (p)-[:ASKED_IN]->(:RFP)-[:ISSUED_BY]->(owner:Customer)
            MATCH (p)-[:ANSWERED_BY]->(a:Answer)
            RETURN p.id AS id, owner.name AS owner, collect(a.id) AS answer_ids
            ORDER BY id
            """,
            qid=secret["question_id"],
        )
        rows = [record.data() async for record in result]
        assert len(rows) == 3, "the Bluepine family carries three paraphrases"
        assert {r["owner"] for r in rows} == {BLUEPINE}
        assert all(secret["answer_id"] in r["answer_ids"] for r in rows)

    async def test_the_confidential_answer_stays_unreachable_through_a_paraphrase(
        self, session: AsyncSession, pairs: list[dict[str, Any]]
    ) -> None:
        """The leak stated as an outcome, through the retrieval boundary itself.

        Embedding a Bluepine paraphrase's own text is the strongest possible
        query for reaching ANS-0014 by synonym. Meridian must get neither the
        paraphrase (amendment P) nor the answer (confidentiality).
        """
        secret = next(p for p in pairs if p["confidential"])
        with (FIXTURES / "question_paraphrases.json").open(encoding="utf-8") as handle:
            rephrased = [
                row for row in json.load(handle) if row["answer_id"] == secret["answer_id"]
            ]
        assert rephrased, "expected paraphrases of the confidential question"

        for row in rephrased:
            hits = await queries.find_similar_questions(
                session,
                embedding=fake_embedding(row["question"], 768, role=EmbedRole.QUERY),
                domain="cloud_migration",
                requesting_customer=MERIDIAN,
                k=20,
            )
            assert row["question_id"] not in {hit.question_id for hit in hits}
            reachable = {answer_id for hit in hits for answer_id in hit.answer_ids}
            assert secret["answer_id"] not in reachable, (
                f"{secret['answer_id']} reachable to {MERIDIAN} via paraphrase {row['id']}"
            )

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
        embedding = fake_embedding("landing zone guardrails", 768, role=EmbedRole.QUERY)
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
        """Re-running ingest must not duplicate. Asserted against known totals.

        The Question and ANSWERED_BY figures were STALE, and had been since the
        paraphrases landed: they still read 40, which is the count of Q&A pairs,
        while the graph had held 148 Question nodes and 148 ANSWERED_BY edges
        ever since. The test was failing on the real graph and nothing said so,
        because the integration suite needs a live stack and the verification
        habit quotes tests/unit and tests/security.

        Corrected rather than relaxed, and split so each number says which
        population it describes.
        """
        totals = await queries.counts(session)
        # 40 originals + 108 paraphrases, and every paraphrase carries both
        # labels (amendment P) so the two counts deliberately overlap.
        assert totals["by_label"]["Question"] == 148
        assert totals["by_label"]["Paraphrase"] == 108
        # A paraphrase carries no answer of its own, so this stays at 40.
        assert totals["by_label"]["Answer"] == 40
        assert totals["by_label"]["Vendor"] == 12
        assert totals["by_relationship"]["SUPERSEDES"] == 4
        # 148: each paraphrase points at the answer its original already has.
        assert totals["by_relationship"]["ANSWERED_BY"] == 148
        assert totals["by_relationship"]["PARAPHRASE_OF"] == 108

    async def test_every_indexed_question_carries_an_embedding_of_the_pinned_width(
        self, session: AsyncSession
    ) -> None:
        """Amendment P narrows this from "every question" to "every INDEXED one".

        Paraphrases are calibration-only: they are embedded by `scripts/
        calibrate` from the fixtures, on the query side, and never enter the
        vector index. So "no embedding" is the correct state for 108 of the 148
        nodes, and the assertion has to say which population it means or it
        would fail on a correct graph.
        """
        dimensions = embedding_config().model.dimensions
        result = await session.run(
            "MATCH (q:Question) WHERE NOT q:Paraphrase "
            "RETURN q.id AS id, size(q.embedding) AS width ORDER BY id"
        )
        widths = {record["id"]: record["width"] async for record in result}
        assert len(widths) == 40
        assert set(widths.values()) == {dimensions}
