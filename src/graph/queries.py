"""The complete set of graph reads. Agents never write Cypher (CLAUDE.md rule 4).

Every query here is a static string with bound parameters. There is no
query-builder and no raw-query tool anywhere in the system; the MCP server
exposes exactly these functions and Keycloak roles gate them.

Two properties are load-bearing throughout:

**Determinism.** Every result set has a total ordering, with an id as the final
tie-break. Neo4j does not promise an order otherwise, and a retrieval pipeline
whose candidate order varies between identical runs cannot be evaluated.

**Filtering in the query, not after it.** Supersession and confidentiality are
excluded inside Cypher rather than in Python, so a caller cannot forget to
apply them.
"""

from __future__ import annotations

from typing import Any

from neo4j import AsyncSession
from pydantic import BaseModel, ConfigDict, Field

from src.contracts import EntityCheckResult
from src.contracts.embedding import embedding_config
from src.contracts.enums import EntityType, Outcome
from src.graph.driver import normalise_name


class SimilarQuestion(BaseModel):
    """One vector-index hit, before any graph multiplier is applied."""

    model_config = ConfigDict(extra="forbid")

    question_id: str
    text: str
    normalized_text: str
    score: float = Field(ge=0.0, le=1.0)
    answer_ids: list[str] = Field(default_factory=list)


class AnswerRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    answer_id: str
    text: str
    answer_date: str
    confidential: bool
    superseded: bool
    outcome: Outcome
    customer: str | None = None
    sme_id: str | None = None


class AnswerLineage(BaseModel):
    """A citation's full provenance, one hop away (build prompt §7)."""

    model_config = ConfigDict(extra="forbid")

    answer_id: str
    text: str
    answer_date: str
    confidential: bool
    superseded: bool
    outcome: Outcome
    question_id: str | None = None
    question_text: str | None = None
    customer: str | None = None
    sme_id: str | None = None
    sme_name: str | None = None
    capability_ids: list[str] = Field(default_factory=list)
    evidence_codes: list[str] = Field(default_factory=list)
    supersedes: list[str] = Field(default_factory=list)
    superseded_by: list[str] = Field(default_factory=list)


class SMERecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sme_id: str
    name: str
    capability_ids: list[str] = Field(default_factory=list)


class CoverageGap(BaseModel):
    """A question nobody owns: no answer, and no SME owning its capability."""

    model_config = ConfigDict(extra="forbid")

    question_id: str
    text: str
    capability_ids: list[str] = Field(default_factory=list)
    reason: str


# ---------------------------------------------------------------------------
# Queries
# ---------------------------------------------------------------------------

_FIND_SIMILAR = """
CALL db.index.vector.queryNodes($index_name, $k, $embedding)
YIELD node, score
MATCH (node)-[:BELONGS_TO]->(:Domain {key: $domain})
OPTIONAL MATCH (node)-[:ANSWERED_BY]->(a:Answer)
WITH node, score, collect(DISTINCT a.id) AS answer_ids
RETURN node.id            AS question_id,
       node.text          AS text,
       node.normalized_text AS normalized_text,
       score              AS score,
       [x IN answer_ids WHERE x IS NOT NULL] AS answer_ids
ORDER BY score DESC, question_id ASC
"""


async def find_similar_questions(
    session: AsyncSession,
    *,
    embedding: list[float],
    domain: str,
    k: int,
) -> list[SimilarQuestion]:
    """Top-k in-domain questions by cosine similarity.

    The domain filter is part of the same query as the vector search — that is
    the point of a native vector index, and why there is no separate vector
    store to drift out of sync.
    """
    if k <= 0:
        return []
    expected = embedding_config().model.dimensions
    if len(embedding) != expected:
        raise ValueError(f"embedding has {len(embedding)} dimensions, index expects {expected}")
    result = await session.run(
        _FIND_SIMILAR,
        index_name=embedding_config().index.name,
        k=k,
        embedding=embedding,
        domain=domain,
    )
    records = [record.data() async for record in result]
    return [SimilarQuestion.model_validate(record) for record in records]


_ANSWER_LINEAGE = """
MATCH (a:Answer {id: $answer_id})
OPTIONAL MATCH (q:Question)-[:ANSWERED_BY]->(a)
OPTIONAL MATCH (a)-[:AUTHORED_BY]->(sme:SME)
OPTIONAL MATCH (a)-[:RESULTED_IN]->(o:Outcome)
OPTIONAL MATCH (a)-[:EVIDENCED_BY]->(cs:CaseStudy)
OPTIONAL MATCH (a)-[:SUPERSEDES]->(older:Answer)
OPTIONAL MATCH (a)<-[:SUPERSEDES]-(newer:Answer)
OPTIONAL MATCH (q)-[:BELONGS_TO]->(cap:Capability)
OPTIONAL MATCH (q)-[:ASKED_IN]->(:RFP)-[:ISSUED_BY]->(cust:Customer)
RETURN a.id                AS answer_id,
       a.text              AS text,
       a.answer_date       AS answer_date,
       coalesce(a.confidential, false) AS confidential,
       coalesce(a.superseded, false)   AS superseded,
       coalesce(o.value, 'unknown')    AS outcome,
       q.id                AS question_id,
       q.text              AS question_text,
       cust.name           AS customer,
       sme.id              AS sme_id,
       sme.name            AS sme_name,
       [x IN collect(DISTINCT cap.id)   WHERE x IS NOT NULL] AS capability_ids,
       [x IN collect(DISTINCT cs.code)  WHERE x IS NOT NULL] AS evidence_codes,
       [x IN collect(DISTINCT older.id) WHERE x IS NOT NULL] AS supersedes,
       [x IN collect(DISTINCT newer.id) WHERE x IS NOT NULL] AS superseded_by
"""


async def get_answer_with_lineage(session: AsyncSession, *, answer_id: str) -> AnswerLineage | None:
    """Everything a reviewer needs to judge a citation, in one hop."""
    result = await session.run(_ANSWER_LINEAGE, answer_id=answer_id)
    record = await result.single()
    if record is None:
        return None
    data = dict(record.data())
    for key in ("capability_ids", "evidence_codes", "supersedes", "superseded_by"):
        data[key] = sorted(data.get(key) or [])
    return AnswerLineage.model_validate(data)


_ENTITY_BY_CODE = """
MATCH (n:RegistryEntity)
WHERE n.entity_kind = $kind AND n.code = $needle
RETURN n.code AS node_id
ORDER BY node_id ASC
LIMIT 1
"""

_ENTITY_BY_EXACT_NAME = """
MATCH (n:RegistryEntity)
WHERE n.entity_kind = $kind AND n.normalised_name = $normalised
RETURN n.code AS node_id
ORDER BY node_id ASC
LIMIT 1
"""

_ENTITY_BY_PARTIAL_NAME = """
MATCH (n:RegistryEntity)
WHERE n.entity_kind = $kind
  AND (n.normalised_name STARTS WITH $normalised OR $normalised STARTS WITH n.normalised_name)
RETURN n.code AS node_id, size(n.normalised_name) AS name_length
ORDER BY name_length ASC, node_id ASC
LIMIT 1
"""


async def entity_exists(
    session: AsyncSession, *, name_or_code: str, entity_type: EntityType
) -> EntityCheckResult:
    """Resolve a named entity against the closed-world registry.

    Three passes, most precise first: exact code, exact normalised name, then a
    prefix match in either direction. Every pass orders deterministically and
    takes one row, so the same input always resolves to the same node — an
    entity checker that resolved differently between runs would make grounding
    failures irreproducible.

    A miss is a hard fail for the answer that named it (build prompt §15), so
    this deliberately does not guess: no edit-distance, no "closest" match.
    """
    needle = name_or_code.strip()
    if not needle:
        return EntityCheckResult(entity_text=name_or_code, entity_type=entity_type, passed=False)

    kind = entity_type.value
    normalised = normalise_name(needle)

    for query, params in (
        (_ENTITY_BY_CODE, {"kind": kind, "needle": needle}),
        (_ENTITY_BY_EXACT_NAME, {"kind": kind, "normalised": normalised}),
        (_ENTITY_BY_PARTIAL_NAME, {"kind": kind, "normalised": normalised}),
    ):
        if not normalised and query is not _ENTITY_BY_CODE:
            continue
        result = await session.run(query, **params)
        record = await result.single()
        if record is not None:
            return EntityCheckResult(
                entity_text=name_or_code,
                entity_type=entity_type,
                resolved_node_id=str(record["node_id"]),
                passed=True,
            )

    return EntityCheckResult(entity_text=name_or_code, entity_type=entity_type, passed=False)


_SME_FOR_CAPABILITY = """
MATCH (s:SME)-[:OWNS]->(:Capability {id: $capability_id})
OPTIONAL MATCH (s)-[:OWNS]->(c:Capability)
WITH s, [x IN collect(DISTINCT c.id) WHERE x IS NOT NULL] AS capability_ids
RETURN s.id AS sme_id, s.name AS name, capability_ids
ORDER BY sme_id ASC
"""


async def get_sme_for_capability(session: AsyncSession, *, capability_id: str) -> list[SMERecord]:
    """Who owns this capability. An escalation carries a name, not a queue."""
    result = await session.run(_SME_FOR_CAPABILITY, capability_id=capability_id)
    records = [record.data() async for record in result]
    for record in records:
        record["capability_ids"] = sorted(record.get("capability_ids") or [])
    return [SMERecord.model_validate(record) for record in records]


_ANSWERS_FOR_QUESTION = """
MATCH (q:Question {id: $question_id})-[:ANSWERED_BY]->(a:Answer)
WHERE ($exclude_superseded = false OR coalesce(a.superseded, false) = false)
  AND ($exclude_confidential = false OR coalesce(a.confidential, false) = false)
OPTIONAL MATCH (a)-[:RESULTED_IN]->(o:Outcome)
OPTIONAL MATCH (a)-[:AUTHORED_BY]->(sme:SME)
OPTIONAL MATCH (q)-[:ASKED_IN]->(:RFP)-[:ISSUED_BY]->(cust:Customer)
RETURN a.id          AS answer_id,
       a.text        AS text,
       a.answer_date AS answer_date,
       coalesce(a.confidential, false) AS confidential,
       coalesce(a.superseded, false)   AS superseded,
       coalesce(o.value, 'unknown')    AS outcome,
       cust.name     AS customer,
       sme.id        AS sme_id
ORDER BY answer_date DESC, answer_id ASC
"""


async def answers_for_question(
    session: AsyncSession,
    *,
    question_id: str,
    exclude_superseded: bool = True,
    exclude_confidential: bool = True,
) -> list[AnswerRecord]:
    """Answers attached to a question, newest first.

    Both exclusions default to on and are applied inside Cypher. A superseded
    answer reaching a draft is a staleness bug the eval harness scores at zero
    tolerance, so the safe behaviour is the one you get by not thinking about it.
    """
    result = await session.run(
        _ANSWERS_FOR_QUESTION,
        question_id=question_id,
        exclude_superseded=exclude_superseded,
        exclude_confidential=exclude_confidential,
    )
    records = [record.data() async for record in result]
    return [AnswerRecord.model_validate(record) for record in records]


_COVERAGE_GAPS = """
MATCH (q:Question)-[:ASKED_IN]->(:RFP {id: $rfp_id})
OPTIONAL MATCH (q)-[:ANSWERED_BY]->(a:Answer)
WHERE coalesce(a.superseded, false) = false
OPTIONAL MATCH (q)-[:BELONGS_TO]->(cap:Capability)
OPTIONAL MATCH (:SME)-[:OWNS]->(cap)
WITH q,
     count(DISTINCT a)   AS answer_count,
     [x IN collect(DISTINCT cap.id) WHERE x IS NOT NULL] AS capability_ids,
     count(DISTINCT cap) AS capability_count
WHERE answer_count = 0
RETURN q.id   AS question_id,
       q.text AS text,
       capability_ids,
       CASE WHEN capability_count = 0
            THEN 'no answer and no capability owner — a human must own this'
            ELSE 'no answer, but the capability has an owner to route to'
       END AS reason
ORDER BY question_id ASC
"""


async def coverage_gaps(session: AsyncSession, *, rfp_id: str) -> list[CoverageGap]:
    """Questions in an RFP with no live answer.

    The reason distinguishes the two cases that matter: a topic somebody owns
    but has never written up, versus one nobody owns at all. The second is the
    list the business actually needs (build prompt §7).
    """
    result = await session.run(_COVERAGE_GAPS, rfp_id=rfp_id)
    records = [record.data() async for record in result]
    for record in records:
        record["capability_ids"] = sorted(record.get("capability_ids") or [])
    return [CoverageGap.model_validate(record) for record in records]


async def counts(session: AsyncSession) -> dict[str, Any]:
    """Node and relationship totals. Used to assert ingest is idempotent."""
    nodes = await session.run("MATCH (n) RETURN count(n) AS c")
    node_record = await nodes.single()
    rels = await session.run("MATCH ()-[r]->() RETURN count(r) AS c")
    rel_record = await rels.single()
    by_label = await session.run(
        "MATCH (n) UNWIND labels(n) AS label RETURN label, count(*) AS c ORDER BY label ASC"
    )
    label_counts = {record["label"]: record["c"] async for record in by_label}
    by_type = await session.run(
        "MATCH ()-[r]->() RETURN type(r) AS t, count(*) AS c ORDER BY t ASC"
    )
    type_counts = {record["t"]: record["c"] async for record in by_type}
    return {
        "nodes": node_record["c"] if node_record else 0,
        "relationships": rel_record["c"] if rel_record else 0,
        "by_label": label_counts,
        "by_relationship": type_counts,
    }
