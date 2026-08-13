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
    """One vector-index hit, before any graph multiplier is applied.

    `score` is a TRUE COSINE, converted from the index's own scale by
    :func:`index_score_to_cosine`. See that function for why the distinction is
    load-bearing rather than pedantic.
    """

    model_config = ConfigDict(extra="forbid")

    question_id: str
    text: str
    normalized_text: str
    #: Cosine similarity in [-1, 1]. Not the raw index score.
    score: float = Field(ge=-1.0, le=1.0)
    answer_ids: list[str] = Field(default_factory=list)


def index_score_to_cosine(score: float) -> float:
    """Convert Neo4j's cosine index score into an actual cosine.

    **Neo4j does not return the cosine.** For a `cosine` vector index it returns
    a similarity normalised into [0, 1]:

        score = (1 + cos) / 2        so        cos = 2 * score - 1

    Everything downstream — the calibration mapping, the derived floor, the
    relevance blend — is defined on the COSINE. Calibration measures its anchors
    with a plain dot product over unit vectors (`src.retrieval.calibration.
    cosine`), never through the index, so the two numbers live in different
    spaces and only one of them is what the floor was derived over.

    WHAT THIS COST, recorded because the shape recurs. Feeding the index score
    straight into `calibrated()` inflates every candidate: the background median
    0.4930 arrives as 0.7465 and the same-subject median 0.7718 arrives as
    0.8859, so scores that should sit near the middle of the band saturate at
    the top of it. Measured on the golden set, that made **all twenty** questions
    MATCHED — including the three the corpus deliberately cannot answer — with
    Recall@5 at 0.2667, because the floor no longer rejected anything and the
    ordering was decided by preference among uniformly-saturated relevances.

    It is the third instance of one fault: two numbers in different units
    compared as though they were the same. Amendment L was a calibrated-space
    floor compared against raw cosine; the task-prefix bug was two differently
    conditioned embedding spaces; this is an index score compared against a
    cosine-derived floor.

    WHY THE D19 PROBE GATE DID NOT CATCH IT. The tier-2 canary lands the golden
    questions on the derived floor using calibration's own dot product. It never
    reads the vector index, so calibration was entirely self-consistent while
    disagreeing with the graph. A guard that shares its inputs with the thing it
    guards can only prove internal consistency — which is why the retrieval eval
    runs against the real index and is the check that found this.
    """
    return 2.0 * score - 1.0


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

#: The index is asked for more than `k` because the visibility filter runs after
#: it. Without over-fetching, a run whose nearest neighbours happen to be another
#: customer's confidential material would silently come back short.
_VECTOR_OVERFETCH = 4

_FIND_SIMILAR = """
CALL db.index.vector.queryNodes($index_name, $fetch, $embedding)
YIELD node, score
// AMENDMENT P: paraphrases are calibration-only and are never candidates.
//
// Ingest already keeps them out of the vector index, so in a correctly built
// graph this clause matches nothing. It is here anyway, because the two
// mechanisms fail in different ways: the ingest rule protects against a
// paraphrase being INDEXED, this one against a paraphrase being RETURNED if one
// ever is — by a re-embed against an older graph, or a hand-run write. A
// paraphrase carries no answer of its own, so surfacing one would put a
// duplicate of an existing candidate into the list to compete for a rank.
WHERE NOT node:Paraphrase
WITH node, score
MATCH (node)-[:BELONGS_TO]->(:Domain {key: $domain})
// A question only qualifies if it has at least one LIVE answer this customer is
// allowed to see. Confidential material belongs to the customer whose RFP the
// question was asked in.
MATCH (node)-[:ANSWERED_BY]->(a:Answer)
WHERE coalesce(a.superseded, false) = false
  AND (
    coalesce(a.confidential, false) = false
    OR EXISTS {
      MATCH (node)-[:ASKED_IN]->(:RFP)-[:ISSUED_BY]->(owner:Customer)
      WHERE owner.name = $requesting_customer
    }
  )
WITH node, score, collect(DISTINCT a.id) AS answer_ids
RETURN node.id              AS question_id,
       node.text            AS text,
       node.normalized_text AS normalized_text,
       score                AS score,
       answer_ids           AS answer_ids
ORDER BY score DESC, question_id ASC
LIMIT $k
"""


async def find_similar_questions(
    session: AsyncSession,
    *,
    embedding: list[float],
    domain: str,
    requesting_customer: str,
    k: int,
) -> list[SimilarQuestion]:
    """Top-k in-domain questions this customer is permitted to see.

    Domain filtering, supersession and confidentiality all resolve inside the
    same query as the vector search. That is the point of a native vector index
    — and it means a caller cannot obtain a candidate they should not have, no
    matter what they do afterwards.

    `requesting_customer` is required, not optional with a permissive default:
    an optional visibility parameter is one forgotten argument away from a leak.
    """
    if k <= 0:
        return []
    if not requesting_customer.strip():
        raise ValueError("requesting_customer is required — visibility cannot be left implicit")
    expected = embedding_config().model.dimensions
    if len(embedding) != expected:
        raise ValueError(f"embedding has {len(embedding)} dimensions, index expects {expected}")

    # `index_score_to_cosine` inverts the normalisation Neo4j applies to a
    # COSINE index specifically. A euclidean index normalises differently, so
    # the same conversion would silently produce a number that is not a cosine
    # and the floor would judge it anyway — the exact failure this whole path
    # exists to have caught once.
    similarity = embedding_config().index.similarity
    if similarity != "cosine":
        raise ValueError(
            f"the vector index is configured for '{similarity}' similarity, but retrieval "
            f"converts scores assuming 'cosine'. Calibration measures cosine anchors, so a "
            f"different index similarity needs both a new conversion and a recalibration."
        )
    result = await session.run(
        _FIND_SIMILAR,
        index_name=embedding_config().index.name,
        fetch=k * _VECTOR_OVERFETCH,
        k=k,
        embedding=embedding,
        domain=domain,
        requesting_customer=requesting_customer,
    )
    records = [record.data() async for record in result]
    # Converted HERE, at the one boundary that knows the number came out of a
    # Neo4j cosine index. Every consumer downstream is entitled to assume a
    # cosine, because that is what calibration measured its anchors in.
    return [
        SimilarQuestion.model_validate({**record, "score": index_score_to_cosine(record["score"])})
        for record in records
    ]


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

    passes: list[tuple[str, dict[str, Any]]] = [
        (_ENTITY_BY_CODE, {"kind": kind, "needle": needle}),
        (_ENTITY_BY_EXACT_NAME, {"kind": kind, "normalised": normalised}),
        (_ENTITY_BY_PARTIAL_NAME, {"kind": kind, "normalised": normalised}),
    ]
    for query, params in passes:
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
OPTIONAL MATCH (q)-[:ASKED_IN]->(:RFP)-[:ISSUED_BY]->(cust:Customer)
WITH q, a, cust
WHERE ($exclude_superseded = false OR coalesce(a.superseded, false) = false)
  // Confidentiality is not optional. A confidential answer is visible only to
  // the customer it belongs to; there is no parameter that relaxes this.
  AND (
    coalesce(a.confidential, false) = false
    OR cust.name = $requesting_customer
  )
OPTIONAL MATCH (a)-[:RESULTED_IN]->(o:Outcome)
OPTIONAL MATCH (a)-[:AUTHORED_BY]->(sme:SME)
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
    requesting_customer: str,
    exclude_superseded: bool = True,
) -> list[AnswerRecord]:
    """Answers attached to a question that this customer may see, newest first.

    There is deliberately **no** `exclude_confidential` parameter. It was
    removed rather than defaulted, because a boolean a caller can pass `False`
    to is a leak waiting for one careless call site; confidentiality is now a
    property of the query, not a choice the caller makes.

    `exclude_superseded` remains a parameter because a caller sometimes
    legitimately wants the full chain (lineage, audit). It defaults to on, since
    a superseded answer reaching a draft is a staleness bug the harness scores
    at zero tolerance.
    """
    if not requesting_customer.strip():
        raise ValueError("requesting_customer is required — visibility cannot be left implicit")
    result = await session.run(
        _ANSWERS_FOR_QUESTION,
        question_id=question_id,
        requesting_customer=requesting_customer,
        exclude_superseded=exclude_superseded,
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
