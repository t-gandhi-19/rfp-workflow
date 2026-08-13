"""Load the synthetic fixtures into Neo4j. Run via `make ingest`.

Idempotent by construction: every write is a MERGE on a natural key, so running
it twice leaves identical node and relationship counts. That is asserted by a
test, because "re-running is safe" is the kind of claim that silently stops
being true the first time someone adds a CREATE.

Embeddings are computed here, at ingest time, through the gateway's
`embed-model` alias. Agents never embed (build prompt §10).

    python -m scripts.ingest              # full ingest
    python -m scripts.ingest --reembed    # recompute embeddings only
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import os
import sys
from pathlib import Path
from typing import Any

from neo4j import AsyncSession

from src.contracts.embedding import EmbedRole, embedding_config
from src.gateway.client import GatewayClient
from src.gateway.fake_embedder import fake_embeddings, fake_embeddings_enabled
from src.graph.driver import close_driver, get_driver, normalise_name
from src.graph.queries import counts

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURES = REPO_ROOT / "fixtures"
REGISTRY = FIXTURES / "registry"

BATCH = 16


def load_csv(name: str) -> list[dict[str, str]]:
    with (REGISTRY / name).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def load_pairs() -> list[dict[str, Any]]:
    with (FIXTURES / "qa_pairs.json").open(encoding="utf-8") as handle:
        data: list[dict[str, Any]] = json.load(handle)
    return sorted(data, key=lambda pair: pair["id"])


def load_paraphrases() -> list[dict[str, Any]]:
    with (FIXTURES / "question_paraphrases.json").open(encoding="utf-8") as handle:
        data: list[dict[str, Any]] = json.load(handle)
    return sorted(data, key=lambda row: row["id"])


# ---------------------------------------------------------------------------
# Cypher. Every statement is a MERGE on a natural key.
# ---------------------------------------------------------------------------

MERGE_DOMAIN = "MERGE (d:Domain {key: $key}) SET d.name = $name"

# One statement per entity kind. Verbose, but every label is a literal in a
# static string — no label is ever interpolated from data, which is what keeps
# "agents never write free Cypher" true of the ingest path too.
MERGE_BY_KIND: dict[str, str] = {
    "vendor": """
        UNWIND $rows AS row
        MERGE (n:Vendor {code: row.code})
        SET n:RegistryEntity, n.entity_kind = 'vendor', n.name = row.name,
            n.normalised_name = row.normalised_name, n.services = row.services,
            n.active = row.active
    """,
    "product": """
        UNWIND $rows AS row
        MERGE (n:Product {code: row.code})
        SET n:RegistryEntity, n.entity_kind = 'product', n.name = row.name,
            n.normalised_name = row.normalised_name, n.description = row.description,
            n.domain = row.domain
    """,
    "certification": """
        UNWIND $rows AS row
        MERGE (n:Certification {code: row.code})
        SET n:RegistryEntity, n.entity_kind = 'certification', n.name = row.name,
            n.normalised_name = row.normalised_name, n.scope = row.scope
    """,
    "client": """
        UNWIND $rows AS row
        MERGE (n:Customer {id: row.code})
        SET n:RegistryEntity, n.entity_kind = 'client', n.code = row.code,
            n.name = row.name, n.normalised_name = row.normalised_name,
            n.industry = row.industry
    """,
    "location": """
        UNWIND $rows AS row
        MERGE (n:Location {code: row.code})
        SET n:RegistryEntity, n.entity_kind = 'location', n.name = row.name,
            n.normalised_name = row.normalised_name, n.country = row.country,
            n.kind = row.kind
    """,
}

MERGE_CASE_STUDIES = """
UNWIND $rows AS row
MERGE (c:CaseStudy {code: row.code})
SET c.title = row.title, c.customer = row.customer, c.domain = row.domain,
    c.publicly_usable = row.publicly_usable, c.summary = row.summary
"""

MERGE_CAPABILITIES = """
UNWIND $rows AS row
MERGE (c:Capability {id: row.id}) SET c.name = row.name
"""

MERGE_SMES = """
UNWIND $rows AS row
MERGE (s:SME {id: row.id}) SET s.name = row.name, s.role = row.role
WITH s, row
UNWIND row.capabilities AS capability_id
MATCH (c:Capability {id: capability_id})
MERGE (s)-[:OWNS]->(c)
"""

MERGE_OUTCOMES = """
UNWIND $values AS value
MERGE (o:Outcome {value: value})
"""

MERGE_PAIRS = """
UNWIND $rows AS row
MERGE (cust:Customer {name: row.customer})
MERGE (rfp:RFP {id: row.rfp_id})
  SET rfp.domain = row.domain, rfp.historical = true
MERGE (rfp)-[:ISSUED_BY]->(cust)
MERGE (d:Domain {key: row.domain})
MERGE (q:Question {id: row.question_id})
  SET q.text = row.question, q.normalized_text = row.normalized_question,
      q.domain = row.domain, q.question_type = row.question_type
MERGE (q)-[:ASKED_IN]->(rfp)
MERGE (q)-[:BELONGS_TO]->(d)
WITH q, row
MATCH (cap:Capability {id: row.capability_id})
MERGE (q)-[:BELONGS_TO]->(cap)
WITH q, row
MERGE (a:Answer {id: row.answer_id})
  SET a.text = row.answer, a.answer_date = row.answer_date,
      a.confidential = row.confidential, a.superseded = row.superseded,
      a.word_count = row.word_count
MERGE (q)-[:ANSWERED_BY]->(a)
WITH a, row
MATCH (sme:SME {id: row.author_sme_id})
MERGE (a)-[:AUTHORED_BY]->(sme)
WITH a, row
MATCH (o:Outcome {value: row.outcome})
MERGE (a)-[:RESULTED_IN]->(o)
"""

# A paraphrase is an alternate PHRASING of an existing question, not a new Q&A
# pair. It gets a Question node and points at the answer the original already
# has, so the corpus still holds 40 answers and nothing new competes for a rank.
#
# It is ASKED_IN the SAME RFP as the question it rephrases, and that is load
# bearing rather than tidy: `find_similar_questions` decides confidentiality by
# walking (question)-[:ASKED_IN]->(:RFP)-[:ISSUED_BY]->(owner). A paraphrase of
# the confidential question hung off a different customer's RFP would make the
# confidential answer reachable through the paraphrase — a leak by way of a
# synonym.
MERGE_PARAPHRASES = """
UNWIND $rows AS row
MERGE (cust:Customer {name: row.customer})
MERGE (rfp:RFP {id: row.rfp_id})
MERGE (rfp)-[:ISSUED_BY]->(cust)
MERGE (d:Domain {key: row.domain})
MERGE (q:Question {id: row.question_id})
  // AMENDMENT P. The second label is what keeps a paraphrase out of the
  // retrieval candidate space. It carries no answer of its own — it points at
  // the one its original already has — so a paraphrase surfacing as a candidate
  // would be a duplicate of a result already in the list, competing with it for
  // a rank it cannot deserve.
  //
  // A LABEL rather than a property because it is what the node IS, and because
  // `QUESTIONS_TO_EMBED` can then exclude it without reading a field that a
  // future writer might forget to set. Belt and braces: the label keeps them out
  // of the vector index at all, AND `find_similar_questions` filters on it, so
  // neither an accidental re-embed nor a new query can put one in a result.
  SET q:Paraphrase,
      q.text = row.question, q.normalized_text = row.normalized_question,
      q.domain = row.domain, q.question_type = row.question_type,
      q.paraphrase_of = row.paraphrase_of
MERGE (q)-[:ASKED_IN]->(rfp)
MERGE (q)-[:BELONGS_TO]->(d)
WITH q, row
MATCH (cap:Capability {id: row.capability_id})
MERGE (q)-[:BELONGS_TO]->(cap)
WITH q, row
MATCH (a:Answer {id: row.answer_id})
MERGE (q)-[:ANSWERED_BY]->(a)
WITH q, row
MATCH (original:Question {id: row.paraphrase_of})
MERGE (q)-[:PARAPHRASE_OF]->(original)
"""

MERGE_EVIDENCE = """
UNWIND $rows AS row
MATCH (a:Answer {id: row.answer_id})
MATCH (c:CaseStudy {code: row.case_study_code})
MERGE (a)-[:EVIDENCED_BY]->(c)
"""

MERGE_SUPERSESSION = """
UNWIND $rows AS row
MATCH (old:Answer {id: row.old_id})
MATCH (new:Answer {id: row.new_id})
MERGE (new)-[:SUPERSEDES]->(old)
SET old.superseded = true
"""

SET_EMBEDDINGS = """
UNWIND $rows AS row
MATCH (q:Question {id: row.question_id})
SET q.embedding = row.embedding
"""

# Amendment P: paraphrases are calibration-only and never enter the vector
# index. Calibration embeds them itself, from the fixtures, because it needs
# them on the QUERY side — a paraphrase models an unseen phrasing arriving in a
# new RFP. It never needs them on the document side, which is exactly what being
# absent from the index means.
#
# This is also what the shipped calibration artifact already describes: its
# population is `query:any_family_member x document:indexed_originals`, and
# `calibration_corpus()` marks every paraphrase `indexed: False`. Until this
# clause existed the graph disagreed with that artifact — 148 questions were
# embedded where the statistics assumed 40.
QUESTIONS_TO_EMBED = """
MATCH (q:Question)
WHERE NOT q:Paraphrase
  AND ($force OR q.embedding IS NULL)
RETURN q.id AS question_id, q.normalized_text AS text
ORDER BY question_id ASC
"""

#: Amendment P is retroactive: a graph ingested before it exists holds paraphrase
#: vectors that are still in the index. Skipping them from now on would leave the
#: old ones in place — invisible, and exactly the candidates the amendment
#: forbids — so ingest clears them rather than assuming a clean volume.
CLEAR_PARAPHRASE_EMBEDDINGS = """
MATCH (q:Question:Paraphrase)
WHERE q.embedding IS NOT NULL
REMOVE q.embedding
RETURN count(q) AS cleared
"""


# ---------------------------------------------------------------------------
# Ingest
# ---------------------------------------------------------------------------


async def _embed(texts: list[str]) -> list[list[float]]:
    """Embed through the gateway, or with the CI stand-in when opted in."""
    dimensions = embedding_config().model.dimensions
    if fake_embeddings_enabled():
        return fake_embeddings(texts, dimensions, role=EmbedRole.DOCUMENT)

    client = GatewayClient.from_env()
    alias = embedding_config().model.alias
    vectors: list[list[float]] = []
    for start in range(0, len(texts), BATCH):
        vectors.extend(
            await client.embed(texts[start : start + BATCH], alias=alias, role=EmbedRole.DOCUMENT)
        )

    bad = [len(vector) for vector in vectors if len(vector) != dimensions]
    if bad:
        raise RuntimeError(
            f"gateway returned {sorted(set(bad))}-dim vectors; the index expects {dimensions}. "
            "Run: make preflight"
        )
    return vectors


async def ingest_registry(session: AsyncSession) -> None:
    await session.run(MERGE_DOMAIN, key="cloud_migration", name="Cloud Migration Services")

    def rows(
        items: list[dict[str, str]], code_field: str, extra: tuple[str, ...]
    ) -> list[dict[str, Any]]:
        return [
            {
                "code": item[code_field],
                "name": item["name"],
                "normalised_name": normalise_name(item["name"]),
                **{key: item[key] for key in extra},
            }
            for item in sorted(items, key=lambda i: i[code_field])
        ]

    await session.run(
        MERGE_BY_KIND["vendor"], rows=rows(load_csv("vendors.csv"), "code", ("services", "active"))
    )
    await session.run(
        MERGE_BY_KIND["product"],
        rows=rows(load_csv("products.csv"), "code", ("description", "domain")),
    )
    await session.run(
        MERGE_BY_KIND["certification"],
        rows=rows(load_csv("certifications.csv"), "code", ("scope",)),
    )
    await session.run(
        MERGE_BY_KIND["client"], rows=rows(load_csv("customers.csv"), "id", ("industry",))
    )
    await session.run(
        MERGE_BY_KIND["location"],
        rows=rows(load_csv("locations.csv"), "code", ("country", "kind")),
    )

    await session.run(
        MERGE_CASE_STUDIES, rows=sorted(load_csv("case_studies.csv"), key=lambda r: r["code"])
    )
    await session.run(
        MERGE_CAPABILITIES, rows=sorted(load_csv("capabilities.csv"), key=lambda r: r["id"])
    )
    await session.run(
        MERGE_SMES,
        rows=[
            {
                "id": sme["id"],
                "name": sme["name"],
                "role": sme["role"],
                "capabilities": sme["capabilities"].split("|"),
            }
            for sme in sorted(load_csv("smes.csv"), key=lambda r: r["id"])
        ],
    )
    await session.run(MERGE_OUTCOMES, values=["won", "lost", "unknown"])


async def ingest_pairs(session: AsyncSession, pairs: list[dict[str, Any]]) -> None:
    customers = {row["name"]: row["id"] for row in load_csv("customers.csv")}
    case_studies = sorted(load_csv("case_studies.csv"), key=lambda r: r["code"])
    superseded_ids = {pair["answer_id"] for pair in pairs if pair["superseded_by"]}

    rows = [
        {
            "rfp_id": f"HRFP-{customers[pair['customer']]}",
            "question_id": pair["question_id"],
            "answer_id": pair["answer_id"],
            "question": pair["question"],
            "normalized_question": " ".join(pair["question"].split()),
            "answer": pair["answer"],
            "domain": pair["domain"],
            "question_type": pair["question_type"],
            "capability_id": pair["capability_id"],
            "customer": pair["customer"],
            "answer_date": pair["answer_date"],
            "author_sme_id": pair["author_sme_id"],
            "outcome": pair["outcome"],
            "confidential": pair["confidential"],
            "superseded": pair["answer_id"] in superseded_ids,
            "word_count": pair["word_count"],
        }
        for pair in pairs
    ]
    await session.run(MERGE_PAIRS, rows=rows)

    # Evidence: attach each answer to a case study from the same customer where
    # one exists. Deterministic, so re-ingest produces the same edges.
    evidence = [
        {"answer_id": pair["answer_id"], "case_study_code": study["code"]}
        for pair in pairs
        for study in case_studies
        if study["customer"] == pair["customer"]
    ]
    if evidence:
        await session.run(MERGE_EVIDENCE, rows=sorted(evidence, key=lambda r: r["answer_id"]))

    chains = sorted(
        (
            {"old_id": pair["answer_id"], "new_id": pair["superseded_by"]}
            for pair in pairs
            if pair["superseded_by"]
        ),
        key=lambda r: r["old_id"],
    )
    if chains:
        await session.run(MERGE_SUPERSESSION, rows=chains)


async def ingest_paraphrases(session: AsyncSession, paraphrases: list[dict[str, Any]]) -> None:
    """Alternate phrasings, attached to the answer their original already has."""
    customers = {row["name"]: row["id"] for row in load_csv("customers.csv")}
    rows = [
        {
            "rfp_id": f"HRFP-{customers[row['customer']]}",
            "question_id": row["question_id"],
            "question": row["question"],
            "normalized_question": row["normalized_question"],
            "domain": row["domain"],
            "question_type": row["question_type"],
            "capability_id": row["capability_id"],
            "customer": row["customer"],
            "answer_id": row["answer_id"],
            "paraphrase_of": row["paraphrase_of"],
        }
        for row in paraphrases
    ]
    if rows:
        await session.run(MERGE_PARAPHRASES, rows=rows)


async def clear_paraphrase_embeddings(session: AsyncSession) -> int:
    """Remove any paraphrase vectors left by a pre-amendment-P ingest."""
    result = await session.run(CLEAR_PARAPHRASE_EMBEDDINGS)
    record = await result.single()
    return int(record["cleared"]) if record else 0


async def embed_questions(session: AsyncSession, *, force: bool) -> int:
    result = await session.run(QUESTIONS_TO_EMBED, force=force)
    pending = [record.data() async for record in result]
    if not pending:
        return 0

    vectors = await _embed([item["text"] for item in pending])
    rows = [
        {"question_id": item["question_id"], "embedding": vector}
        for item, vector in zip(pending, vectors, strict=True)
    ]
    await session.run(SET_EMBEDDINGS, rows=rows)
    return len(rows)


async def run(*, reembed_only: bool) -> int:
    pairs = load_pairs()
    paraphrases = load_paraphrases()
    driver = get_driver()
    try:
        async with driver.session() as session:
            if not reembed_only:
                await ingest_registry(session)
                await ingest_pairs(session, pairs)
                await ingest_paraphrases(session, paraphrases)
            cleared = await clear_paraphrase_embeddings(session)
            embedded = await embed_questions(session, force=reembed_only)
            totals = await counts(session)
    finally:
        await close_driver()

    # Diagnostics go to stderr so stdout stays a clean JSON document. CI parses
    # this output to assert idempotency, and a warning printed to stdout made
    # that parse fail — the warning is important, but not at the cost of making
    # the summary unreadable to anything but a human.
    if fake_embeddings_enabled():
        sys.stderr.write(
            "WARNING: RFP_FAKE_EMBEDDINGS=1 — vectors are deterministic stand-ins, "
            "not semantic embeddings. Retrieval quality means nothing in this state.\n"
        )
    sys.stdout.write(
        json.dumps(
            {
                "mode": "reembed" if reembed_only else "ingest",
                "pairs": len(pairs),
                "paraphrases": len(paraphrases),
                # Amendment P. Nonzero exactly once, on the first ingest after
                # the amendment lands against a graph that predates it.
                "paraphrase_embeddings_cleared": cleared,
                "questions_embedded": embedded,
                "nodes": totals["nodes"],
                "relationships": totals["relationships"],
                "by_label": totals["by_label"],
                "by_relationship": totals["by_relationship"],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Ingest synthetic fixtures into Neo4j")
    parser.add_argument(
        "--reembed",
        action="store_true",
        help="recompute every embedding without re-ingesting nodes",
    )
    args = parser.parse_args()
    if "NEO4J_PASSWORD" not in os.environ:
        sys.stderr.write("NEO4J_PASSWORD is not set. Load .env first.\n")
        return 2
    return asyncio.run(run(reembed_only=args.reembed))


if __name__ == "__main__":
    raise SystemExit(main())
