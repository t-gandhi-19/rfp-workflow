"""Units agreement between calibration and the vector index (amendment S).

WHY THIS EXISTS. The D19 probe gate was blind to a units defect for its entire
life. Calibration computes similarity with its own dot product and never reads
the vector index, so it could be — and was — perfectly self-consistent while
disagreeing with production: Neo4j returns `(1 + cos) / 2` for a cosine index,
and the floor derived from true cosine was being applied to that. Every golden
question MATCHED, including the three the corpus cannot answer, and the guard
built to notice exactly that kind of drift saw nothing.

THE GENERAL RULE, which is the point of this module rather than the specific
bug: **when two subsystems must agree on a unit or a scale, an agreement test
exists between them. Self-consistency is not agreement.** A check that shares
its inputs with the thing it checks can only prove the thing is consistent with
itself.

So this compares the two paths directly, on the same vectors, and refuses if
they disagree:

    direct    sum(a[i] * b[i]) over the stored vectors, in float64 — what
              `src.retrieval.calibration.cosine` does and what the anchors and
              the derived floor are measured in.
    index     `2 * score - 1` applied to what `db.index.vector.queryNodes`
              returns — what retrieval actually judges.
"""

from __future__ import annotations

from dataclasses import dataclass

from neo4j import AsyncSession

from src.contracts.embedding import embedding_config
from src.graph.queries import index_score_to_cosine

#: How many probe questions to sample. Fixed and small: this runs inside
#: preflight, where cost is a feature, and a units mismatch is systemic — it
#: shows up on the first pair or not at all.
SAMPLE_SIZE = 3

#: Neighbours to compare per probe.
NEIGHBOURS = 5

#: Largest tolerated disagreement between the two paths.
#:
#: DERIVATION, and an honest account of where it stops being a derivation.
#:
#: float32 carries a 24-bit mantissa, so its unit roundoff is 2^-24 = 5.96e-8.
#: Neo4j stores vectors as float32 and scores in the index's own representation;
#: accumulated over a 768-dimension dot product the worst-case relative error is
#: about n*u = 768 * 5.96e-8 = 4.6e-5, and `2 * score - 1` doubles the absolute
#: error, giving roughly 9.2e-5.
#:
#: MEASURED disagreement exceeds that bound, and by a widening margin as more
#: pairs are sampled: 6.5e-4 on one real-corpus pair, 1.4e-3 on the stand-in's
#: near-orthogonal vectors, and 2.5e-3 as the worst of the 15 real-corpus pairs
#: this check actually compares. That is consistent with the index scoring on a
#: reduced-precision or quantised representation rather than on the stored
#: float32, and it means a bound derived from float32 alone would be far too
#: tight — an earlier 1e-3, fitted to a single observation, failed in CI on the
#: other geometry.
#:
#: So epsilon is set from what this check must DISCRIMINATE, with the float32
#: analysis as the floor rather than the answer. The two competing hypotheses —
#: "the index returns cos" and "the index returns (1 + cos) / 2" — are separated
#: by (1 - cos) / 2, which is 0.455 at a cosine of 0.09 and never below ~0.1 for
#: any pair this corpus produces outside exact duplicates. 1e-2 is therefore
#: about 4x the largest disagreement observed and at least 10x smaller than the
#: fault it exists to catch.
#:
#: If a future sample approaches 1e-2, that is a finding about the index's
#: precision and belongs in the register — not a number to raise.
EPSILON = 1e-2


class UnitsDisagreementError(RuntimeError):
    """The two similarity paths do not agree. Retrieval must not run."""


@dataclass(frozen=True)
class AgreementSample:
    """One comparison, kept so a failure can report observed values."""

    probe_id: str
    neighbour_id: str
    direct_cosine: float
    raw_index_score: float
    converted_cosine: float

    @property
    def delta(self) -> float:
        return abs(self.direct_cosine - self.converted_cosine)


async def sample_agreement(
    session: AsyncSession, *, sample_size: int = SAMPLE_SIZE, neighbours: int = NEIGHBOURS
) -> list[AgreementSample]:
    """Compare the two paths over a small, deterministic sample.

    Probes are the lowest question ids in the index, sorted — fixed rather than
    random, so a failure is reproducible and two runs compare the same things.
    """
    index_name = embedding_config().index.name

    result = await session.run(
        "MATCH (q:Question) WHERE q.embedding IS NOT NULL "
        "RETURN q.id AS id, q.embedding AS embedding ORDER BY q.id LIMIT $limit",
        limit=sample_size,
    )
    probes = [(record["id"], list(record["embedding"])) async for record in result]

    samples: list[AgreementSample] = []
    for probe_id, probe_vector in probes:
        hits = await session.run(
            "CALL db.index.vector.queryNodes($index, $k, $embedding) YIELD node, score "
            "RETURN node.id AS id, score AS score, node.embedding AS embedding "
            "ORDER BY score DESC, node.id ASC",
            index=index_name,
            k=neighbours,
            embedding=probe_vector,
        )
        for record in [r async for r in hits]:
            neighbour = list(record["embedding"])
            direct = sum(a * b for a, b in zip(probe_vector, neighbour, strict=True))
            samples.append(
                AgreementSample(
                    probe_id=probe_id,
                    neighbour_id=record["id"],
                    direct_cosine=direct,
                    raw_index_score=record["score"],
                    converted_cosine=index_score_to_cosine(record["score"]),
                )
            )
    return samples


def verify_agreement(samples: list[AgreementSample], *, epsilon: float = EPSILON) -> None:
    """Refuse if the paths disagree, naming both and showing the observed values.

    An empty sample is a failure, not a pass. A check that silently measures
    nothing is the shape of guard this amendment exists to stop shipping.
    """
    if not samples:
        raise UnitsDisagreementError(
            "the units-agreement check compared nothing: no question in the vector index "
            "returned a neighbour. Run: make ingest"
        )

    worst = max(samples, key=lambda sample: sample.delta)
    if worst.delta > epsilon:
        raise UnitsDisagreementError(
            "UNITS DISAGREEMENT between calibration and the vector index.\n"
            f"  probe             {worst.probe_id} vs {worst.neighbour_id}\n"
            f"  direct cosine     {worst.direct_cosine:.6f}   "
            "(dot product over the stored vectors — what the anchors and the floor "
            "are measured in)\n"
            f"  raw index score   {worst.raw_index_score:.6f}   "
            "(what db.index.vector.queryNodes returned)\n"
            f"  converted         {worst.converted_cosine:.6f}   (2 * score - 1)\n"
            f"  disagreement      {worst.delta:.6f}   exceeds epsilon {epsilon}\n"
            f"  compared          {len(samples)} pair(s)\n"
            "Causes in check order: 1. the index similarity function is no longer cosine, "
            "so the conversion in src/graph/queries.py no longer inverts it; 2. the stored "
            "vectors are not unit length, so a dot product is not a cosine; 3. Neo4j has "
            "changed how it normalises index scores. The floor is derived in the direct "
            "path and applied in the index path, so retrieval must not run until they agree."
        )


async def check_units_agreement(
    session: AsyncSession, *, sample_size: int = SAMPLE_SIZE, epsilon: float = EPSILON
) -> list[AgreementSample]:
    """Sample and verify. Raises :class:`UnitsDisagreementError` on failure."""
    samples = await sample_agreement(session, sample_size=sample_size)
    verify_agreement(samples, epsilon=epsilon)
    return samples
