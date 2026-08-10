"""Stage C — parsing the batched rerank response (build prompt §10 C).

A local model asked for structured JSON does not reliably produce it. This was
not a hypothetical: the very first real call made against `llama3.1:8b` on this
host returned **seven scores for eight candidates**, silently omitting index 1.
That response is committed verbatim at
`fixtures/recorded/rerank/malformed_missing_index.json` and is fed through this
parser by a regression test.

The rule when a response is unusable is **treat that question as
rerank-disabled**, never partially applied. Scoring the seven candidates the
model did return and defaulting the eighth to zero would be much worse than not
reranking at all: the missing candidate could be the right answer, and zero is
an active claim of irrelevance rather than an absence of information.

Nothing here raises. A rerank failure degrades one question's scoring; it never
crashes a run (build prompt §18).
"""

from __future__ import annotations

import json
from dataclasses import dataclass


@dataclass(frozen=True)
class RerankOutcome:
    """Either usable scores for every candidate, or a reason there are none."""

    #: index -> relevance, covering exactly 0..candidate_count-1. None when the
    #: response could not be used at all.
    scores: dict[int, float] | None
    #: Why the response was rejected. Logged and surfaced in the eval report so
    #: a degraded question is explainable rather than mysterious.
    reason: str | None = None

    @property
    def usable(self) -> bool:
        return self.scores is not None


def _reject(reason: str) -> RerankOutcome:
    return RerankOutcome(scores=None, reason=reason)


def parse_rerank_response(content: str, *, candidate_count: int) -> RerankOutcome:
    """Parse a rerank reply, rejecting anything not fully usable.

    Deliberately strict. Every rejection below is a case where a lenient parser
    would produce plausible-looking scores that quietly misrank candidates.
    """
    if candidate_count <= 0:
        return _reject("no candidates to rerank")

    text = content.strip()
    if not text:
        return _reject("empty response")

    # Models often wrap JSON in prose or a code fence; take the outermost object
    # rather than failing on decoration alone.
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        return _reject("no JSON object in response")

    try:
        payload = json.loads(text[start : end + 1])
    except json.JSONDecodeError as exc:
        return _reject(f"response is not valid JSON: {exc.msg}")

    if not isinstance(payload, dict):
        return _reject("response JSON is not an object")

    raw_scores = payload.get("scores")
    if not isinstance(raw_scores, list):
        return _reject("response has no `scores` array")

    scores: dict[int, float] = {}
    for entry in raw_scores:
        if not isinstance(entry, dict):
            return _reject("a score entry is not an object")
        index, score = entry.get("index"), entry.get("score")
        if not isinstance(index, int) or isinstance(index, bool):
            return _reject("a score entry has a non-integer index")
        if not isinstance(score, int | float) or isinstance(score, bool):
            return _reject(f"index {index} has a non-numeric score")
        if index in scores:
            return _reject(f"index {index} scored more than once")
        if not 0.0 <= float(score) <= 1.0:
            return _reject(f"index {index} scored {score}, outside [0, 1]")
        scores[index] = float(score)

    expected = set(range(candidate_count))
    actual = set(scores)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        parts = []
        if missing:
            parts.append(f"missing {missing}")
        if extra:
            parts.append(f"unexpected {extra}")
        return _reject(f"expected scores for all {candidate_count} candidates; " + ", ".join(parts))

    return RerankOutcome(scores=scores)
