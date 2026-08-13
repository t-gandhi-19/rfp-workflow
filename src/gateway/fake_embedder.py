"""A deterministic stand-in embedder, for CI only.

CI has no Ollama (CLAUDE.md rule 26), but ingest, the vector index and — since
amendment K — calibration all need vectors. This produces them without a model.

**It is a bag-of-words scheme, not a hash of the whole string.** Each token is
hashed to its own seeded unit vector; a text's embedding is the normalised sum
of its tokens'. That gives the one property the previous version could not have:
texts that share vocabulary land near each other, and texts that do not land
near orthogonal. Two paraphrases of the same question score high, two questions
about different subjects score low, and the resulting distribution has a real
shape rather than a uniform one.

That matters because the old scheme — one hash per whole text — made every pair
of distinct strings equidistant. Calibration measured over it would have found no
separation whatever between "the same question" and "an unrelated question".

**It is still not enough to calibrate the shipped corpus, and that is a finding
rather than a defect.** Measured over the 112 corpus questions, this scheme puts
the same-subject p05 at 0.2254 and the background p99 at 0.2841 — a separation of
-0.0587, so `make calibrate` refuses. Three rounds of improvement (stopwords,
stemming, paraphrases that retain domain vocabulary) moved that from -0.4217 to
-0.0587 without closing it.

The reason is structural. Every question in this corpus is about cloud migration,
so cross-subject pairs legitimately share vocabulary — "migration", "model",
"data", "programme". Distinguishing "the same question, reworded" from "a
different question about a neighbouring topic" on shared vocabulary alone is
exactly the judgement that requires semantics, which is what the real model is
for. A lexical stand-in comparing the 12th-worst match against the 122nd-best
non-match cannot make it. Loosening the guard to hide this would be measuring CI
rather than the corpus.

What this does NOT claim: anything about retrieval quality. The geometry is real,
the semantics are not — "downtime" and "outage" are as unrelated here as any two
words, where a real model knows better. Retrieval quality is measured against
real embeddings, and any eval depending on semantic similarity must not run
against this.

Enabling it takes an explicit opt-in (`RFP_FAKE_EMBEDDINGS=1`). It is off by
default, `make ingest` never sets it, and a test asserts both.
"""

from __future__ import annotations

import hashlib
import math
import os
import re

from src.contracts.embedding import EmbedRole, strip_task_prefix

#: The opt-in. Deliberately verbose and unlikely to be set by accident.
FAKE_EMBEDDINGS_ENV = "RFP_FAKE_EMBEDDINGS"

#: Words, numbers and internal apostrophes. Punctuation is a separator.
_TOKEN = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")

#: Mixed into every token hash, so these vectors cannot be confused with any
#: other hash-derived vector in the codebase.
_SALT = "rfp-fake-embedder-v2"

#: Function words and RFP boilerplate verbs, dropped before hashing.
#:
#: In a bag-of-words scheme these are pure noise, and on short questions they
#: are the MAJORITY of tokens: "How do you ...", "Describe your ...", "What is
#: your ..." open most of the corpus. Left in, every question resembles every
#: other question because they all ask politely, and the subject nouns that
#: actually distinguish them get outvoted. Measured on this corpus that put the
#: background p99 above the same-subject p05 — the separation guard fired on
#: what was really a stopword artifact.
#:
#: Deliberately small and specific to this domain's phrasing rather than a
#: general English stopword list: the point is to remove the boilerplate this
#: corpus actually repeats, not to do linguistics.
_STOPWORDS = frozenset(
    [
        "a",
        "an",
        "and",
        "any",
        "are",
        "as",
        "at",
        "be",
        "been",
        "between",
        "by",
        "can",
        "describe",
        "do",
        "does",
        "during",
        "each",
        "explain",
        "for",
        "from",
        "give",
        "have",
        "how",
        "in",
        "include",
        "including",
        "into",
        "is",
        "it",
        "its",
        "list",
        "of",
        "on",
        "or",
        "our",
        "out",
        "provide",
        "set",
        "that",
        "the",
        "their",
        "them",
        "there",
        "these",
        "this",
        "to",
        "us",
        "we",
        "what",
        "when",
        "where",
        "which",
        "who",
        "will",
        "with",
        "would",
        "you",
        "your",
    ]
)


def fake_embeddings_enabled(env: dict[str, str] | None = None) -> bool:
    """Whether the stand-in embedder is switched on. Default: no."""
    source = os.environ if env is None else env
    return source.get(FAKE_EMBEDDINGS_ENV, "").strip() == "1"


#: Suffixes stripped to fold inflected forms together, longest first.
#:
#: Without this the scheme measures SPELLING rather than vocabulary:
#: "vulnerability" and "vulnerabilities" hash to unrelated vectors, as do
#: "subcontractor"/"subcontractors" and "patch"/"patched". On this corpus those
#: three pairs were the lowest-scoring same-subject pairs in the whole set, and
#: they alone dragged the p05 below the background p99 — a separation failure
#: that was entirely an artifact of not stemming.
#:
#: Crude on purpose. This is not a linguistic stemmer and does not need to be;
#: it needs to be deterministic, specified in ten lines, and to stop the
#: measurement being dominated by plurals.
_SUFFIXES: tuple[tuple[str, str], ...] = (("ies", "y"), ("ing", ""), ("ed", ""), ("s", ""))

#: Below this, stripping does more harm than good ("is" -> "i").
_MIN_STEM = 4


def stem(token: str) -> str:
    """Fold common inflections onto a shared form. Deliberately approximate.

    Two stages, and the second is not optional. Stripping the plural alone
    leaves "timescales" as "timescale" but "processes" as "processe" — so the
    trailing "e" is folded too, which lands both a word and its plural on the
    same stem whichever shape they take. Stripping "es" instead would have made
    "timescales" and "timescale" disagree, which is the bug this replaced.
    """
    for suffix, replacement in _SUFFIXES:
        if suffix == "s" and token.endswith("ss"):
            continue  # "process", not "proces"
        if token.endswith(suffix):
            candidate = token[: -len(suffix)] + replacement
            if len(candidate) >= _MIN_STEM:
                token = candidate
            break
    if token.endswith("e") and len(token) - 1 >= _MIN_STEM:
        token = token[:-1]
    return token


def tokenize(text: str) -> list[str]:
    """Lowercase alphanumeric tokens, minus stopwords, stemmed.

    Deliberately crude and fully specified. If every token is a stopword the
    stripped list is returned instead, so a question made entirely of
    boilerplate still gets a vector rather than an empty bag.
    """
    # Stopwords are filtered BEFORE stemming, because the list holds surface
    # forms: stemming first turns "describe" into "describ", which matches
    # nothing in it and survives as content.
    tokens = _TOKEN.findall(text.lower())
    content = [token for token in tokens if token not in _STOPWORDS]
    return [stem(token) for token in (content or tokens)]


def _token_vector(token: str, dimensions: int) -> list[float]:
    """A stable pseudo-random unit vector for one token.

    Hashed with a counter until enough bytes exist, so the result depends only
    on the token and the width — not on Python's hash seed, the platform, or the
    order things were embedded in.
    """
    needed = dimensions * 2
    material = bytearray()
    counter = 0
    while len(material) < needed:
        material.extend(hashlib.sha256(f"{_SALT}:{counter}:{token}".encode()).digest())
        counter += 1

    values = [
        (int.from_bytes(material[index * 2 : index * 2 + 2], "big") / 32768.0) - 1.0
        for index in range(dimensions)
    ]
    norm = math.sqrt(sum(value * value for value in values))
    if norm == 0:  # pragma: no cover - astronomically unlikely
        return [1.0] + [0.0] * (dimensions - 1)
    return [value / norm for value in values]


def fake_embedding(text: str, dimensions: int, *, role: EmbedRole) -> list[float]:
    """A stable unit vector for `text`, built from its tokens.

    `role` is required and unused, on purpose. The real embedder's `role` has no
    default because embedding a query as a document produces a plausible vector
    that quietly retrieves badly; a stand-in whose signature was laxer would let
    a call site omit it in CI and only fail against the real model. So the shape
    of the two is kept identical.

    The task prefix is STRIPPED rather than embedded. If this mirrored the real
    model's role split, a CI query would never match its own corpus entry and
    the vector-index tests would assert nothing. Stripping makes the same text
    embed identically from either side, which keeps CI a clean plumbing oracle
    and keeps the calibration geometry symmetric — cross-prefix scores in CI
    measure the token overlap and nothing else.
    """
    if dimensions <= 0:
        raise ValueError("dimensions must be positive")

    tokens = tokenize(strip_task_prefix(text))
    if not tokens:
        # An empty text still needs a valid unit vector; the index rejects a zero
        # one and a raise here would make ingest fail on a blank field.
        return [1.0] + [0.0] * (dimensions - 1)

    summed = [0.0] * dimensions
    for token in tokens:
        vector = _token_vector(token, dimensions)
        for index in range(dimensions):
            summed[index] += vector[index]

    norm = math.sqrt(sum(value * value for value in summed))
    if norm == 0:  # pragma: no cover - requires exactly cancelling tokens
        return [1.0] + [0.0] * (dimensions - 1)
    return [value / norm for value in summed]


def fake_embeddings(texts: list[str], dimensions: int, *, role: EmbedRole) -> list[list[float]]:
    return [fake_embedding(text, dimensions, role=role) for text in texts]
