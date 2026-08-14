"""A deterministic stand-in embedder, for CI only.

CI has no Ollama (CLAUDE.md rule 26), but ingest, the vector index and — since
amendment K — calibration all need vectors. This produces them without a model.

THE EPISTEMIC SPLIT. State it before anything else, because everything below
depends on it:

    CI proves the calibration and retrieval MACHINERY end to end.
    The real-model run quoted in every PR proves the SEMANTICS.

Neither substitutes for the other. A green CI says the guards compute, gate and
refuse correctly on vectors whose geometry is known by construction. It says
nothing whatever about whether retrieval finds the right answer, because these
vectors carry no meaning. That question is settled only by the zero-tolerance
retrieval evals on real embeddings, and no CI result may ever be quoted in their
place.

**The scheme (v3, amendment N).** Each vector blends two parts:

    embedding(text) = normalize(a * family_anchor(family(text))
                                + (1 - a) * token_bag(text))

* **token_bag** — the v2 scheme, unchanged. Each token is hashed to its own
  seeded unit vector and the text's is the normalised sum, so texts sharing
  vocabulary land near each other.
* **family_anchor** — a seeded unit vector per `topic_family`, looked up from a
  committed mapping generated with the fixtures. Texts about the same subject
  share it; texts about different subjects get near-orthogonal ones.
* **a** — the anchor's weight, carried in the same generated artifact.

A text the lookup cannot place gets a PURE TOKEN BAG. That is not a fallback, it
is the mechanism: the three unanswerable golden questions and the two baits are
deliberately absent from the mapping, so they sit far from every anchor and land
below the floor. NO_MATCH works in CI by construction rather than by assertion.

**Why v2 was not enough, and why this is not cheating.** v2 could not calibrate
the shipped corpus at all: same-subject p05 0.2254 against background p99 0.2841,
a separation of -0.0587, so `make calibrate` refused and the D18/D19 gates could
only be skipped in CI — the guards went untested precisely where testing is
cheapest. Three rounds of lexical improvement (stopwords, stemming, paraphrases
retaining domain vocabulary) moved that from -0.4217 to -0.0587 without closing
it, and the reason is structural: every question here is about cloud migration,
so cross-subject pairs legitimately share vocabulary. Telling "the same question,
reworded" from "a different question about a neighbouring topic" on shared
vocabulary alone is exactly the judgement that needs semantics.

So v3 stops trying to DERIVE the subject and is TOLD it. That is honest for what
this is for — exercising machinery — and it is why the split above is stated
first. The anchor is not a model that learned the corpus; it is a label the
fixtures already contain, injected so the geometry has the shape a real embedder
would produce. Making CI's numbers good is not evidence that retrieval is good,
and the guard values CI commissions are its own and never the shipped ones.

What this still does NOT claim: anything about retrieval quality. Within a
family the geometry remains purely lexical — "downtime" and "outage" are as
unrelated here as any two words. Any eval depending on semantic similarity must
not run against this.

Enabling it takes an explicit opt-in (`RFP_FAKE_EMBEDDINGS=1`). It is off by
default, `make ingest` never sets it, and a test asserts both.
"""

from __future__ import annotations

import functools
import hashlib
import json
import math
import os
import re
from pathlib import Path
from typing import Any

from src.contracts.embedding import EmbedRole, strip_task_prefix

#: The opt-in. Deliberately verbose and unlikely to be set by accident.
FAKE_EMBEDDINGS_ENV = "RFP_FAKE_EMBEDDINGS"

#: Words, numbers and internal apostrophes. Punctuation is a separator.
_TOKEN = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")

#: Mixed into every token hash, so these vectors cannot be confused with any
#: other hash-derived vector in the codebase.
_SALT = "rfp-fake-embedder-v2"

#: A DIFFERENT salt for family anchors, so an anchor can never collide with the
#: token vector of a word that happens to be spelled like a family name.
_ANCHOR_SALT = "rfp-fake-embedder-v3-anchor"

#: Generated with the fixtures by `scripts/generate_fixtures.py`.
FAMILY_LOOKUP_PATH = (
    Path(__file__).resolve().parents[2] / "fixtures" / "fake_embedder_families.json"
)

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


def _seeded_unit_vector(salt: str, name: str, dimensions: int) -> list[float]:
    """A stable pseudo-random unit vector for one salted name.

    Hashed with a counter until enough bytes exist, so the result depends only
    on the salt, the name and the width — not on Python's hash seed, the
    platform, or the order things were embedded in.

    The seed layout is `salt:counter:name`, which is v2's exactly. Amendment N
    added the family anchor on top of the token bag and did NOT change the bag,
    so every token vector this produces is byte-identical to the one v2
    produced; only the salt distinguishes an anchor from a token.
    """
    needed = dimensions * 2
    material = bytearray()
    counter = 0
    while len(material) < needed:
        material.extend(hashlib.sha256(f"{salt}:{counter}:{name}".encode()).digest())
        counter += 1

    values = [
        (int.from_bytes(material[index * 2 : index * 2 + 2], "big") / 32768.0) - 1.0
        for index in range(dimensions)
    ]
    norm = math.sqrt(sum(value * value for value in values))
    if norm == 0:  # pragma: no cover - astronomically unlikely
        return [1.0] + [0.0] * (dimensions - 1)
    return [value / norm for value in values]


def _token_vector(token: str, dimensions: int) -> list[float]:
    """A stable pseudo-random unit vector for one token."""
    return _seeded_unit_vector(_SALT, token, dimensions)


@functools.lru_cache(maxsize=1)
def _lookup() -> dict[str, Any]:
    """The committed family mapping, read once.

    Loaded lazily rather than at import, so merely importing this module in a
    production process touches no fixture file.
    """
    if not FAMILY_LOOKUP_PATH.is_file():
        raise FileNotFoundError(
            f"the stand-in embedder's family lookup is missing at {FAMILY_LOOKUP_PATH}. "
            "It is generated with the fixtures. Run: make fixtures"
        )
    with FAMILY_LOOKUP_PATH.open(encoding="utf-8") as handle:
        data: dict[str, Any] = json.load(handle)
    return data


def family_anchor_weight() -> float:
    """The blend weight, from the generated artifact."""
    weight = float(_lookup()["config"]["family_anchor_weight"])
    if not 0.0 <= weight <= 1.0:
        raise ValueError(f"family_anchor_weight must be in [0, 1], got {weight}")
    return weight


def family_of(text: str) -> str | None:
    """The topic family of `text`, or None if the corpus does not place it.

    None is MEANINGFUL, not an error. The unanswerable golden questions and the
    two baits are deliberately absent, so they embed as a pure token bag and sit
    far from every anchor.

    Whitespace is collapsed before lookup, matching how the mapping is keyed —
    the same text re-wrapped across lines must not become a different question.
    """
    families: dict[str, str] = _lookup()["families"]
    return families.get(" ".join(text.split()))


def _anchor_vector(family: str, dimensions: int) -> list[float]:
    """A stable pseudo-random unit vector for one topic family.

    Same construction as a token vector under a different salt: two families are
    near-orthogonal in high dimensions, which is exactly the relation two
    unrelated subjects should have.
    """
    return _seeded_unit_vector(_ANCHOR_SALT, family, dimensions)


def _normalized(values: list[float], dimensions: int) -> list[float]:
    """Scale to unit length, with a valid fallback for a zero vector.

    The index rejects a zero vector, and raising here would make ingest fail on
    a blank field rather than store a harmless one.
    """
    norm = math.sqrt(sum(value * value for value in values))
    if norm == 0:  # pragma: no cover - requires exactly cancelling components
        return [1.0] + [0.0] * (dimensions - 1)
    return [value / norm for value in values]


def token_bag(text: str, dimensions: int) -> list[float]:
    """The v2 scheme, unchanged: the normalised sum of the tokens' vectors."""
    tokens = tokenize(text)
    if not tokens:
        return [1.0] + [0.0] * (dimensions - 1)

    summed = [0.0] * dimensions
    for token in tokens:
        vector = _token_vector(token, dimensions)
        for index in range(dimensions):
            summed[index] += vector[index]
    return _normalized(summed, dimensions)


def fake_embedding(
    text: str, dimensions: int, *, role: EmbedRole, alpha: float | None = None
) -> list[float]:
    """A stable unit vector for `text`: its family anchor blended with its tokens.

    `role` is required and unused, on purpose. The real embedder's `role` has no
    default because embedding a query as a document produces a plausible vector
    that quietly retrieves badly; a stand-in whose signature was laxer would let
    a call site omit it in CI and only fail against the real model. So the shape
    of the two is kept identical.

    The task prefix is STRIPPED rather than embedded. If this mirrored the real
    model's role split, a CI query would never match its own corpus entry and
    the vector-index tests would assert nothing. Stripping makes the same text
    embed identically from either side, which keeps CI a clean plumbing oracle
    and keeps the calibration geometry symmetric.

    `alpha` overrides the artifact's weight. It exists so tests can pin the two
    ends of the blend — 0.0 is exactly v2, 1.0 is the anchor alone — and no
    caller in the pipeline passes it.

    A text with no family is a pure token bag. That is the mechanism the
    unanswerable probes rely on, not a degraded mode.
    """
    if dimensions <= 0:
        raise ValueError("dimensions must be positive")

    stripped = strip_task_prefix(text)
    bag = token_bag(stripped, dimensions)

    family = family_of(stripped)
    if family is None:
        return bag

    weight = family_anchor_weight() if alpha is None else alpha
    if not 0.0 <= weight <= 1.0:
        raise ValueError(f"alpha must be in [0, 1], got {weight}")

    anchor = _anchor_vector(family, dimensions)
    blended = [weight * anchor[i] + (1.0 - weight) * bag[i] for i in range(dimensions)]
    return _normalized(blended, dimensions)


def fake_embeddings(
    texts: list[str], dimensions: int, *, role: EmbedRole, alpha: float | None = None
) -> list[list[float]]:
    """Order-independent: each text is embedded from itself alone."""
    return [fake_embedding(text, dimensions, role=role, alpha=alpha) for text in texts]
