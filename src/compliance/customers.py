"""The customer names the cross-customer guardrail knows about.

WHY THIS READS THE FIXTURE REGISTRY. The confidentiality guarantee that matters
is enforced inside the Cypher: another customer's confidential material is
excluded from candidates before anything sees it (Phase 2 amendment A). This
list is the belt to that query's braces, and it catches a different failure —
a name the DRAFTER produced from its own weights rather than from a source it
was given, which no query filter can see.

That distinction is why a hardcoded list here would be wrong twice over: it
would drift from the corpus, and it would suggest the guardrail is the primary
control when it is the secondary one.

The registry is synthetic and every name in it is fictional (rule 1).
"""

from __future__ import annotations

import csv
from functools import lru_cache
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
CUSTOMERS_CSV = REPO_ROOT / "fixtures" / "registry" / "customers.csv"


@lru_cache(maxsize=1)
def known_customers() -> tuple[str, ...]:
    """Every customer name in the registry, in file order.

    Returns an empty tuple when the registry is absent rather than raising: the
    guardrail's other five rules are still worth running in a checkout without
    fixtures, and a missing file is a deployment fact rather than a reason to
    fail every question. Callers that need the list to be populated say so —
    `tests/unit/test_guardrail_suite.py` asserts it is non-empty here, so an
    empty list cannot silently become the normal case.
    """
    if not CUSTOMERS_CSV.is_file():
        return ()
    with CUSTOMERS_CSV.open(encoding="utf-8", newline="") as handle:
        return tuple(
            row["name"].strip() for row in csv.DictReader(handle) if row.get("name", "").strip()
        )
