"""The category registry — what §3 specifies, and what is measured yet.

WHY UNIMPLEMENTED CATEGORIES ARE IN THE REPORT AT ALL. A category that is simply
absent reads as "nothing to say about grounding". A category named, greyed, and
carrying the reason it is not measured reads as "grounding is not measured yet,
and here is what it will take". Those are different claims and only the second is
true — the harness is built in Phase 3, deliberately BEFORE the agents in Phase
4, so the categories that judge agent output have nothing to judge yet.

That ordering is the design, not a shortfall: an eval harness written after the
thing it grades tends to grade what the thing happens to do.

The registry is also the single place that knows the full category list, so the
report, the Postgres writer and the delta comparison cannot disagree about which
categories exist.
"""

from __future__ import annotations

from src.evals.contracts import CategoryResult, CategoryStatus


class Category:
    """Names for the categories. Strings, so a typo in one consumer cannot
    silently create a category nobody renders."""

    EXTRACTION = "extraction"
    RETRIEVAL = "retrieval"
    GROUNDING = "grounding"
    COMPLIANCE = "compliance"
    QUALITY = "quality"
    ADVERSARIAL = "adversarial"
    #: Phase 4. NOT one of the six §3 quality categories — it grades the RUN
    #: rather than the answers, and it is here because cost per accepted answer
    #: is the number that decides whether this workflow is worth operating, and
    #: it had nowhere else to be reported. Named separately rather than folded
    #: into another category so nobody reads a cost overrun as a quality failure.
    OPERATIONAL = "operational"


#: Every category, in report order. Extraction and retrieval first because they
#: are the two that carry numbers; the rest follow in the order Phase 4 and
#: Phase 5 will fill them in.
ORDER: tuple[str, ...] = (
    Category.EXTRACTION,
    Category.RETRIEVAL,
    Category.GROUNDING,
    Category.COMPLIANCE,
    Category.QUALITY,
    Category.ADVERSARIAL,
    Category.OPERATIONAL,
)

#: Phase 4 implemented the five that needed drafted output. QUALITY remains a
#: placeholder in a run without judge-model — it is the only category that needs
#: a third provider call per answer, and D16 puts the judge in eval runs only.
IMPLEMENTED: frozenset[str] = frozenset(
    {
        Category.EXTRACTION,
        Category.RETRIEVAL,
        Category.GROUNDING,
        Category.COMPLIANCE,
        Category.ADVERSARIAL,
        Category.OPERATIONAL,
    }
)

#: Label and reason for each placeholder. The reason names the MISSING INPUT
#: rather than saying "not done": what a category needs is a fact about the
#: build order, and it tells a reader when to expect it.
PLACEHOLDERS: dict[str, tuple[str, str]] = {
    Category.QUALITY: (
        "Quality — LLM-judged, per D16",
        "Needs drafted prose and the judge-model. Per D16 this category is scored "
        "by judge-model alone, resolving to a DIFFERENT MODEL FAMILY than the "
        "drafter so a model never grades its own prose; the judge prompt and "
        "rubric are versioned files whose version is logged on every judge span, "
        "and every score attaches to the run trace with the judged text's span "
        "ids. The human spot-check sample and judge-vs-human agreement metric are "
        "deliberately NOT implemented — see D16, including its accepted risk that "
        "judge scores are uncalibrated against human opinion.",
    ),
}


def placeholder(key: str) -> CategoryResult:
    """A named, greyed, unmeasured category.

    Carries no metrics BY CONSTRUCTION. A placeholder with a zero in it would be
    written to Postgres and differenced against the next SHA, which would turn
    "not measured" into a score of zero and then into an apparent regression the
    moment it was implemented.
    """
    label, reason = PLACEHOLDERS[key]
    return CategoryResult(
        key=key,
        label=label,
        status=CategoryStatus.NOT_IMPLEMENTED,
        metrics=[],
        violations=[],
        not_implemented_reason=reason,
    )


def deselected(key: str) -> CategoryResult:
    """An IMPLEMENTED category the caller chose not to run.

    Rendered exactly like a placeholder — greyed, no metrics — but the reason
    says DESELECTED, because "this code does not exist yet" and "this run did
    not execute it" are different facts and a report that conflated them would
    misrepresent the build. CI deselects retrieval for precisely this reason.
    """
    return CategoryResult(
        key=key,
        label=f"{key.title()} — not run in this invocation",
        status=CategoryStatus.NOT_IMPLEMENTED,
        metrics=[],
        violations=[],
        not_implemented_reason=(
            f"DESELECTED — the {key} category is implemented but was excluded from this "
            "run via --categories. This is a fact about the invocation, not about the "
            "build. Run `make evals` with no selector to measure everything. "
            "Where CI deselects retrieval, no coverage is lost: the retrieval MACHINERY "
            "is proved by the integration suite against a real graph and by calibration "
            "running with both separation tiers enforcing, while retrieval's SEMANTIC "
            "numbers are real-model numbers of record by design — a score computed over "
            "stand-in vectors is not a weaker version of the real one, it is a number "
            "about a different thing."
        ),
    )


def all_placeholders() -> list[CategoryResult]:
    return [placeholder(key) for key in ORDER if key not in IMPLEMENTED]


def in_report_order(results: list[CategoryResult]) -> list[CategoryResult]:
    """Sort into `ORDER`, and refuse anything the registry does not name.

    The refusal matters: a category key that reaches the report without a
    registry entry would render, be written to Postgres, and be differenced —
    all while nobody had decided it exists.
    """
    known = {result.key for result in results}
    unknown = sorted(known - set(ORDER))
    if unknown:
        raise ValueError(
            f"categories not in the registry: {unknown}. Add them to "
            f"src/evals/registry.py:ORDER, with a placeholder entry if they are "
            f"not implemented, so every consumer agrees on the category list."
        )
    position = {key: index for index, key in enumerate(ORDER)}
    return sorted(results, key=lambda result: position[result.key])
