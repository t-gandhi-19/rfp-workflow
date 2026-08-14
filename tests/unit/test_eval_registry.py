"""The category registry — the single place that knows what §3 specifies.

The properties here are all about one failure: the report, the Postgres writer
and the delta comparison silently disagreeing about which categories exist. Each
consumer derives its list from this module, so these assertions are what keep
that derivation honest.
"""

from __future__ import annotations

import re

import pytest

from src.evals.contracts import CategoryResult, CategoryStatus
from src.evals.registry import (
    IMPLEMENTED,
    ORDER,
    PLACEHOLDERS,
    Category,
    all_placeholders,
    in_report_order,
    placeholder,
)

SPECIFIED_BY_SECTION_3 = {
    "extraction",
    "retrieval",
    "grounding",
    "compliance",
    "quality",
    "adversarial",
}


class TestTheRegistryCoversTheSpec:
    def test_all_six_specified_categories_are_named(self) -> None:
        assert set(ORDER) == SPECIFIED_BY_SECTION_3

    def test_the_order_has_no_duplicates(self) -> None:
        assert len(ORDER) == len(set(ORDER))

    def test_every_unimplemented_category_has_a_placeholder(self) -> None:
        """A category with neither an implementation nor a placeholder would
        vanish from the report, which reads as 'nothing to say'."""
        assert set(PLACEHOLDERS) == set(ORDER) - IMPLEMENTED

    def test_no_category_is_both_implemented_and_a_placeholder(self) -> None:
        assert not (IMPLEMENTED & set(PLACEHOLDERS))

    def test_the_two_phase_3_categories_are_the_implemented_ones(self) -> None:
        assert {Category.EXTRACTION, Category.RETRIEVAL} == IMPLEMENTED


class TestPlaceholdersCarryNoNumbers:
    @pytest.mark.parametrize("key", sorted(PLACEHOLDERS))
    def test_a_placeholder_has_no_metrics(self, key: str) -> None:
        """A zero in a placeholder would be persisted, then differenced — turning
        'not measured' into a score, and later into a false regression."""
        assert placeholder(key).metrics == []

    @pytest.mark.parametrize("key", sorted(PLACEHOLDERS))
    def test_a_placeholder_has_no_violations(self, key: str) -> None:
        assert placeholder(key).violations == []

    @pytest.mark.parametrize("key", sorted(PLACEHOLDERS))
    def test_a_placeholder_is_marked_not_implemented(self, key: str) -> None:
        assert placeholder(key).status is CategoryStatus.NOT_IMPLEMENTED

    @pytest.mark.parametrize("key", sorted(PLACEHOLDERS))
    def test_a_placeholder_states_what_it_is_waiting_on(self, key: str) -> None:
        """'Not implemented' alone tells a reader nothing about when to expect it."""
        reason = placeholder(key).not_implemented_reason or ""
        assert len(reason) > 40, key
        assert "Phase" in reason or "Needs" in reason


class TestTheQualityPlaceholderRecordsD16:
    """D16 is a ruling with compensating controls, and the report is where a
    reader will look for them. A placeholder that said only 'needs the drafter'
    would drop the part of the decision that carries the trust load."""

    def test_it_names_the_judge_only_decision(self) -> None:
        reason = placeholder(Category.QUALITY).not_implemented_reason or ""
        assert "judge-model" in reason

    def test_it_names_the_different_model_family_control(self) -> None:
        reason = placeholder(Category.QUALITY).not_implemented_reason or ""
        assert "DIFFERENT MODEL FAMILY" in reason

    def test_it_records_that_human_agreement_is_deliberately_absent(self) -> None:
        """The removed metric is the part a reader is most likely to assume is
        merely pending. D16 removed it on purpose, and said why."""
        reason = placeholder(Category.QUALITY).not_implemented_reason or ""
        assert "NOT implemented" in reason
        assert "judge-vs-human" in reason

    def test_it_carries_the_accepted_risk(self) -> None:
        reason = placeholder(Category.QUALITY).not_implemented_reason or ""
        assert "uncalibrated" in reason


class TestReportOrdering:
    def test_results_are_sorted_into_registry_order(self) -> None:
        shuffled = [
            CategoryResult(key="adversarial", label="a", status=CategoryStatus.NOT_IMPLEMENTED),
            CategoryResult(key="extraction", label="e", status=CategoryStatus.PASS),
            CategoryResult(key="quality", label="q", status=CategoryStatus.NOT_IMPLEMENTED),
        ]
        assert [r.key for r in in_report_order(shuffled)] == [
            "extraction",
            "quality",
            "adversarial",
        ]

    def test_an_unregistered_category_is_refused(self) -> None:
        """It would otherwise render, persist and difference — all without
        anyone having decided it exists."""
        rogue = [CategoryResult(key="vibes", label="v", status=CategoryStatus.PASS)]
        with pytest.raises(ValueError, match="not in the registry"):
            in_report_order(rogue)

    def test_the_refusal_says_where_to_add_it(self) -> None:
        rogue = [CategoryResult(key="vibes", label="v", status=CategoryStatus.PASS)]
        with pytest.raises(ValueError, match=re.escape("registry.py")):
            in_report_order(rogue)

    def test_all_placeholders_returns_exactly_the_unimplemented_ones(self) -> None:
        assert {result.key for result in all_placeholders()} == set(ORDER) - IMPLEMENTED
