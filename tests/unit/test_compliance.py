"""The deterministic compliance checker and the policy behind it."""

from __future__ import annotations

import pytest

from src.compliance import (
    DeterministicComplianceChecker,
    count_words,
    domain_policy,
    scan_forbidden,
    summarise,
)
from src.contracts import DraftedAnswer, ExtractedQuestion, QuestionType


def question(
    *, mandatory: bool = False, word_limit: int | None = None, qid: str = "GQ-001"
) -> ExtractedQuestion:
    return ExtractedQuestion(
        id=qid,
        rfp_id="rfp-golden",
        text="Describe your approach.",
        normalized_text="describe your approach",
        section="Technical Approach",
        question_type=QuestionType.TECHNICAL,
        mandatory=mandatory,
        word_limit=word_limit,
        order=0,
        printed_number="3.1",
    )


def answer(text: str, *, qid: str = "GQ-001") -> DraftedAnswer:
    return DraftedAnswer(
        question_id=qid,
        answer_text=text,
        source_ids=["ANS-0001"],
        confidence=0.85,
        needs_sme_review=False,
    )


class TestThePolicyLoads:
    def test_the_shipped_domain_parses(self) -> None:
        assert domain_policy().key == "cloud_migration"

    def test_pricing_and_legal_are_separate_rules(self) -> None:
        """Same action, different meaning: pricing is never drafted; legal is
        drafted and always escalated."""
        forbidden = domain_policy().compliance.forbidden_content
        assert forbidden.pricing.patterns
        assert forbidden.legal.terms
        assert "warranty" in [term.lower() for term in forbidden.legal.terms]

    def test_the_filename_and_the_declared_key_must_agree(self) -> None:
        """Or triage compares a detected domain against the wrong one."""
        assert domain_policy("cloud_migration").key == "cloud_migration"


class TestWordCounting:
    @pytest.mark.parametrize(
        ("text", "expected"),
        [("", 0), ("one", 1), ("one two three", 3), ("  spaced   out  ", 2)],
    )
    def test_words_are_whitespace_separated(self, text: str, expected: int) -> None:
        """What a reviewer counting by eye approximates, and what Word reports.
        Tokens or characters would be defensible and would not be the number the
        customer checks against."""
        assert count_words(text) == expected

    def test_a_hyphenated_term_is_one_word(self) -> None:
        assert count_words("multi-region active-active") == 2


class TestForbiddenContentScanning:
    @pytest.mark.parametrize(
        "text",
        [
            "The engagement is priced at USD 250 per workload.",
            "Approximately $1,200 for the assessment.",
            "We charge 4500 EUR for the landing zone.",
            "Rates are £95 per hour for 40 hours.",
        ],
    )
    def test_currency_shapes_are_caught(self, text: str) -> None:
        hits = scan_forbidden(text)
        assert any(hit.rule == "pricing" for hit in hits), text

    @pytest.mark.parametrize(
        "text",
        [
            "We migrate 40 workloads across 3 regions in 12 months.",
            "Our team of 25 engineers completed 8 migrations.",
        ],
    )
    def test_ordinary_numbers_are_not_pricing(self, text: str) -> None:
        """A guardrail that fires on every number is one people learn to
        override."""
        assert not [hit for hit in scan_forbidden(text) if hit.rule == "pricing"]

    @pytest.mark.parametrize(
        ("text", "term"),
        [
            ("We offer a 99.95% SLA on the platform.", "SLA"),
            ("The warranty period runs for twelve months.", "warranty"),
            ("We accept indemnity for data loss.", "indemnity"),
            ("Liquidated damages apply on late cutover.", "liquidated damages"),
        ],
    )
    def test_legal_terms_are_caught_and_named(self, text: str, term: str) -> None:
        """Named, because an SME handed only `legal_term` has to re-read the
        draft to learn which term."""
        hits = [hit for hit in scan_forbidden(text) if hit.rule.startswith("legal:")]
        assert hits
        assert hits[0].rule == f"legal:{term}"

    def test_a_term_inside_a_longer_word_is_not_a_hit(self) -> None:
        """Word boundaries: 'penalise' is not 'penalty', and 'translate'
        contains neither."""
        assert not [
            h
            for h in scan_forbidden("We translate and penalise nothing.")
            if h.rule.startswith("legal:")
        ]

    def test_one_span_produces_one_hit(self) -> None:
        """Two hits for one span would double-count in the zero-tolerance
        total the eval gates on."""
        hits = [
            hit
            for hit in scan_forbidden("Our service level agreement is attached.")
            if hit.rule.startswith("legal:")
        ]
        assert len(hits) == 1
        assert hits[0].rule == "legal:service level agreement"

    def test_clean_prose_produces_nothing(self) -> None:
        assert scan_forbidden("We migrate workloads in waves with rollback at each gate.") == []


class TestTheChecker:
    def test_an_answered_question_within_its_limit_passes(self) -> None:
        result = DeterministicComplianceChecker().check(
            question(word_limit=50), answer("Three words here.")
        )
        assert result.within_word_limit is True
        assert result.mandatory_answered is True

    def test_exceeding_the_stated_limit_is_recorded(self) -> None:
        result = DeterministicComplianceChecker().check(
            question(word_limit=3), answer("One two three four five.")
        )
        assert result.within_word_limit is False

    def test_exactly_the_limit_is_within_it(self) -> None:
        result = DeterministicComplianceChecker().check(
            question(word_limit=3), answer("One two three")
        )
        assert result.within_word_limit is True

    def test_a_question_with_no_limit_cannot_violate_one(self) -> None:
        result = DeterministicComplianceChecker().check(
            question(word_limit=None), answer("word " * 500)
        )
        assert result.within_word_limit is True

    def test_an_escalated_mandatory_question_is_not_answered(self) -> None:
        """A compliance FACT the report must state. Skipping escalated questions
        would make mandatory coverage describe only the ones that went well."""
        result = DeterministicComplianceChecker().check(question(mandatory=True), None)
        assert result.mandatory_answered is False

    def test_an_escalated_optional_question_is_not_a_coverage_failure(self) -> None:
        result = DeterministicComplianceChecker().check(question(mandatory=False), None)
        assert result.mandatory_answered is True

    def test_an_escalated_question_has_no_filled_slots(self) -> None:
        result = DeterministicComplianceChecker().check(question(), None)
        assert result.template_slots_filled is False

    @pytest.mark.parametrize(
        "text",
        [
            "We serve {customer_name} across three regions.",
            "Delivery centres: TBD.",
            "Our approach is <<insert approach>>.",
        ],
    )
    def test_an_unfilled_slot_is_caught(self, text: str) -> None:
        """A template rendered with a missing value produces a document that
        LOOKS complete and ships a literal placeholder to a customer."""
        result = DeterministicComplianceChecker().check(question(), answer(text))
        assert result.template_slots_filled is False

    def test_forbidden_content_reaches_the_result(self) -> None:
        result = DeterministicComplianceChecker().check(
            question(), answer("Priced at USD 400 per workload.")
        )
        assert [hit.rule for hit in result.forbidden_content_hits] == ["pricing"]


class TestTheSummary:
    def test_it_counts_what_the_evals_gate_on(self) -> None:
        checker = DeterministicComplianceChecker()
        results = [
            checker.check(question(qid="GQ-001", mandatory=True), None),
            checker.check(
                question(qid="GQ-002", word_limit=2), answer("too many words", qid="GQ-002")
            ),
            checker.check(question(qid="GQ-003"), answer("Costs $500.", qid="GQ-003")),
        ]
        counts = summarise(results)
        assert counts["questions"] == 3
        assert counts["mandatory_missing"] == 1
        assert counts["limit_violations"] == 1
        assert counts["forbidden_hits"] == 1

    def test_a_clean_run_summarises_to_zeroes(self) -> None:
        checker = DeterministicComplianceChecker()
        results = [checker.check(question(), answer("A clean grounded answer."))]
        counts = summarise(results)
        assert counts["mandatory_missing"] == 0
        assert counts["limit_violations"] == 0
        assert counts["forbidden_hits"] == 0
        assert counts["unfilled_slots"] == 0
