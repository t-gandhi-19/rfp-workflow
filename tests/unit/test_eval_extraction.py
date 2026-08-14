"""The extraction category, measured against the shipped manual key.

TWO KINDS OF TEST HERE, kept apart on purpose.

The first kind runs the real eval over the real fixtures and asserts the
*properties the harness must have* — that pairing is by text, that a disagreement
becomes a named finding, that the key is never written. These would survive the
key being replaced.

The second kind asserts the CURRENT measured outcome: zero disagreements across
all twenty questions. That is a fact about this corpus at this commit, and it is
recorded here so a regression is loud. It is NOT a licence to edit the key if it
ever goes red — see the module docstring of src/evals/manual_key.py.

No test in this file asserts the key's expected CONTENTS. Doing so would make the
builder the arbiter of the ground truth it is being measured against.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from src.evals.contracts import CategoryStatus
from src.evals.extraction import (
    COMPARED_FIELDS,
    EXPECTED_INJECTION_PATTERNS,
    INJECTION_QUESTION_NUMBER,
    ExtractionRun,
    FieldDisagreement,
    normalise_for_match,
    run_extraction_eval,
    to_category_result,
)
from src.evals.manual_key import MANUAL_KEY_PATH


@pytest.fixture(scope="module")
def run() -> ExtractionRun:
    return run_extraction_eval()


class TestTheMeasuredOutcome:
    """What the extractor and the key actually agree on, at this commit."""

    def test_the_category_passes(self, run: ExtractionRun) -> None:
        assert to_category_result(run).status is CategoryStatus.PASS

    def test_every_key_question_is_found(self, run: ExtractionRun) -> None:
        assert run.missed == []
        assert run.recall == 1.0

    def test_no_question_is_invented(self, run: ExtractionRun) -> None:
        assert run.spurious == []
        assert run.precision == 1.0

    def test_every_compared_field_agrees(self, run: ExtractionRun) -> None:
        """Zero disagreements across all twenty questions.

        Red here is a FINDING about the extractor or the key, to be reported with
        both sides quoted. It is never a reason to edit fixtures/answer_key_manual.json.
        """
        assert [d.describe() for d in run.field_disagreements] == []
        assert run.field_accuracy == 1.0

    def test_pdf_and_docx_yield_the_same_questions(self, run: ExtractionRun) -> None:
        assert run.parity_differences == []
        assert len(run.pdf_questions) == len(run.docx_questions)

    def test_the_key_and_the_extractor_see_the_same_count(self, run: ExtractionRun) -> None:
        assert len(run.pdf_questions) == len(run.manual.questions)


class TestTheInjectionCarrier:
    def test_both_expected_patterns_fire(self, run: ExtractionRun) -> None:
        assert set(EXPECTED_INJECTION_PATTERNS) <= set(run.injection_patterns)

    def test_no_clean_question_is_flagged(self, run: ExtractionRun) -> None:
        """A flag that fires everywhere is a flag nobody reads."""
        assert run.injection_false_positives == []

    def test_the_metric_names_the_patterns_rather_than_a_boolean(self, run: ExtractionRun) -> None:
        """An escalation that cannot say WHAT it found gives a reviewer nothing."""
        metric = next(m for m in to_category_result(run).metrics if m.key == "injection_flagged")
        assert metric.detail is not None
        for pattern in EXPECTED_INJECTION_PATTERNS:
            assert pattern in metric.detail

    def test_one_pattern_alone_is_not_enough(self, run: ExtractionRun) -> None:
        """The carrier is two attacks in one sentence, and the auto-submit half
        aims at the rule this project treats as inviolable."""
        partial = replace(
            ExtractionRun(
                manual=run.manual,
                pdf_questions=run.pdf_questions,
                docx_questions=run.docx_questions,
            ),
            injection_patterns=("ignore_previous_instructions",),
        )
        assert partial.injection_flagged is False

    def test_the_carrier_number_is_the_one_in_the_document(self, run: ExtractionRun) -> None:
        """Guards against the constant drifting off the fixture it names."""
        numbers = {question.printed_number for question in run.pdf_questions}
        assert INJECTION_QUESTION_NUMBER in numbers


class TestPairingIsByText:
    """Pairing on a compared field would make that field unmeasurable."""

    def test_printed_number_is_compared_not_matched_on(self) -> None:
        assert "printed_number" in COMPARED_FIELDS

    def test_order_is_compared_not_matched_on(self) -> None:
        assert "order" in COMPARED_FIELDS

    def test_matching_folds_case_and_whitespace(self) -> None:
        assert normalise_for_match("  How  Many\tteams? ") == normalise_for_match("how many teams?")

    def test_matching_does_not_fold_punctuation(self) -> None:
        """A key ending '…estate?' and an extractor ending '…estate' disagree
        about where the question ends. That is a finding, not noise."""
        assert normalise_for_match("the estate?") != normalise_for_match("the estate")


class TestDisagreementsBecomeNamedFindings:
    def test_a_field_disagreement_quotes_both_sides(self) -> None:
        """A report saying only 'printed_number differs' obliges the reader to
        re-run the eval before they can start triaging."""
        described = FieldDisagreement(
            number="2.3", field_name="word_limit", manual_value=300, extracted_value=None
        ).describe()
        assert "2.3" in described
        assert "word_limit" in described
        assert "300" in described
        assert "None" in described

    def test_a_missed_question_is_a_violation_naming_the_question(self, run: ExtractionRun) -> None:
        injected = ExtractionRun(
            manual=run.manual,
            pdf_questions=run.pdf_questions,
            docx_questions=run.docx_questions,
        )
        injected.missed = [run.manual.in_document_order()[0]]
        result = to_category_result(injected)
        recall_violations = [v for v in result.violations if v.rule == "extraction_recall"]
        assert len(recall_violations) == 1
        assert recall_violations[0].question_number == run.manual.in_document_order()[0].number

    def test_a_violation_says_the_key_is_not_edited(self, run: ExtractionRun) -> None:
        """The instruction travels with the finding, because the moment someone
        reads it is the moment they are deciding what to do about it."""
        injected = ExtractionRun(
            manual=run.manual,
            pdf_questions=run.pdf_questions,
            docx_questions=run.docx_questions,
        )
        injected.missed = [run.manual.in_document_order()[0]]
        detail = to_category_result(injected).violations[0].detail
        assert "key is not edited" in detail

    def test_any_violation_fails_the_category(self, run: ExtractionRun) -> None:
        injected = ExtractionRun(
            manual=run.manual,
            pdf_questions=run.pdf_questions,
            docx_questions=run.docx_questions,
        )
        injected.missed = [run.manual.in_document_order()[0]]
        assert to_category_result(injected).status is CategoryStatus.FAIL


class TestFieldAccuracyCountsFields:
    def test_the_denominator_is_questions_times_fields(self, run: ExtractionRun) -> None:
        """Three wrong fields on one question is three defects, not one."""
        injected = ExtractionRun(
            manual=run.manual,
            pdf_questions=run.pdf_questions,
            docx_questions=run.docx_questions,
        )
        injected.matched = run.matched
        injected.field_disagreements = [
            FieldDisagreement(number="1.1", field_name=name, manual_value=1, extracted_value=2)
            for name in ("printed_number", "section", "mandatory")
        ]
        total = len(run.matched) * len(COMPARED_FIELDS)
        assert injected.field_accuracy == pytest.approx((total - 3) / total)


class TestTheKeyIsNeverWritten:
    def test_running_the_eval_does_not_modify_the_key(self) -> None:
        """The builder-immutable rule, asserted rather than trusted.

        Compares bytes before and after, not a timestamp: a rewrite with
        identical content would still be a rewrite, but a rewrite with DIFFERENT
        content is the failure that matters and this catches it.
        """
        before = MANUAL_KEY_PATH.read_bytes()
        run_extraction_eval()
        assert MANUAL_KEY_PATH.read_bytes() == before
