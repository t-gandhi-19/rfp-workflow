"""Intake invariants: section ordering, triage halts, question shape."""

from __future__ import annotations

from datetime import UTC, date, datetime

import pytest
from pydantic import ValidationError

from src.contracts import (
    DocumentFormat,
    DocumentSection,
    ExtractedQuestion,
    HaltReason,
    QuestionType,
    RFPDocument,
    TriageResult,
)

NOW = datetime(2026, 6, 1, 12, 0, 0, tzinfo=UTC)


class TestRFPDocument:
    def test_accepts_ordered_sections(self) -> None:
        doc = RFPDocument(
            id="rfp-001",
            source_filename="golden_rfp.pdf",
            format=DocumentFormat.PDF,
            raw_text="...",
            sections=[
                DocumentSection(name="Company", order=0, text="..."),
                DocumentSection(name="Technical Approach", order=1, text="..."),
            ],
            detected_deadline=date(2026, 7, 15),
            customer_name="Meridian Insurance Group",
            domain="cloud_migration",
            intake_timestamp=NOW,
        )
        assert [s.name for s in doc.sections] == ["Company", "Technical Approach"]

    def test_rejects_out_of_order_sections(self) -> None:
        with pytest.raises(ValidationError, match="ascending"):
            RFPDocument(
                id="rfp-001",
                source_filename="golden_rfp.pdf",
                format=DocumentFormat.PDF,
                raw_text="...",
                sections=[
                    DocumentSection(name="Technical Approach", order=2, text="..."),
                    DocumentSection(name="Company", order=1, text="..."),
                ],
                customer_name="Meridian Insurance Group",
                domain="cloud_migration",
                intake_timestamp=NOW,
            )

    def test_rejects_duplicate_section_order(self) -> None:
        with pytest.raises(ValidationError, match="unique"):
            RFPDocument(
                id="rfp-001",
                source_filename="golden_rfp.docx",
                format=DocumentFormat.DOCX,
                raw_text="...",
                sections=[
                    DocumentSection(name="Company", order=1, text="..."),
                    DocumentSection(name="Compliance", order=1, text="..."),
                ],
                customer_name="Meridian Insurance Group",
                domain="cloud_migration",
                intake_timestamp=NOW,
            )

    def test_rejects_naive_timestamp(self) -> None:
        """Naive datetimes and timestamptz columns disagree silently. Reject early."""
        with pytest.raises(ValidationError):
            RFPDocument(
                id="rfp-001",
                source_filename="golden_rfp.pdf",
                format=DocumentFormat.PDF,
                raw_text="...",
                customer_name="Meridian Insurance Group",
                domain="cloud_migration",
                intake_timestamp=datetime(2026, 6, 1, 12, 0, 0),
            )


class TestTriageResult:
    def test_accepts_a_domain_match(self) -> None:
        triage = TriageResult(
            rfp_id="rfp-001", domain_match=True, detected_domain="cloud_migration"
        )
        assert triage.halt_reason is None

    def test_rejects_a_mismatch_without_a_halt_reason(self) -> None:
        """A mismatch that does not halt would quietly draft in the wrong domain."""
        with pytest.raises(ValidationError, match="halt_reason"):
            TriageResult(rfp_id="rfp-001", domain_match=False, detected_domain="payroll_services")

    def test_accepts_a_mismatch_that_halts(self) -> None:
        triage = TriageResult(
            rfp_id="rfp-001",
            domain_match=False,
            detected_domain="payroll_services",
            halt_reason=HaltReason.DOMAIN_MISMATCH,
        )
        assert triage.halt_reason is HaltReason.DOMAIN_MISMATCH

    def test_rejects_contradictory_match_and_halt(self) -> None:
        with pytest.raises(ValidationError, match="contradicts"):
            TriageResult(
                rfp_id="rfp-001",
                domain_match=True,
                detected_domain="cloud_migration",
                halt_reason=HaltReason.DOMAIN_MISMATCH,
            )


class TestExtractedQuestion:
    def test_accepts_a_well_formed_question(self) -> None:
        question = ExtractedQuestion(
            id="q-001",
            rfp_id="rfp-001",
            text="3.2 Describe your approach to dependency discovery.",
            normalized_text="Describe your approach to dependency discovery.",
            section="Technical Approach",
            question_type=QuestionType.TECHNICAL,
            word_limit=500,
            mandatory=True,
            order=4,
        )
        assert question.normalized_text != question.text

    @pytest.mark.parametrize("field", ["text", "normalized_text", "section"])
    def test_rejects_empty_required_text(self, field: str) -> None:
        payload = {
            "id": "q-001",
            "rfp_id": "rfp-001",
            "text": "Describe your approach.",
            "normalized_text": "Describe your approach.",
            "section": "Technical Approach",
            "question_type": QuestionType.TECHNICAL,
            "order": 0,
        }
        payload[field] = ""
        with pytest.raises(ValidationError):
            ExtractedQuestion.model_validate(payload)

    def test_rejects_a_non_positive_word_limit(self) -> None:
        with pytest.raises(ValidationError):
            ExtractedQuestion(
                id="q-001",
                rfp_id="rfp-001",
                text="Describe your approach.",
                normalized_text="Describe your approach.",
                section="Technical Approach",
                question_type=QuestionType.TECHNICAL,
                word_limit=0,
                order=0,
            )

    def test_word_limit_is_optional(self) -> None:
        question = ExtractedQuestion(
            id="q-002",
            rfp_id="rfp-001",
            text="List your certifications.",
            normalized_text="List your certifications.",
            section="Company",
            question_type=QuestionType.COMPANY_INFO,
            order=1,
        )
        assert question.word_limit is None
        assert question.mandatory is False
