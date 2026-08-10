"""Extraction: parity between renditions, and accuracy against the answer key.

These are written as tests now and graduate into the Phase 3 harness unchanged.

The parity test is the important one. Both fixtures are rendered from a single
source structure, so any difference between the extracted question lists is a
property of the *extractor* — PDF line wrapping, DOCX paragraph handling — which
is exactly what it is supposed to prove is handled.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from src.contracts import ExtractedQuestion, QuestionType
from src.extraction.document import TEXT_DENSITY_THRESHOLD, ExtractionError, extract
from src.extraction.questions import normalise, parse_annotation, parse_questions
from src.guardrails.injection import scan

FIXTURES = Path(__file__).resolve().parents[2] / "fixtures"
PDF = FIXTURES / "golden_rfp.pdf"
DOCX = FIXTURES / "golden_rfp.docx"

RFP_ID = "rfp-golden"


def _extract(path: Path) -> list[ExtractedQuestion]:
    return parse_questions(extract(path).text, rfp_id=RFP_ID)


@pytest.fixture(scope="module")
def from_pdf() -> list[ExtractedQuestion]:
    return _extract(PDF)


@pytest.fixture(scope="module")
def from_docx() -> list[ExtractedQuestion]:
    return _extract(DOCX)


@pytest.fixture(scope="module")
def answer_key() -> dict[str, Any]:
    with (FIXTURES / "answer_key.json").open(encoding="utf-8") as handle:
        return dict(json.load(handle))


@pytest.fixture(scope="module")
def source() -> dict[str, Any]:
    with (FIXTURES / "golden_rfp_source.json").open(encoding="utf-8") as handle:
        return dict(json.load(handle))


class TestParity:
    """Field-for-field equality between the two renditions."""

    def test_same_number_of_questions(
        self, from_pdf: list[ExtractedQuestion], from_docx: list[ExtractedQuestion]
    ) -> None:
        assert len(from_pdf) == len(from_docx) == 20

    def test_identical_question_lists(
        self, from_pdf: list[ExtractedQuestion], from_docx: list[ExtractedQuestion]
    ) -> None:
        assert [q.model_dump() for q in from_pdf] == [q.model_dump() for q in from_docx]

    @pytest.mark.parametrize(
        "field", ["normalized_text", "section", "question_type", "word_limit", "mandatory", "order"]
    )
    def test_field_by_field(
        self, from_pdf: list[ExtractedQuestion], from_docx: list[ExtractedQuestion], field: str
    ) -> None:
        assert [getattr(q, field) for q in from_pdf] == [getattr(q, field) for q in from_docx]


class TestAccuracy:
    """Recall and precision against the answer key (build prompt §20)."""

    def _expected(self, source: dict[str, Any]) -> list[dict[str, Any]]:
        return sorted(source["questions"], key=lambda q: q["order"])

    def test_recall_and_precision(
        self, from_pdf: list[ExtractedQuestion], source: dict[str, Any]
    ) -> None:
        expected = {" ".join(q["text"].split()).casefold() for q in self._expected(source)}
        extracted = {q.normalized_text.casefold() for q in from_pdf}

        true_positives = len({e for e in expected if any(e in x or x in e for x in extracted)})
        recall = true_positives / len(expected)
        precision = true_positives / len(extracted)
        assert recall >= 0.95, f"recall {recall:.3f}"
        assert precision >= 0.95, f"precision {precision:.3f}"

    def test_mandatory_flags_match(
        self, from_pdf: list[ExtractedQuestion], source: dict[str, Any]
    ) -> None:
        assert [q.mandatory for q in from_pdf] == [q["mandatory"] for q in self._expected(source)]

    def test_word_limits_match(
        self, from_pdf: list[ExtractedQuestion], source: dict[str, Any]
    ) -> None:
        assert [q.word_limit for q in from_pdf] == [q["word_limit"] for q in self._expected(source)]

    def test_sections_match(
        self, from_pdf: list[ExtractedQuestion], source: dict[str, Any]
    ) -> None:
        assert [q.section for q in from_pdf] == [q["section"] for q in self._expected(source)]

    def test_question_types_match(
        self, from_pdf: list[ExtractedQuestion], source: dict[str, Any]
    ) -> None:
        assert [q.question_type.value for q in from_pdf] == [
            q["question_type"] for q in self._expected(source)
        ]

    def test_counts_match_the_answer_key(
        self, from_pdf: list[ExtractedQuestion], answer_key: dict[str, Any]
    ) -> None:
        assert sum(1 for q in from_pdf if q.mandatory) == answer_key["counts"]["mandatory"]
        assert (
            sum(1 for q in from_pdf if q.word_limit is not None)
            == answer_key["counts"]["with_word_limit"]
        )


class TestNormalisation:
    def test_strips_numbering_and_annotation(self) -> None:
        raw = "3.2 How do you guarantee residency? [Mandatory; maximum 300 words]"
        assert normalise(raw) == "How do you guarantee residency?"

    def test_collapses_wrapped_whitespace(self) -> None:
        """PDF extraction produces ragged spacing; matching must not depend on it."""
        assert normalise("2.1  Describe   your\tapproach.") == "Describe your approach."

    def test_raw_text_is_preserved_on_the_contract(self, from_pdf: list[ExtractedQuestion]) -> None:
        numbered = [q for q in from_pdf if q.text != q.normalized_text]
        assert len(numbered) == len(from_pdf)

    @pytest.mark.parametrize(
        ("annotation", "expected"),
        [
            ("[Mandatory]", (True, None)),
            ("[maximum 250 words]", (False, 250)),
            ("[Mandatory; maximum 500 words]", (True, 500)),
            ("", (False, None)),
        ],
    )
    def test_annotation_parsing(self, annotation: str, expected: tuple[bool, int | None]) -> None:
        assert parse_annotation(f"1.1 Some question. {annotation}".strip()) == expected


class TestInjectionInTheGoldenRfp:
    def test_the_embedded_injection_is_detected(
        self, from_pdf: list[ExtractedQuestion], source: dict[str, Any]
    ) -> None:
        """The specific string planted in the golden RFP must be flagged."""
        carriers = [q for q in from_pdf if scan(q.text)]
        assert len(carriers) == 1
        hits = scan(carriers[0].text)
        assert "ignore_previous_instructions" in {hit.pattern_name for hit in hits}

    def test_it_survives_both_renditions(
        self, from_pdf: list[ExtractedQuestion], from_docx: list[ExtractedQuestion]
    ) -> None:
        assert [bool(scan(q.text)) for q in from_pdf] == [bool(scan(q.text)) for q in from_docx]

    def test_ordinary_questions_are_not_flagged(self, from_pdf: list[ExtractedQuestion]) -> None:
        """A scanner that flags everything is as useless as one that flags nothing."""
        flagged = sum(1 for q in from_pdf if scan(q.normalized_text))
        assert flagged == 1


class TestDocumentLoading:
    def test_pdf_reports_its_format_and_pages(self) -> None:
        result = extract(PDF)
        assert result.document_format.value == "pdf"
        assert result.page_count >= 2

    def test_docx_reports_its_format(self) -> None:
        assert extract(DOCX).document_format.value == "docx"

    def test_the_golden_pdf_needs_no_ocr(self) -> None:
        """It has a real text layer; OCR would signal a broken fixture."""
        assert extract(PDF).ocr_pages == ()

    def test_unsupported_format_fails_readably(self, tmp_path: Path) -> None:
        stray = tmp_path / "notes.txt"
        stray.write_text("not a document", encoding="utf-8")
        with pytest.raises(ExtractionError, match="unsupported"):
            extract(stray)

    def test_density_threshold_is_a_real_threshold(self) -> None:
        """Guards against someone setting it to 0 and disabling OCR fallback."""
        assert TEXT_DENSITY_THRESHOLD > 0


class TestParserRobustness:
    def test_ignores_cover_page_prose(self) -> None:
        text = "\n".join(
            [
                "Request for Proposal",
                "Reference: RFP-2026-MIG-014",
                "Section Company",
                "1.1 What is your team model? [Mandatory]",
            ]
        )
        questions = parse_questions(text, rfp_id="x")
        assert len(questions) == 1
        assert questions[0].section == "Company"

    def test_reassembles_a_wrapped_question(self) -> None:
        """PDF wrapping must not split one question into several."""
        text = "\n".join(
            [
                "Section Compliance",
                "3.1 Describe how personal data is handled during migration",
                "and how GDPR obligations are met. [Mandatory]",
            ]
        )
        questions = parse_questions(text, rfp_id="x")
        assert len(questions) == 1
        assert questions[0].normalized_text.endswith("GDPR obligations are met.")

    def test_orders_are_sequential_and_zero_based(self) -> None:
        text = "\n".join(
            [
                "Section Company",
                "1.1 First? [Mandatory]",
                "1.2 Second?",
                "Section Delivery",
                "4.1 Third?",
            ]
        )
        questions = parse_questions(text, rfp_id="x")
        assert [q.order for q in questions] == [0, 1, 2]
        assert questions[2].question_type is QuestionType.COMMERCIAL

    def test_empty_document_yields_no_questions(self) -> None:
        assert parse_questions("", rfp_id="x") == []
