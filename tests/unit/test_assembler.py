"""The assembler: the response body, and `escalations.json`.

The load-bearing assertions are about what a HUMAN receives — which question a
TODO block names, who it is assigned to, and that no block is ever blank.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from src.assembly import TemplateAssembler
from src.contracts import (
    DraftedAnswer,
    EscalationRecord,
    EscalationsRecord,
    EscalationTrigger,
    ExtractedQuestion,
    QuestionType,
)
from tests.stubs import make_document

NOW = datetime(2026, 6, 1, 12, 0, tzinfo=UTC)


def question(
    qid: str, *, order: int, printed: str | None = "3.1", section: str = "Technical Approach"
) -> ExtractedQuestion:
    return ExtractedQuestion(
        id=qid,
        rfp_id="rfp-golden",
        text=f"Question {qid}: describe your approach.",
        normalized_text="describe your approach",
        section=section,
        question_type=QuestionType.TECHNICAL,
        order=order,
        printed_number=printed,
    )


def answered(qid: str) -> DraftedAnswer:
    return DraftedAnswer(
        question_id=qid,
        answer_text=f"A grounded answer for {qid}.",
        source_ids=["ANS-0001", "ANS-0002"],
        confidence=0.87,
        needs_sme_review=False,
    )


def escalation(
    qid: str,
    *,
    order: int,
    printed: str | None = "3.1",
    trigger: EscalationTrigger = EscalationTrigger.NO_MATCH,
    draft: str | None = None,
    detail: str | None = None,
) -> EscalationRecord:
    return EscalationRecord(
        question_id=qid,
        printed_number=printed,
        order=order,
        question_text=f"Question {qid}: describe your approach.",
        trigger=trigger,
        reason="retrieval cleared no candidate over the relevance floor",
        detail=detail,
        draft_text=draft,
    )


def record(records: list[EscalationRecord]) -> EscalationsRecord:
    return EscalationsRecord(
        run_id="run-1", rfp_id="rfp-golden", generated_at=NOW, escalations=records
    )


def assemble(tmp_path: Path, assembler: TemplateAssembler, questions, answers, escalations):  # type: ignore[no-untyped-def]
    return assembler.assemble(
        run_id="run-1",
        document=make_document(),
        questions=questions,
        answers=answers,
        escalations=escalations,
    )


class TestTheResponseBody:
    def test_an_answered_question_carries_its_sources_and_confidence(self, tmp_path: Path) -> None:
        assembler = TemplateAssembler(out_dir=tmp_path)
        paths = assemble(
            tmp_path,
            assembler,
            [question("GQ-001", order=0)],
            {"GQ-001": answered("GQ-001")},
            record([]),
        )
        body = Path(paths["response_docx"]).read_text(encoding="utf-8")
        assert "ANS-0001, ANS-0002" in body
        assert "0.87 (computed)" in body

    def test_the_header_says_nothing_is_submitted(self, tmp_path: Path) -> None:
        """Rule 2. The document a human opens says so on its first page."""
        assembler = TemplateAssembler(out_dir=tmp_path)
        paths = assemble(
            tmp_path,
            assembler,
            [question("GQ-001", order=0)],
            {},
            record([escalation("GQ-001", order=0)]),
        )
        body = Path(paths["response_docx"]).read_text(encoding="utf-8")
        assert "NOT submitted" in body

    def test_questions_appear_in_document_order(self, tmp_path: Path) -> None:
        assembler = TemplateAssembler(out_dir=tmp_path)
        questions = [
            question("GQ-001", order=0, printed="1.1"),
            question("GQ-002", order=1, printed="1.2"),
            question("GQ-003", order=2, printed="2.1"),
        ]
        paths = assemble(
            tmp_path,
            assembler,
            questions,
            {q.id: answered(q.id) for q in questions},
            record([]),
        )
        body = Path(paths["response_docx"]).read_text(encoding="utf-8")
        assert body.index("1.1") < body.index("1.2") < body.index("2.1")


class TestTheSmeTodoBlocks:
    def test_an_escalated_question_gets_a_todo_and_no_answer(self, tmp_path: Path) -> None:
        assembler = TemplateAssembler(out_dir=tmp_path)
        paths = assemble(
            tmp_path,
            assembler,
            [question("GQ-001", order=0)],
            {},
            record([escalation("GQ-001", order=0)]),
        )
        body = Path(paths["response_docx"]).read_text(encoding="utf-8")
        assert "SME TODO — NOT ANSWERED" in body
        assert "no_match" in body

    def test_the_block_names_the_printed_number(self, tmp_path: Path) -> None:
        """Amendment I. A reviewer looks for "Question 3.4" in the document
        they were sent; our zero-based index means nothing to them."""
        assembler = TemplateAssembler(out_dir=tmp_path)
        paths = assemble(
            tmp_path,
            assembler,
            [question("GQ-001", order=7, printed="3.4")],
            {},
            record([escalation("GQ-001", order=7, printed="3.4")]),
        )
        body = Path(paths["response_docx"]).read_text(encoding="utf-8")
        assert "3.4" in body

    def test_an_unnumbered_question_says_so_rather_than_faking_a_number(
        self, tmp_path: Path
    ) -> None:
        assembler = TemplateAssembler(out_dir=tmp_path)
        paths = assemble(
            tmp_path,
            assembler,
            [question("GQ-001", order=7, printed=None)],
            {},
            record([escalation("GQ-001", order=7, printed=None)]),
        )
        body = Path(paths["response_docx"]).read_text(encoding="utf-8")
        assert "[unnumbered #8]" in body

    def test_a_block_is_never_blank(self, tmp_path: Path) -> None:
        """Build prompt §12. Every block carries a trigger and a reason."""
        assembler = TemplateAssembler(out_dir=tmp_path)
        paths = assemble(
            tmp_path,
            assembler,
            [question("GQ-001", order=0)],
            {},
            record([escalation("GQ-001", order=0)]),
        )
        body = Path(paths["response_docx"]).read_text(encoding="utf-8")
        assert "relevance floor" in body

    def test_a_flagged_draft_is_offered_to_the_sme(self, tmp_path: Path) -> None:
        """An SME starting from a flagged draft is doing review; one starting
        from nothing is doing the whole question."""
        assembler = TemplateAssembler(out_dir=tmp_path)
        paths = assemble(
            tmp_path,
            assembler,
            [question("GQ-001", order=0)],
            {},
            record([escalation("GQ-001", order=0, draft="A draft the guardrail stopped.")]),
        )
        body = Path(paths["response_docx"]).read_text(encoding="utf-8")
        assert "NOT part of the response" in body
        assert "A draft the guardrail stopped." in body

    def test_the_detail_reaches_the_block(self, tmp_path: Path) -> None:
        assembler = TemplateAssembler(out_dir=tmp_path)
        paths = assemble(
            tmp_path,
            assembler,
            [question("GQ-001", order=0)],
            {},
            record(
                [
                    escalation(
                        "GQ-001",
                        order=0,
                        trigger=EscalationTrigger.PROMPT_INJECTION,
                        detail="approval_request",
                    )
                ]
            ),
        )
        body = Path(paths["response_docx"]).read_text(encoding="utf-8")
        assert "Found: approval_request" in body


class TestSmeRouting:
    def test_a_mapped_capability_names_its_owner(self, tmp_path: Path) -> None:
        assembler = TemplateAssembler(
            out_dir=tmp_path,
            capability_for_question={"GQ-001": "CAP-0003"},
            sme_for_capability={"CAP-0003": ("SME-0001", "Priya Raghunathan")},
        )
        paths = assemble(
            tmp_path,
            assembler,
            [question("GQ-001", order=0)],
            {},
            record([escalation("GQ-001", order=0)]),
        )
        body = Path(paths["response_docx"]).read_text(encoding="utf-8")
        assert "Priya Raghunathan (SME-0001)" in body

    def test_an_unmapped_capability_admits_it_needs_assigning(self, tmp_path: Path) -> None:
        """A TODO with no owner that LOOKS complete is worse than one that says
        it is unassigned."""
        assembler = TemplateAssembler(out_dir=tmp_path)
        paths = assemble(
            tmp_path,
            assembler,
            [question("GQ-001", order=0)],
            {},
            record([escalation("GQ-001", order=0)]),
        )
        body = Path(paths["response_docx"]).read_text(encoding="utf-8")
        assert "unassigned" in body

    def test_the_owner_reaches_escalations_json_too(self, tmp_path: Path) -> None:
        """One record, two renderings — they cannot disagree about the owner."""
        assembler = TemplateAssembler(
            out_dir=tmp_path,
            capability_for_question={"GQ-001": "CAP-0003"},
            sme_for_capability={"CAP-0003": ("SME-0001", "Priya Raghunathan")},
        )
        paths = assemble(
            tmp_path,
            assembler,
            [question("GQ-001", order=0)],
            {},
            record([escalation("GQ-001", order=0)]),
        )
        payload = json.loads(Path(paths["escalations_json"]).read_text(encoding="utf-8"))
        assert payload["escalations"][0]["sme_name"] == "Priya Raghunathan"
        assert payload["escalations"][0]["sme_id"] == "SME-0001"
        assert payload["escalations"][0]["capability_id"] == "CAP-0003"


class TestEscalationsJson:
    def test_it_is_valid_json_and_round_trips(self, tmp_path: Path) -> None:
        assembler = TemplateAssembler(out_dir=tmp_path)
        paths = assemble(
            tmp_path,
            assembler,
            [question("GQ-001", order=0)],
            {},
            record([escalation("GQ-001", order=0)]),
        )
        payload = json.loads(Path(paths["escalations_json"]).read_text(encoding="utf-8"))
        assert EscalationsRecord.model_validate(payload).run_id == "run-1"

    def test_an_empty_escalation_list_still_writes_the_file(self, tmp_path: Path) -> None:
        """A run that escalated nothing is a good run. A missing file would be
        indistinguishable from a run that never got that far."""
        assembler = TemplateAssembler(out_dir=tmp_path)
        paths = assemble(
            tmp_path,
            assembler,
            [question("GQ-001", order=0)],
            {"GQ-001": answered("GQ-001")},
            record([]),
        )
        payload = json.loads(Path(paths["escalations_json"]).read_text(encoding="utf-8"))
        assert payload["escalations"] == []


class TestAssemblyIsDeterministic:
    def test_the_same_inputs_produce_byte_identical_output(self, tmp_path: Path) -> None:
        """What makes the resumed-artifact comparison meaningful at all."""
        questions = [question("GQ-001", order=0), question("GQ-002", order=1, printed="1.2")]
        answers = {"GQ-001": answered("GQ-001")}
        escalations = record([escalation("GQ-002", order=1, printed="1.2")])

        first = TemplateAssembler(out_dir=tmp_path / "a")
        second = TemplateAssembler(out_dir=tmp_path / "b")
        paths_a = assemble(tmp_path, first, questions, answers, escalations)
        paths_b = assemble(tmp_path, second, questions, answers, escalations)

        assert (
            Path(paths_a["response_docx"]).read_bytes()
            == Path(paths_b["response_docx"]).read_bytes()
        )
