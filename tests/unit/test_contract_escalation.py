"""`escalations.json` and its records.

The load-bearing assertions here are about what an SME RECEIVES: which question
a record names, that they receive each question once, and that the list walks
the document in the order they will read it.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from src.contracts import EscalationRecord, EscalationsRecord, EscalationTrigger

NOW = datetime(2026, 6, 1, 12, 0, 0, tzinfo=UTC)


def record(
    *,
    question_id: str = "GQ-001",
    printed_number: str | None = "3.4",
    order: int = 0,
    trigger: EscalationTrigger = EscalationTrigger.NO_MATCH,
    reason: str = "retrieval cleared no candidate over the relevance floor",
    sme_id: str | None = None,
    sme_name: str | None = None,
) -> EscalationRecord:
    return EscalationRecord(
        question_id=question_id,
        printed_number=printed_number,
        order=order,
        question_text="Describe your approach to migrating stateful workloads.",
        trigger=trigger,
        reason=reason,
        sme_id=sme_id,
        sme_name=sme_name,
    )


class TestTheQuestionReference:
    """Amendment I: a reviewer looks for "Question 3.4", not for our index."""

    def test_the_printed_number_is_preferred(self) -> None:
        assert record(printed_number="3.4", order=7).reference == "3.4"

    def test_an_unnumbered_question_falls_back_to_a_marked_index(self) -> None:
        """Marked, so a reviewer who cannot find "Q8" in their document learns
        that the document numbered nothing, rather than hunting for a number
        that was never printed."""
        assert record(printed_number=None, order=7).reference == "[unnumbered #8]"

    def test_the_fallback_is_one_based(self) -> None:
        """`order` is zero-based internally and no human counts that way."""
        assert record(printed_number=None, order=0).reference == "[unnumbered #1]"

    def test_a_blank_printed_number_is_refused_rather_than_used(self) -> None:
        """An empty string would render as an empty reference — a TODO block
        naming no question at all, which is worse than the honest fallback."""
        with pytest.raises(ValidationError):
            record(printed_number="")


class TestTheSmeIdentity:
    def test_neither_is_the_normal_case(self) -> None:
        """Not every question maps to a capability, so not every escalation
        resolves an SME."""
        assert record().sme_id is None

    def test_both_together_are_accepted(self) -> None:
        resolved = record(sme_id="sme-014", sme_name="Priya Raghavan")
        assert (resolved.sme_id, resolved.sme_name) == ("sme-014", "Priya Raghavan")

    def test_an_id_without_a_name_is_refused(self) -> None:
        """An id with no name cannot be addressed by the person reading it."""
        with pytest.raises(ValidationError, match="travel together"):
            record(sme_id="sme-014")

    def test_a_name_without_an_id_is_refused(self) -> None:
        """A name with no id cannot be looked up to check it is still current."""
        with pytest.raises(ValidationError, match="travel together"):
            record(sme_name="Priya Raghavan")


class TestTheReasonIsNeverBlank:
    def test_a_reason_is_required(self) -> None:
        """Build prompt §12: the TODO blocks are never blank."""
        with pytest.raises(ValidationError):
            record(reason="")

    def test_the_trigger_alone_does_not_substitute_for_it(self) -> None:
        """The trigger says WHICH RULE fired; the reason says what was found.
        An SME handed only `legal_term` still has to re-read the draft to learn
        which term."""
        escalation = record(
            trigger=EscalationTrigger.LEGAL_TERM,
            reason="draft offered a 99.95% availability SLA; no source commits to one",
        )
        assert escalation.trigger is EscalationTrigger.LEGAL_TERM
        assert "SLA" in escalation.reason


class TestTheArtifact:
    def test_an_empty_escalation_list_is_valid(self) -> None:
        """A run that escalated nothing is a good run, not a broken artifact."""
        artifact = EscalationsRecord(
            run_id="run-1", rfp_id="rfp-1", generated_at=NOW, escalations=[]
        )
        assert artifact.escalations == []

    def test_one_record_per_question(self) -> None:
        """Several guardrails can fire on one draft, and the natural loop
        appends a record per hit — handing the SME the same question three
        times and inflating the count the operational report publishes."""
        with pytest.raises(ValidationError, match="escalated more than once"):
            EscalationsRecord(
                run_id="run-1",
                rfp_id="rfp-1",
                generated_at=NOW,
                escalations=[
                    record(
                        question_id="GQ-004", order=0, trigger=EscalationTrigger.PROMPT_INJECTION
                    ),
                    record(question_id="GQ-004", order=1, trigger=EscalationTrigger.LOW_CONFIDENCE),
                ],
            )

    def test_records_are_ordered_as_the_document_reads(self) -> None:
        """An SME works through this next to the source document."""
        with pytest.raises(ValidationError, match="ascending question order"):
            EscalationsRecord(
                run_id="run-1",
                rfp_id="rfp-1",
                generated_at=NOW,
                escalations=[
                    record(question_id="GQ-009", order=8),
                    record(question_id="GQ-002", order=1),
                ],
            )

    def test_ascending_order_is_accepted(self) -> None:
        artifact = EscalationsRecord(
            run_id="run-1",
            rfp_id="rfp-1",
            generated_at=NOW,
            escalations=[
                record(question_id="GQ-002", order=1),
                record(question_id="GQ-009", order=8),
            ],
        )
        assert [e.order for e in artifact.escalations] == [1, 8]

    def test_the_timestamp_must_be_aware(self) -> None:
        """A naive timestamp in an artifact a human reads across timezones is a
        number that means nothing."""
        with pytest.raises(ValidationError):
            EscalationsRecord(
                run_id="run-1",
                rfp_id="rfp-1",
                generated_at=datetime(2026, 6, 1, 12, 0, 0),
                escalations=[],
            )
