"""The fixed-template assembler, and `escalations.json`.

TWO ARTIFACTS FROM ONE SET OF FACTS. The response document carries an SME-TODO
block for every escalated question; `escalations.json` carries the same records
for a reviewer working through them. `EscalationRecord.reference` is computed
once and used by both, so they cannot drift about which question is which.

AMENDMENT I IS HONOURED HERE OR NOWHERE. A TODO block says "Question 3.4",
because that is what a reviewer looks for in the document they were sent. Our
zero-based `order` means nothing to them and cannot be cross-checked against the
RFP. Where the document numbered nothing, the reference says so explicitly
rather than printing an internal index that looks like a printed one.

THE SME NAME IS RESOLVED HERE, not in the controller, because this is where the
graph session is and because a name is only useful once there is a block to put
it in. Where a question's capability does not map, the block says the routing is
unresolved rather than omitting the line — a TODO with no owner that LOOKS
complete is worse than one that admits it needs assigning.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

from src.contracts import (
    DraftedAnswer,
    EscalationRecord,
    EscalationsRecord,
    ExtractedQuestion,
    RFPDocument,
)

logger = logging.getLogger("rfp.assembly")

#: Where a run's artifacts land. `out/<run_id>/`.
DEFAULT_OUT_DIR = Path("out")

TEMPLATE_HEADER = """RESPONSE TO REQUEST FOR PROPOSAL
{customer}
Source document: {source}
Run: {run_id}

This response was drafted by an automated workflow and is NOT submitted.
Every answer below carries the source ids it was drafted from. Questions marked
SME TODO were escalated and have no answer; a human must write them.
"""

ANSWER_BLOCK = """
--------------------------------------------------------------------------
{reference}  {section}
{question}

{answer}

Sources: {sources}
Confidence: {confidence:.2f} (computed)
"""

TODO_BLOCK = """
--------------------------------------------------------------------------
{reference}  {section}
{question}

*** SME TODO — NOT ANSWERED ***
Reason ({trigger}): {reason}
{detail}Assigned to: {owner}
{draft}"""


@dataclass
class TemplateAssembler:
    """The controller's `Assembler`. Deterministic, fixed template, no model."""

    out_dir: Path = field(default_factory=lambda: DEFAULT_OUT_DIR)
    #: question_id -> capability id, where the domain maps one. Supplied by the
    #: caller because capability routing is corpus knowledge, not template
    #: knowledge.
    capability_for_question: dict[str, str] = field(default_factory=dict)
    #: capability id -> (sme_id, sme_name), resolved through
    #: `get_sme_for_capability` before assembly.
    sme_for_capability: dict[str, tuple[str, str]] = field(default_factory=dict)

    def assemble(
        self,
        *,
        run_id: str,
        document: RFPDocument,
        questions: list[ExtractedQuestion],
        answers: dict[str, DraftedAnswer],
        escalations: EscalationsRecord,
    ) -> dict[str, str]:
        """Write both artifacts. Returns a mapping of artifact name to path."""
        enriched = self._with_owners(escalations)
        by_question = {record.question_id: record for record in enriched.escalations}

        body = [
            TEMPLATE_HEADER.format(
                customer=document.customer_name,
                source=document.source_filename,
                run_id=run_id,
            )
        ]
        for question in questions:
            answer = answers.get(question.id)
            if answer is not None and not answer.needs_sme_review:
                body.append(_render_answer(question, answer))
            else:
                body.append(_render_todo(question, by_question.get(question.id)))

        directory = self.out_dir / run_id
        directory.mkdir(parents=True, exist_ok=True)

        response_path = directory / "response.txt"
        response_path.write_text("".join(body), encoding="utf-8")

        escalations_path = directory / "escalations.json"
        escalations_path.write_text(
            json.dumps(enriched.model_dump(mode="json"), indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

        logger.info(
            "run %s: wrote %d answer(s) and %d escalation(s)",
            run_id,
            sum(1 for q in questions if q.id in answers and not answers[q.id].needs_sme_review),
            len(enriched.escalations),
        )
        return {
            "response_docx": str(response_path),
            "escalations_json": str(escalations_path),
        }

    def _with_owners(self, escalations: EscalationsRecord) -> EscalationsRecord:
        """Attach the SME each escalation routes to, where a capability maps."""
        resolved: list[EscalationRecord] = []
        for record in escalations.escalations:
            capability = self.capability_for_question.get(record.question_id)
            owner = self.sme_for_capability.get(capability or "")
            if owner is None:
                resolved.append(record.model_copy(update={"capability_id": capability}))
                continue
            sme_id, sme_name = owner
            resolved.append(
                record.model_copy(
                    update={
                        "capability_id": capability,
                        "sme_id": sme_id,
                        "sme_name": sme_name,
                    }
                )
            )
        return escalations.model_copy(update={"escalations": resolved})


def _render_answer(question: ExtractedQuestion, answer: DraftedAnswer) -> str:
    return ANSWER_BLOCK.format(
        reference=question.printed_number or f"[unnumbered #{question.order + 1}]",
        section=question.section,
        question=question.text,
        answer=answer.answer_text,
        sources=", ".join(answer.source_ids),
        confidence=answer.confidence,
    )


def _render_todo(question: ExtractedQuestion, record: EscalationRecord | None) -> str:
    """A TODO block. Never blank (build prompt §12).

    `record` is None only if a question produced neither an answer nor an
    escalation, which the controller does not do — but rendering a block that
    says so is better than a KeyError during assembly, because the failure it
    would describe is real and the reviewer needs to see it.
    """
    if record is None:
        return TODO_BLOCK.format(
            reference=question.printed_number or f"[unnumbered #{question.order + 1}]",
            section=question.section,
            question=question.text,
            trigger="unknown",
            reason=(
                "this question produced neither an answer nor an escalation record; "
                "treat it as unanswered and report the run"
            ),
            detail="",
            owner="unassigned — no capability mapped",
            draft="",
        )

    owner = (
        f"{record.sme_name} ({record.sme_id})"
        if record.sme_name
        else "unassigned — no capability mapped for this question"
    )
    draft = (
        f"\nDraft for review (NOT part of the response):\n{record.draft_text}\n"
        if record.draft_text
        else ""
    )
    return TODO_BLOCK.format(
        reference=record.reference,
        section=question.section,
        question=question.text,
        trigger=record.trigger.value,
        reason=record.reason,
        detail=f"Found: {record.detail}\n" if record.detail else "",
        owner=owner,
        draft=draft,
    )
