"""Deterministic compliance (build prompt §12, agent #6 is explicitly "NO").

Four questions, none of which a model is asked:

* Was a mandatory question answered?
* Is the answer within the word limit the DOCUMENT stated?
* Does it contain forbidden content — currency shapes, contractual terms?
* Are the template's slots filled?

WHAT COMPLIANCE IS NOT. It does not escalate anything and it does not change an
answer. It runs after the fan-out and reports what is true of the assembled
response, which is a different job from the guardrails in `src/guardrails/` —
those run per draft and can send a question to a human. Conflating them would
mean a compliance report that had already fixed what it was reporting on.

WORD COUNTING IS A DECISION, not an obvious operation. `str.split()` on
whitespace is what a reviewer counting by eye approximates, what Word's word
count approximates, and what an RFP's "maximum 300 words" means in practice.
Counting tokens or characters would be defensible and would not be the number
the customer is checking against.
"""

from __future__ import annotations

import re

from src.compliance.policy import DomainPolicy, domain_policy
from src.contracts import ComplianceResult, DraftedAnswer, ExtractedQuestion, ForbiddenContentHit

#: What "a word" means for a stated limit. See the module docstring.
_WORD = re.compile(r"\S+")

#: Placeholders the assembler's template leaves behind if a slot went unfilled.
#: Scanned for rather than assumed absent, because a template rendered with a
#: missing value produces a document that LOOKS complete and ships a literal
#: `{customer_name}` to a customer.
_UNFILLED_SLOT = re.compile(r"\{[a-z_]+\}|\bTBD\b|\bTODO\b|<<[^>]+>>", re.IGNORECASE)


def count_words(text: str) -> int:
    return len(_WORD.findall(text))


def scan_forbidden(text: str, *, policy: DomainPolicy | None = None) -> list[ForbiddenContentHit]:
    """Every forbidden-content match, pricing first then legal.

    Returns HITS rather than a boolean so an escalation can say what was found.
    A boolean would make the eval's "forbidden content: 0" assertion true of a
    system that found something and forgot which.
    """
    resolved = policy or domain_policy()
    forbidden = resolved.compliance.forbidden_content
    hits: list[ForbiddenContentHit] = []

    for pattern in forbidden.pricing.compiled:
        for match in pattern.finditer(text):
            matched = match.group(0).strip()
            if matched:
                hits.append(
                    ForbiddenContentHit(
                        rule="pricing",
                        matched_text=matched,
                        action=forbidden.pricing.action,
                    )
                )

    # Longest term first, and each SPAN reported once: "service level agreement"
    # contains no shorter term from the list, but a future list might, and two
    # hits for one span would double-count in the eval's zero-tolerance total.
    claimed: list[tuple[int, int]] = []
    for term, pattern in forbidden.legal.compiled:
        for match in pattern.finditer(text):
            span = match.span()
            if any(start <= span[0] and span[1] <= end for start, end in claimed):
                continue
            claimed.append(span)
            hits.append(
                ForbiddenContentHit(
                    rule=f"legal:{term}",
                    matched_text=match.group(0),
                    action=forbidden.legal.action,
                )
            )
    return hits


class DeterministicComplianceChecker:
    """The controller's `ComplianceChecker`. No model, by design."""

    def __init__(self, *, policy: DomainPolicy | None = None) -> None:
        self._policy = policy or domain_policy()

    def check(self, question: ExtractedQuestion, answer: DraftedAnswer | None) -> ComplianceResult:
        """One question's compliance verdict.

        `answer` is None for an escalated question, and that is a normal input
        rather than an error: a mandatory question that escalated is a
        compliance FACT the report has to state, and skipping it would make the
        mandatory-coverage number describe only the questions that went well.
        """
        if answer is None:
            return ComplianceResult(
                question_id=question.id,
                # An escalated mandatory question is not answered. The response
                # is not complete until a human writes it.
                mandatory_answered=not question.mandatory,
                # Vacuously true: there is no text to exceed a limit.
                within_word_limit=True,
                forbidden_content_hits=[],
                template_slots_filled=False,
            )

        limit = (
            question.word_limit
            if question.word_limit is not None
            else (self._policy.compliance.defaults.word_limit)
        )
        within_limit = limit is None or count_words(answer.answer_text) <= limit

        return ComplianceResult(
            question_id=question.id,
            mandatory_answered=True,
            within_word_limit=within_limit,
            forbidden_content_hits=scan_forbidden(answer.answer_text, policy=self._policy),
            template_slots_filled=not _UNFILLED_SLOT.search(answer.answer_text),
        )


def summarise(results: list[ComplianceResult]) -> dict[str, int]:
    """Counts the eval harness gates on.

    `mandatory_missing` counts questions the response is obliged to answer and
    does not — the number that must be zero before anything is sent.
    """
    return {
        "questions": len(results),
        "mandatory_missing": sum(1 for r in results if not r.mandatory_answered),
        "limit_violations": sum(1 for r in results if not r.within_word_limit),
        "forbidden_hits": sum(len(r.forbidden_content_hits) for r in results),
        "unfilled_slots": sum(1 for r in results if not r.template_slots_filled),
    }
