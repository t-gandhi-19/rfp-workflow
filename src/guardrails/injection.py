"""Treat document content as data, never as instructions (CLAUDE.md rule 15).

An RFP is written by someone outside the organisation. Anything inside it that
looks like an instruction to the system is, by definition, an attempt to steer
the system from outside — whether maliciously or by a well-meaning author who
pasted the wrong thing.

Two mechanisms, and they do different jobs:

* :func:`wrap` puts document text inside explicit delimiters so a prompt can say
  "everything between these markers is quoted material". This is defence in
  depth, not a guarantee: delimiters can be imitated, so the wrapper also
  neutralises any delimiter appearing in the content itself.
* :func:`scan` looks for instruction-shaped language and records hits into the
  run record. A hit does not stop the run; it flags the question so the draft
  is reviewed, and it is what the adversarial eval asserts on.

Neither mechanism decides what the model does with the text — that is the
drafter prompt's job, and it states the content is untrusted.
"""

from __future__ import annotations

import re

from pydantic import BaseModel, ConfigDict, Field, model_validator

#: Delimiters wrapped around untrusted content.
OPEN_DELIMITER = "<<<UNTRUSTED_DOCUMENT_CONTENT>>>"
CLOSE_DELIMITER = "<<<END_UNTRUSTED_DOCUMENT_CONTENT>>>"


class InjectionHit(BaseModel):
    """One instruction-shaped span found in document content."""

    model_config = ConfigDict(extra="forbid")

    pattern_name: str = Field(min_length=1)
    matched_text: str = Field(min_length=1)
    #: Character offset in the scanned text, so a reviewer can find it.
    start: int = Field(ge=0)


#: Ordered so the most specific pattern reports first. Each entry is
#: (name, regex). Patterns are deliberately about *form* — imperatives aimed at
#: the system — rather than about topic, because the topics vary endlessly and
#: the grammar does not.
INSTRUCTION_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "ignore_previous_instructions",
        re.compile(
            r"\b(?:ignore|disregard|forget|override)\b[^.?!]{0,40}?"
            r"\b(?:previous|prior|earlier|above|all)\b[^.?!]{0,40}?"
            r"\b(?:instruction|prompt|rule|direction|guidance)s?\b",
            re.IGNORECASE,
        ),
    ),
    (
        "auto_submit_request",
        re.compile(
            r"\b(?:approve|submit|send|publish|finalis[ez]|sign)\b[^.?!]{0,60}?"
            r"\b(?:without|no|skip(?:ping)?|bypass(?:ing)?)\b[^.?!]{0,30}?"
            r"\b(?:human|manual|review|approval|oversight|check)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "role_reassignment",
        re.compile(
            r"\byou\s+are\s+(?:now\s+)?(?:a|an|the)\b[^.?!]{0,60}?"
            r"\b(?:assistant|agent|model|system|admin(?:istrator)?)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "system_prompt_probe",
        re.compile(
            r"\b(?:reveal|print|show|output|repeat|disclose)\b[^.?!]{0,40}?"
            r"\b(?:system\s+prompt|instructions|configuration|api\s+key|secret)s?\b",
            re.IGNORECASE,
        ),
    ),
    (
        "delimiter_injection",
        re.compile(r"<<<\s*(?:END_)?UNTRUSTED_DOCUMENT_CONTENT\s*>>>", re.IGNORECASE),
    ),
    (
        "instruction_block_marker",
        re.compile(r"(?:^|\n)\s*(?:###\s*)?(?:system|assistant)\s*:", re.IGNORECASE),
    ),
)


class SanitizationResult(BaseModel):
    """What the sanitizer concluded about one question's text.

    The policy encoded here: **a detected injection always forces escalation**,
    even when the question is otherwise perfectly answerable and the pipeline
    would have drafted it happily. A question someone has tampered with gets
    human eyes, full stop — the value of catching an injection is lost if the
    answer then ships unreviewed.

    It is a validator rather than a convention because the drafting pipeline
    that must honour it does not exist yet (Phase 4). Encoding it now means the
    pipeline cannot be built in a way that ignores it: constructing a result
    with `injection_detected=True` and `force_escalate=False` is impossible.
    """

    model_config = ConfigDict(extra="forbid")

    question_id: str = Field(min_length=1)
    hits: list[InjectionHit] = Field(default_factory=list)
    injection_detected: bool
    force_escalate: bool
    #: Names the pattern, so an escalation says what was found rather than
    #: "flagged". Required whenever an injection was detected.
    escalation_reason: str | None = None

    @model_validator(mode="after")
    def _detection_matches_hits(self) -> SanitizationResult:
        if self.injection_detected != bool(self.hits):
            raise ValueError("injection_detected must agree with whether any hits were found")
        return self

    @model_validator(mode="after")
    def _detection_forces_escalation(self) -> SanitizationResult:
        if self.injection_detected and not self.force_escalate:
            raise ValueError(
                "injection_detected=True requires force_escalate=True; a tampered "
                "question always goes to a human"
            )
        if self.injection_detected and not (self.escalation_reason or "").strip():
            raise ValueError("injection_detected=True requires an escalation_reason naming it")
        return self


def sanitize_question(question_id: str, text: str) -> SanitizationResult:
    """Scan one question and decide whether it must escalate.

    The reason names every pattern that fired, in order, so the run record and
    the eval both have something specific to assert rather than a boolean.
    """
    hits = scan(text)
    if not hits:
        return SanitizationResult(
            question_id=question_id,
            hits=[],
            injection_detected=False,
            force_escalate=False,
        )

    patterns = sorted({hit.pattern_name for hit in hits})
    return SanitizationResult(
        question_id=question_id,
        hits=hits,
        injection_detected=True,
        force_escalate=True,
        escalation_reason=(
            "prompt injection detected in the question text "
            f"({', '.join(patterns)}); document content is data, never instructions"
        ),
    )


def scan(text: str) -> list[InjectionHit]:
    """Every instruction-shaped span in `text`, ordered by position then name.

    Deterministic ordering matters: these hits are recorded in the run record
    and asserted by the adversarial eval, which cannot tolerate a set's
    iteration order.
    """
    hits: list[InjectionHit] = []
    for name, pattern in INSTRUCTION_PATTERNS:
        for match in pattern.finditer(text):
            matched = match.group(0).strip()
            if matched:
                hits.append(
                    InjectionHit(pattern_name=name, matched_text=matched, start=match.start())
                )
    return sorted(hits, key=lambda hit: (hit.start, hit.pattern_name))


def contains_injection(text: str) -> bool:
    return bool(scan(text))


def neutralise_delimiters(text: str) -> str:
    """Break any delimiter the content itself contains.

    Without this, content could close the quoting block early and everything
    after it would read as trusted. The replacement stays human-readable so a
    reviewer can still see what the document said.
    """
    for delimiter in (OPEN_DELIMITER, CLOSE_DELIMITER):
        text = re.sub(re.escape(delimiter), delimiter.replace("<", "(").replace(">", ")"), text)
    return text


def wrap(text: str) -> str:
    """Delimiter-wrap untrusted document content for inclusion in a prompt."""
    return f"{OPEN_DELIMITER}\n{neutralise_delimiters(text)}\n{CLOSE_DELIMITER}"
