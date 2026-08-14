"""Turn extracted document text into validated questions (build prompt §9).

Entirely deterministic. The output is a `list[ExtractedQuestion]` that has
already passed contract validation before anything else in the system runs, so
a malformed document fails here rather than three stages later.

Both the PDF and the DOCX renditions of a document reach this function as plain
text, and the parser is line-oriented, so the same questions come out of both.
That is what the parity test checks — and the reason PDF line wrapping is
handled by accumulating continuation lines rather than treating every line as a
new item.
"""

from __future__ import annotations

import re

from src.contracts import ExtractedQuestion, QuestionType

#: "Section Compliance"
SECTION_RE = re.compile(r"^\s*Section\s+(?P<name>\S.*?)\s*$")

#: "3.2 How do you ... [Mandatory; maximum 300 words]"
QUESTION_RE = re.compile(r"^\s*(?P<number>\d+\.\d+)\s+(?P<rest>\S.*)$")

#: Trailing "[Mandatory; maximum 300 words]"
ANNOTATION_RE = re.compile(r"\s*\[(?P<body>[^\]]*)\]\s*$")

WORD_LIMIT_RE = re.compile(r"maximum\s+(?P<limit>\d+)\s+words", re.IGNORECASE)

#: The document states a section, never a question type. Deriving type from
#: section keeps extraction deterministic and means the answer key and the
#: extractor cannot disagree about a field the document does not contain.
SECTION_TO_TYPE: dict[str, QuestionType] = {
    "company": QuestionType.COMPANY_INFO,
    "technical approach": QuestionType.TECHNICAL,
    "compliance": QuestionType.COMPLIANCE,
    "delivery": QuestionType.COMMERCIAL,
}

DEFAULT_TYPE = QuestionType.TECHNICAL


def question_type_for_section(section: str) -> QuestionType:
    return SECTION_TO_TYPE.get(section.strip().casefold(), DEFAULT_TYPE)


def normalise(text: str) -> str:
    """Strip numbering and the bracketed annotation; collapse whitespace.

    The raw text is kept alongside on the contract, so a citation can always be
    traced back to what the document literally said.
    """
    without_annotation = ANNOTATION_RE.sub("", text)
    without_number = re.sub(r"^\s*\d+\.\d+\s+", "", without_annotation)
    return " ".join(without_number.split())


def parse_annotation(text: str) -> tuple[bool, int | None]:
    """Read `mandatory` and `word_limit` out of the trailing bracket."""
    match = ANNOTATION_RE.search(text)
    if match is None:
        return False, None
    body = match.group("body")
    mandatory = "mandatory" in body.casefold()
    limit_match = WORD_LIMIT_RE.search(body)
    word_limit = int(limit_match.group("limit")) if limit_match else None
    return mandatory, word_limit


def _flush(
    buffer: list[str],
    *,
    section: str,
    rfp_id: str,
    order: int,
) -> ExtractedQuestion | None:
    if not buffer:
        return None
    raw = " ".join(part.strip() for part in buffer if part.strip())
    raw = " ".join(raw.split())
    number_match = QUESTION_RE.match(raw)
    if number_match is None:
        return None
    mandatory, word_limit = parse_annotation(raw)
    normalized = normalise(raw)
    if not normalized:
        return None
    return ExtractedQuestion(
        id=f"{rfp_id}-Q{order + 1:03d}",
        rfp_id=rfp_id,
        text=raw,
        normalized_text=normalized,
        section=section,
        question_type=question_type_for_section(section),
        word_limit=word_limit,
        mandatory=mandatory,
        order=order,
        # Kept rather than discarded. Normalisation strips it from the text, and
        # it used to survive only as a prefix a caller had to re-parse — which
        # made every consumer reimplement the same regex.
        printed_number=number_match.group("number"),
    )


def parse_questions(text: str, *, rfp_id: str) -> list[ExtractedQuestion]:
    """Every numbered question in document order.

    Lines that are neither a section heading nor the start of a numbered
    question are treated as continuations of the current question, which is how
    a PDF's wrapped lines are reassembled into the sentence the author wrote.
    """
    questions: list[ExtractedQuestion] = []
    section = ""
    buffer: list[str] = []

    def flush() -> None:
        question = _flush(buffer, section=section, rfp_id=rfp_id, order=len(questions))
        if question is not None:
            questions.append(question)
        buffer.clear()

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        section_match = SECTION_RE.match(line)
        if section_match:
            flush()
            section = section_match.group("name")
            continue

        if QUESTION_RE.match(line):
            flush()
            buffer.append(line)
            continue

        if buffer:
            buffer.append(line)

    flush()
    return questions
