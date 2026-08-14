"""Closed vocabularies shared by every contract.

These are `str` enums so they serialize to readable JSON at the MCP and
write-api boundaries and store as plain text in Postgres.
"""

from __future__ import annotations

from enum import StrEnum


class DocumentFormat(StrEnum):
    """Intake formats. One channel (file upload), two renditions."""

    PDF = "pdf"
    DOCX = "docx"


class QuestionType(StrEnum):
    """Question taxonomy for the cloud-migration domain (build prompt §6)."""

    TECHNICAL = "technical"
    COMMERCIAL = "commercial"
    COMPLIANCE = "compliance"
    COMPANY_INFO = "company_info"


class RetrievalStatus(StrEnum):
    """Whether retrieval cleared the configured match floor.

    NO_MATCH is not a failure — it is the correct, tested outcome for a question
    the corpus does not cover, and it obliges the drafter to escalate rather
    than invent an answer.
    """

    MATCHED = "MATCHED"
    NO_MATCH = "NO_MATCH"


class Outcome(StrEnum):
    """Commercial outcome of the bid an answer came from.

    Feeds the graph multiplier: a won answer outranks a lost one of equal
    embedding similarity.
    """

    WON = "won"
    LOST = "lost"
    UNKNOWN = "unknown"


class EntityType(StrEnum):
    """Nameable entity classes in the closed-world grounding registry.

    Anything the drafter names must resolve to one of these, or the answer
    hard-fails (build prompt §15).
    """

    VENDOR = "vendor"
    PRODUCT = "product"
    CERTIFICATION = "certification"
    CLIENT = "client"
    TOOL = "tool"
    LOCATION = "location"


class RunStage(StrEnum):
    """Deterministic controller stages (build prompt §13).

    Order is fixed: created -> parsing -> extracting -> drafting -> compliance
    -> assembling -> complete, with HALTED reachable from any stage.
    """

    CREATED = "created"
    PARSING = "parsing"
    EXTRACTING = "extracting"
    DRAFTING = "drafting"
    COMPLIANCE = "compliance"
    ASSEMBLING = "assembling"
    COMPLETE = "complete"
    HALTED = "halted"


class QuestionStatus(StrEnum):
    """Per-question lifecycle.

    A single question failing never halts the run; it becomes ESCALATED (with
    the error as its reason) or FAILED, and the run continues.
    """

    PENDING = "pending"
    DRAFTED = "drafted"
    CRITIQUED = "critiqued"
    ESCALATED = "escalated"
    FAILED = "failed"
    COMPLETE = "complete"


class HaltReason(StrEnum):
    """The only three reasons a whole run may halt (build prompt §12)."""

    DOMAIN_MISMATCH = "domain_mismatch"
    BUDGET = "budget"
    INFRA = "infra"


class SelectionOutcome(StrEnum):
    """What became of one scored candidate at the retriever's boundary.

    Only two values, because only two are REACHABLE there. Superseded and
    confidential records are excluded inside the Cypher (Phase 2 amendment A)
    and again in `gather_candidates`, so a candidate that reaches selection has
    already survived both — an enum member for either would name an outcome no
    code path can produce, which reads as coverage and is not.
    """

    SELECTED = "selected"
    BELOW_RELEVANCE_FLOOR = "below_relevance_floor"


class RankingBasis(StrEnum):
    """How an ordering was arrived at, for the audit trail.

    A rerank failure degrades one question rather than the run: the weight is
    redistributed and the ordering stands on calibrated similarity alone. That
    is a materially weaker ranking, and a citation nobody can tell apart from a
    reranked one hides it.
    """

    SIMILARITY_AND_RERANK = "similarity_and_rerank"
    CALIBRATED_SIMILARITY_ONLY = "calibrated_similarity_only"


class EscalationTrigger(StrEnum):
    """Why one question went to an SME instead of into the response.

    A closed vocabulary rather than free text, because the adversarial eval
    category asserts on these: the planted traps each have a REQUIRED trigger,
    and "escalated for some reason" would pass a test that means to check the
    system escalated for the RIGHT one. Free text also cannot be counted, and
    the operational report counts escalations by cause.
    """

    #: Retrieval cleared no candidate over the relevance floor.
    NO_MATCH = "no_match"
    #: Computed confidence fell below the configured floor.
    LOW_CONFIDENCE = "low_confidence"
    #: The draft asserted something no cited source supports.
    UNSUPPORTED_CLAIMS = "unsupported_claims"
    #: A claim carried no source id at all.
    SOURCE_COVERAGE = "source_coverage"
    #: Pricing is a HARD BLOCK — never drafted, always escalated (trap 4.2).
    PRICING_BLOCKED = "pricing_blocked"
    #: SLA / indemnity / penalty / warranty / liability / liquidated damages.
    LEGAL_TERM = "legal_term"
    #: A named vendor, product, certification, client, tool or location that
    #: does not resolve in the closed-world registry.
    ENTITY_UNRESOLVED = "entity_unresolved"
    #: A number, date or quantity that no cited source states.
    NUMERIC_INCONSISTENT = "numeric_inconsistent"
    #: Content belonging to a customer other than the one being answered.
    CROSS_CUSTOMER_CONTENT = "cross_customer_content"
    #: Source material not cleared for use outside its original engagement.
    NOT_PUBLICLY_USABLE = "not_publicly_usable"
    #: Instructions embedded in the document (trap GQ-004). Never followed.
    PROMPT_INJECTION = "prompt_injection"
    #: `output_pydantic` validation failed, the one retry was spent, and the
    #: question escalates rather than the run failing.
    VALIDATION_FAILED = "validation_failed"
    #: The question's own stage raised. One question failing never halts a run.
    STAGE_ERROR = "stage_error"
