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
