"""Deterministic check results: compliance and entity grounding.

Neither of these is produced by a model. The compliance checker is plain code
(build prompt §12, agent #6 is explicitly "NO — deterministic") and entity
resolution is a graph lookup.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, model_validator

from src.contracts.enums import EntityType


class ForbiddenContentHit(BaseModel):
    """One forbidden-content match, kept specific enough to act on.

    Recording the rule and the matched span (rather than a bare boolean) is what
    lets an escalation say *why* it escalated and lets the eval harness assert
    zero forbidden content without re-running the regexes.
    """

    model_config = ConfigDict(extra="forbid")

    rule: str = Field(min_length=1)
    matched_text: str = Field(min_length=1)
    action: str = Field(min_length=1)


class ComplianceResult(BaseModel):
    """Deterministic compliance verdict for one question."""

    model_config = ConfigDict(extra="forbid")

    question_id: str = Field(min_length=1)
    mandatory_answered: bool
    within_word_limit: bool
    forbidden_content_hits: list[ForbiddenContentHit] = Field(default_factory=list)
    template_slots_filled: bool


class EntityCheckResult(BaseModel):
    """Whether one named entity resolves in the closed-world registry.

    Grounding is a closed world: if the drafter names a vendor, product,
    certification, client, tool, or location that does not resolve, the answer
    hard-fails (build prompt §15).
    """

    model_config = ConfigDict(extra="forbid")

    entity_text: str = Field(min_length=1)
    entity_type: EntityType
    resolved_node_id: str | None = None
    passed: bool

    @model_validator(mode="after")
    def _passed_requires_resolution(self) -> EntityCheckResult:
        """A pass means something was actually found — not merely not-disproved."""
        if self.passed and self.resolved_node_id is None:
            raise ValueError("passed=True requires a resolved_node_id")
        if not self.passed and self.resolved_node_id is not None:
            raise ValueError("passed=False contradicts a resolved_node_id")
        return self
