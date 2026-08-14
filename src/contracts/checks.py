"""Deterministic check results: compliance and entity grounding.

Neither of these is produced by a model. The compliance checker is plain code
(build prompt §12, agent #6 is explicitly "NO — deterministic") and entity
resolution is a graph lookup.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, model_validator

from src.contracts.enums import EntityType, EscalationTrigger


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


class GuardrailVerdict(BaseModel):
    """The deterministic post-processors' verdict on one draft.

    Every guardrail in this system is code, and every one of them can only do
    one of two things: let the answer through, or send it to a human. None of
    them edits a draft, and none of them lowers a score — an answer that trips a
    guardrail is not a worse answer, it is one a person has to look at.

    `trigger` is required on a failure so the escalation names WHICH rule fired,
    which is what the adversarial evals assert on: each planted trap has a
    required trigger, and checking only that something escalated would pass a
    system escalating for the wrong reason.
    """

    model_config = ConfigDict(extra="forbid")

    passed: bool
    trigger: EscalationTrigger | None = None
    reason: str | None = None
    #: The specific finding — the matched term, the unresolved entity, the
    #: injection pattern NAME. Trap GQ-004 asserts the pattern is named.
    detail: str | None = None

    @model_validator(mode="after")
    def _a_failure_says_which_rule_and_why(self) -> GuardrailVerdict:
        if self.passed:
            if self.trigger is not None or self.reason is not None:
                raise ValueError(
                    f"passed=True but a trigger/reason is set ({self.trigger}, {self.reason!r}); "
                    "a guardrail that passed has nothing to escalate"
                )
            return self
        if self.trigger is None:
            raise ValueError("passed=False requires a trigger naming which guardrail fired")
        if not (self.reason or "").strip():
            raise ValueError("passed=False requires a non-empty reason")
        return self


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
