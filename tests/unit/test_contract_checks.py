"""Check invariants: entity grounding is a closed world, compliance is explicit."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.contracts import (
    ComplianceResult,
    EntityCheckResult,
    EntityType,
    ForbiddenContentHit,
)


class TestEntityGrounding:
    """A pass means something was found — not merely that nothing disproved it."""

    def test_accepts_a_resolved_entity(self) -> None:
        check = EntityCheckResult(
            entity_text="CloudNova Partners",
            entity_type=EntityType.VENDOR,
            resolved_node_id="VND-0001",
            passed=True,
        )
        assert check.resolved_node_id == "VND-0001"

    def test_accepts_an_unresolved_entity_as_a_failure(self) -> None:
        check = EntityCheckResult(
            entity_text="Quantum Migration Labs",
            entity_type=EntityType.VENDOR,
            passed=False,
        )
        assert check.resolved_node_id is None

    def test_rejects_a_pass_with_nothing_resolved(self) -> None:
        with pytest.raises(ValidationError, match="requires a resolved_node_id"):
            EntityCheckResult(
                entity_text="Invented Systems Inc",
                entity_type=EntityType.VENDOR,
                passed=True,
            )

    def test_rejects_a_failure_that_resolved(self) -> None:
        with pytest.raises(ValidationError, match="contradicts"):
            EntityCheckResult(
                entity_text="CloudNova Partners",
                entity_type=EntityType.VENDOR,
                resolved_node_id="VND-0001",
                passed=False,
            )

    @pytest.mark.parametrize("entity_type", list(EntityType))
    def test_covers_every_checked_entity_class(self, entity_type: EntityType) -> None:
        check = EntityCheckResult(entity_text="something", entity_type=entity_type, passed=False)
        assert check.entity_type is entity_type


class TestComplianceResult:
    def test_accepts_a_clean_result(self) -> None:
        result = ComplianceResult(
            question_id="q-001",
            mandatory_answered=True,
            within_word_limit=True,
            template_slots_filled=True,
        )
        assert result.forbidden_content_hits == []

    def test_records_why_content_was_forbidden(self) -> None:
        """A bare boolean would not tell an escalation what to say."""
        result = ComplianceResult(
            question_id="q-014",
            mandatory_answered=False,
            within_word_limit=True,
            forbidden_content_hits=[
                ForbiddenContentHit(
                    rule="pricing",
                    matched_text="$45 per workload",
                    action="hard_block_escalate",
                ),
                ForbiddenContentHit(
                    rule="legal",
                    matched_text="unlimited liability",
                    action="hard_block_escalate",
                ),
            ],
            template_slots_filled=False,
        )
        assert {hit.rule for hit in result.forbidden_content_hits} == {"pricing", "legal"}

    def test_rejects_an_empty_matched_span(self) -> None:
        with pytest.raises(ValidationError):
            ForbiddenContentHit(rule="pricing", matched_text="", action="hard_block_escalate")
