"""The domain policy says what it means about legal-adjacent topics.

`warranty` sits in the legal forbidden-term list while warranty questions are
also expected to be answered. Read carelessly those look contradictory, and a
future change could "resolve" the contradiction in the wrong direction — by
dropping warranty from the term list, or by refusing the question outright.

The intent, now asserted: pricing is never drafted; warranty is drafted and then
always escalated. These tests pin the configuration that expresses it.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DOMAIN_PATH = REPO_ROOT / "config" / "domains" / "cloud_migration.yaml"


@pytest.fixture(scope="module")
def domain() -> dict[str, Any]:
    with DOMAIN_PATH.open(encoding="utf-8") as handle:
        return dict(yaml.safe_load(handle))


class TestForbiddenContentPolicy:
    def test_pricing_and_legal_are_distinct_rules(self, domain: dict[str, Any]) -> None:
        forbidden = domain["compliance"]["forbidden_content"]
        assert "pricing" in forbidden
        assert "legal" in forbidden

    def test_warranty_is_a_legal_term(self, domain: dict[str, Any]) -> None:
        """Removing it would let a warranty commitment ship without SME review."""
        terms = {
            term.casefold() for term in domain["compliance"]["forbidden_content"]["legal"]["terms"]
        }
        assert "warranty" in terms

    def test_both_rules_escalate_rather_than_silently_dropping(
        self, domain: dict[str, Any]
    ) -> None:
        forbidden = domain["compliance"]["forbidden_content"]
        assert forbidden["pricing"]["action"] == "hard_block_escalate"
        assert forbidden["legal"]["action"] == "hard_block_escalate"

    def test_the_commercial_type_documents_the_distinction(self, domain: dict[str, Any]) -> None:
        """The description must state that warranty is drafted-then-escalated.

        Without this the config reads as though warranty questions are refused,
        which is not what the pipeline does.
        """
        commercial = next(t for t in domain["question_types"] if t["key"] == "commercial")
        description = commercial["description"].casefold()
        assert "warranty" in description
        assert "escalat" in description, "the description must say warranty escalates"
        assert "pricing" in description


class TestPricingPatterns:
    """The regexes are the actual guard; a broken one fails open."""

    @pytest.fixture(scope="class")
    def patterns(self, domain: dict[str, Any]) -> list[re.Pattern[str]]:
        raw = domain["compliance"]["forbidden_content"]["pricing"]["patterns"]
        return [re.compile(pattern) for pattern in raw]

    @pytest.mark.parametrize(
        "text",
        [
            "The total programme price is GBP 450000.",
            "Day rates start at $1,200 per engineer.",
            "We charge 95 EUR per workload migrated.",
            "Costs are billed per user, 45 monthly.",
        ],
    )
    def test_pricing_shapes_are_matched(self, patterns: list[re.Pattern[str]], text: str) -> None:
        assert any(pattern.search(text) for pattern in patterns), text

    @pytest.mark.parametrize(
        "text",
        [
            "We migrated 240 workloads across four waves.",
            "The warranty period begins at each workload's own cutover.",
            "Governance operates at three levels with a monthly cadence.",
        ],
    )
    def test_ordinary_prose_is_not_matched(
        self, patterns: list[re.Pattern[str]], text: str
    ) -> None:
        assert not any(pattern.search(text) for pattern in patterns), text
