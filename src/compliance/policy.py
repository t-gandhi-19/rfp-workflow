"""Typed access to `config/domains/<key>.yaml`.

The policy has been readable since Phase 1 and consumed by nothing until now.
Loading it through a contract rather than as a dict is what makes the gap
between "written down" and "enforced" visible: a rule the checker forgets to
read is a rule that silently does not apply, and `extra="forbid"` means a rule
added to the YAML that nothing here models fails at load instead.

PRICING AND LEGAL ARE DIFFERENT RULES WITH THE SAME ACTION, and the YAML says so
at length because they are easy to conflate. Pricing is never drafted — the
question escalates unanswered. Legal topics ARE drafted normally and the draft
is then always escalated for review. Both end in an SME's queue; only one of
them produces an answer for the SME to read.
"""

from __future__ import annotations

import re
from functools import lru_cache

from pydantic import BaseModel, ConfigDict, Field

from src.contracts.enums import EntityType
from src.contracts.thresholds import load_config_yaml


class PricingRule(BaseModel):
    """Currency-shaped tokens. Any match means the draft is asserting commercials."""

    model_config = ConfigDict(extra="forbid")

    action: str = Field(min_length=1)
    patterns: list[str] = Field(min_length=1)
    reason: str = Field(min_length=1)

    @property
    def compiled(self) -> list[re.Pattern[str]]:
        return [re.compile(pattern) for pattern in self.patterns]


class LegalRule(BaseModel):
    """Contractual commitments. Drafted, then always escalated."""

    model_config = ConfigDict(extra="forbid")

    action: str = Field(min_length=1)
    terms: list[str] = Field(min_length=1)
    reason: str = Field(min_length=1)

    @property
    def compiled(self) -> list[tuple[str, re.Pattern[str]]]:
        """Word-boundary matches, so 'penalty' does not fire on 'penalise' and
        'SLA' does not fire inside 'translate'.

        Sorted longest-first so a hit reports the most specific term it matched:
        text containing "service level agreement" should escalate naming that,
        not naming "SLA" — which it does not contain.
        """
        ordered = sorted(self.terms, key=len, reverse=True)
        return [(term, re.compile(rf"\b{re.escape(term)}\b", re.IGNORECASE)) for term in ordered]


class ForbiddenContent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    pricing: PricingRule
    legal: LegalRule


class PublicUsability(BaseModel):
    model_config = ConfigDict(extra="forbid")

    require_publicly_usable_flag: bool


class ComplianceDefaults(BaseModel):
    model_config = ConfigDict(extra="forbid")

    word_limit: int | None = None
    mandatory: bool = False


class CompliancePolicy(BaseModel):
    model_config = ConfigDict(extra="forbid")

    forbidden_content: ForbiddenContent
    entity_types_checked: list[EntityType] = Field(min_length=1)
    public_usability: PublicUsability
    defaults: ComplianceDefaults


class RetrievalFilters(BaseModel):
    model_config = ConfigDict(extra="forbid")

    exclude_superseded: bool
    exclude_confidential_for_other_customers: bool
    restrict_to_domain: bool


class RetrievalPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid")

    filters: RetrievalFilters
    #: Doubles as the SME routing key (SME OWNS Capability) and the
    #: coverage-gap axis.
    capabilities: list[str] = Field(min_length=1)


class QuestionTypeDescription(BaseModel):
    model_config = ConfigDict(extra="forbid")

    key: str = Field(min_length=1)
    description: str = Field(min_length=1)


class DomainPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid")

    version: int
    key: str = Field(min_length=1)
    display_name: str = Field(min_length=1)
    description: str = Field(min_length=1)
    question_types: list[QuestionTypeDescription] = Field(min_length=1)
    retrieval: RetrievalPolicy
    compliance: CompliancePolicy


DEFAULT_DOMAIN = "cloud_migration"


@lru_cache(maxsize=4)
def domain_policy(key: str = DEFAULT_DOMAIN) -> DomainPolicy:
    """Parsed, validated domain config. Cached after first read."""
    policy = DomainPolicy.model_validate(load_config_yaml(f"domains/{key}.yaml"))
    if policy.key != key:
        raise ValueError(
            f"config/domains/{key}.yaml declares key '{policy.key}'. The filename and "
            f"the key must agree, or triage compares against the wrong domain."
        )
    return policy


def reload_domain_policy() -> None:
    """Drop the cache so the next read picks up a new ``RFP_CONFIG_DIR``."""
    domain_policy.cache_clear()
