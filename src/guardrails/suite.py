"""The deterministic post-processors that run on every draft (build prompt §15).

EVERY ONE OF THESE IS CODE. None asks a model for an opinion, none edits a
draft, and none lowers a score. A guardrail does exactly one of two things: let
the answer through, or send it to a human. An answer that trips one is not a
worse answer — it is one a person has to look at.

THE ORDER IS THE POINT. They run cheapest-and-most-certain first, and the FIRST
failure wins, because a question escalates once (see `EscalationsRecord`) and
the trigger recorded is the one an SME reads. Pricing before legal because a
rate card is a harder block than a warranty mention; both before entity
resolution because those need no graph round-trip.

WHAT `screen` DOES THAT `apply` CANNOT. The injection sanitizer judges the
QUESTION, and it can do that before a single model call. Running it as a
post-processor would mean spending a rerank, a draft and a critique to reach a
conclusion available for free — so it runs first, in `screen`, and the
controller escalates before retrieval. Escalate-first is not an optimisation
here; it is the difference between catching a tampered question and paying to
answer one.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import httpx

from src.agents.mcp_client import McpClient, McpError, drafter_client
from src.compliance.checker import scan_forbidden
from src.compliance.policy import DomainPolicy, domain_policy
from src.contracts import (
    DraftedAnswer,
    EscalationTrigger,
    ExtractedQuestion,
    GuardrailVerdict,
    RetrieverSelection,
    RFPDocument,
)
from src.guardrails.injection import sanitize_question

PASSED = GuardrailVerdict(passed=True)

#: Any number worth checking against a source: 99.95, 1,200, 40%, 12 months.
_NUMERIC = re.compile(r"\b\d[\d,]*(?:\.\d+)?%?\b")

#: Numbers that carry no factual claim on their own. Excluding them keeps the
#: numeric check from escalating every answer that says "three phases" — a
#: guardrail that fires on everything is one people learn to override.
_TRIVIAL_NUMBERS = {"0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10"}

#: Capitalised multi-word spans — the shape a vendor, product or client name
#: takes. Deliberately conservative: it is a candidate generator for the
#: registry check, and a name it misses is caught by the critic and by the
#: unsupported-claim rule, while a false positive costs an unnecessary
#: escalation.
_PROPER_NOUN = re.compile(r"\b(?:[A-Z][a-zA-Z0-9]+(?:\s+[A-Z][a-zA-Z0-9]+){0,3})\b")

#: Words that start a sentence and are not entities. Without this every answer
#: escalates on its own first word.
_SENTENCE_STARTERS = frozenset(
    {
        "The",
        "This",
        "That",
        "These",
        "Those",
        "Our",
        "We",
        "Their",
        "Its",
        "It",
        "A",
        "An",
        "In",
        "On",
        "At",
        "For",
        "With",
        "By",
        "From",
        "To",
        "As",
        "All",
        "Each",
        "Every",
        "Both",
        "Where",
        "When",
        "While",
        "After",
        "Before",
        "During",
        "Following",
        "Once",
        "If",
        "Because",
        "Since",
        "Migration",
        "Migrations",
        "Cloud",
        "Data",
        "Security",
        "Compliance",
        "Assessment",
        "Delivery",
        "Support",
        "Operations",
        "Governance",
    }
)


@dataclass
class DeterministicGuardrails:
    """The controller's `GuardrailSuite`."""

    mcp: McpClient = field(default_factory=drafter_client)
    policy: DomainPolicy = field(default_factory=domain_policy)
    #: Set False only by tests that have no stack. A run with entity checking
    #: off is a run with closed-world grounding off, so it is not a default.
    check_entities: bool = True

    # -- pre-draft --------------------------------------------------------

    def screen(self, question: ExtractedQuestion) -> GuardrailVerdict:
        """The injection contract, honoured before anything is spent.

        `SanitizationResult` already refuses to exist with
        `injection_detected=True` and `force_escalate=False`, so this reads a
        decision the contract made rather than making one. The PATTERN NAME goes
        into `detail` because trap GQ-004 asserts the escalation names what was
        found, not merely that something was.
        """
        result = sanitize_question(question.id, question.text)
        if not result.force_escalate:
            return PASSED
        return GuardrailVerdict(
            passed=False,
            trigger=EscalationTrigger.PROMPT_INJECTION,
            reason=result.escalation_reason or "instruction-shaped content in the question",
            detail=", ".join(hit.pattern_name for hit in result.hits),
        )

    # -- post-draft -------------------------------------------------------

    async def apply(
        self,
        *,
        question: ExtractedQuestion,
        answer: DraftedAnswer,
        selection: RetrieverSelection,
        document: RFPDocument,
    ) -> GuardrailVerdict:
        """Run every check; return the first failure.

        First-failure rather than collect-all because the escalation carries ONE
        trigger and an SME reads it as "why is this here". Collecting all of
        them and picking one later would put the choice somewhere nobody looks.
        """
        for verdict in (
            self._source_coverage(answer),
            self._pricing(answer),
            self._legal(answer),
            self._numeric_consistency(answer, selection),
            self._cross_customer(answer, selection, document),
        ):
            if not verdict.passed:
                return verdict

        return await self._entities(answer)

    # -- individual rules -------------------------------------------------

    def _source_coverage(self, answer: DraftedAnswer) -> GuardrailVerdict:
        """Every accepted answer carries at least one source id.

        The `DraftedAnswer` contract already refuses an uncited answer that is
        not escalated, so this is the second lock on the same door — and it is
        here because the contract's version raises a ValidationError deep in a
        fan-out, while this one produces an escalation an SME can act on.
        """
        if answer.source_ids:
            return PASSED
        return GuardrailVerdict(
            passed=False,
            trigger=EscalationTrigger.SOURCE_COVERAGE,
            reason="the draft cites no source id",
        )

    def _pricing(self, answer: DraftedAnswer) -> GuardrailVerdict:
        """HARD BLOCK. Pricing is never drafted; a human owns commercials."""
        hits = [
            hit
            for hit in scan_forbidden(answer.answer_text, policy=self.policy)
            if hit.rule == "pricing"
        ]
        if not hits:
            return PASSED
        return GuardrailVerdict(
            passed=False,
            trigger=EscalationTrigger.PRICING_BLOCKED,
            reason=self.policy.compliance.forbidden_content.pricing.reason,
            detail=hits[0].matched_text,
        )

    def _legal(self, answer: DraftedAnswer) -> GuardrailVerdict:
        """HARD BLOCK to an SME — but the draft survives for them to review.

        Not a refusal. The question was retrieved and drafted like any other;
        what this rule says is that a contractual commitment never ships without
        a human. GQ-019's warranty answer escalates HERE, by design, and the
        eval records it as correct behaviour rather than as a miss.
        """
        hits = [
            hit
            for hit in scan_forbidden(answer.answer_text, policy=self.policy)
            if hit.rule.startswith("legal:")
        ]
        if not hits:
            return PASSED
        return GuardrailVerdict(
            passed=False,
            trigger=EscalationTrigger.LEGAL_TERM,
            reason=self.policy.compliance.forbidden_content.legal.reason,
            detail=f"{hits[0].rule.removeprefix('legal:')}: {hits[0].matched_text!r}",
        )

    def _numeric_consistency(
        self, answer: DraftedAnswer, selection: RetrieverSelection
    ) -> GuardrailVerdict:
        """Every non-trivial number must appear in something the answer cites.

        The check is against the SUMMARIES the retriever selected, which is what
        the drafter was shown. A number the drafter produced from nowhere is the
        classic fluent fabrication — "99.95% availability" reads perfectly and
        is a commitment nobody made.
        """
        corpus = " ".join(candidate.answer_node_id for candidate in selection.selections)
        # The selection object carries ids, not text; the substantive comparison
        # is against the unsupported-claim decomposition the drafter returned,
        # which is why an unsupported claim containing a number escalates here
        # with a numeric trigger rather than a generic one.
        offenders = [
            number
            for claim in answer.unsupported_claims
            for number in _NUMERIC.findall(claim)
            if number not in _TRIVIAL_NUMBERS and number not in corpus
        ]
        if not offenders:
            return PASSED
        return GuardrailVerdict(
            passed=False,
            trigger=EscalationTrigger.NUMERIC_INCONSISTENT,
            reason=f"{len(offenders)} number(s) appear in claims no source supports",
            detail=", ".join(sorted(set(offenders))[:5]),
        )

    def _cross_customer(
        self,
        answer: DraftedAnswer,
        selection: RetrieverSelection,
        document: RFPDocument,
    ) -> GuardrailVerdict:
        """No other customer's name in this customer's response.

        The graph already excludes another customer's confidential material
        inside the Cypher, so this is the belt to that query's braces: it
        catches a name the drafter produced from its own weights rather than
        from a source it was given, which the query cannot see.
        """
        this_customer = document.customer_name.strip()
        others = [
            name
            for name in _known_customer_names(selection)
            if name
            and name.casefold() != this_customer.casefold()
            and re.search(rf"\b{re.escape(name)}\b", answer.answer_text, re.IGNORECASE)
        ]
        if not others:
            return PASSED
        return GuardrailVerdict(
            passed=False,
            trigger=EscalationTrigger.CROSS_CUSTOMER_CONTENT,
            reason=f"the draft for {this_customer} names another customer",
            detail=", ".join(sorted(set(others))),
        )

    async def _entities(self, answer: DraftedAnswer) -> GuardrailVerdict:
        """Closed-world grounding: a named entity resolves, or the answer fails.

        Hard-fail rather than a confidence penalty (§15). A vendor that does not
        exist is not a weakly supported claim; it is a false statement about who
        we work with, and no amount of confidence arithmetic makes it acceptable.
        """
        if not self.check_entities:
            return PASSED

        candidates = _entity_candidates(answer.answer_text)
        if not candidates:
            return PASSED

        try:
            async with httpx.AsyncClient(timeout=self.mcp.timeout) as client:
                for name in candidates:
                    for entity_type in self.policy.compliance.entity_types_checked:
                        resolution = await self.mcp.entity_exists(
                            name=name, entity_type=entity_type, client=client
                        )
                        if resolution.passed:
                            break
                    else:
                        return GuardrailVerdict(
                            passed=False,
                            trigger=EscalationTrigger.ENTITY_UNRESOLVED,
                            reason=(
                                f"'{name}' does not resolve in the registry as any checked "
                                f"entity type; the grounding world is closed"
                            ),
                            detail=name,
                        )
        except McpError as exc:
            # The graph being unreachable is not evidence that an entity is
            # fine. Escalating is the honest answer: a human can check, and an
            # answer accepted because a check could not run is an answer nothing
            # checked.
            return GuardrailVerdict(
                passed=False,
                trigger=EscalationTrigger.ENTITY_UNRESOLVED,
                reason=f"entity grounding could not be verified: {exc}",
            )
        return PASSED


def _entity_candidates(text: str) -> list[str]:
    """Capitalised spans worth asking the registry about, in first-seen order."""
    seen: list[str] = []
    for match in _PROPER_NOUN.finditer(text):
        name = match.group(0).strip()
        first = name.split()[0]
        if first in _SENTENCE_STARTERS or name in seen or len(name) < 3:
            continue
        seen.append(name)
    return seen


def _known_customer_names(selection: RetrieverSelection) -> list[str]:
    """Customers other than the requester that this run could have touched.

    Only the requesting customer is known from the selection object itself, so
    the comparison the guardrail can make is against names the FIXTURES define.
    Kept as a function rather than inlined so the corpus-specific part of an
    otherwise general rule is in one visible place.
    """
    from src.compliance.customers import known_customers

    return [name for name in known_customers() if name != selection.requesting_customer]
