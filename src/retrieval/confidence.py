"""Computed confidence (build prompt §10).

    confidence = final_score(primary source) x coverage_ratio + critique_delta

clamped to [0, 1]. Every term is arithmetic over things already measured. The
model is never asked how confident it is — a self-reported number is a fluent
guess, and fluency is exactly what an RFP response is full of.

Two properties are worth stating plainly:

* **Coverage is multiplicative, not additive.** An answer where half the claims
  carry no citation has its confidence halved, not lightly penalised. A partly
  ungrounded answer is not a slightly worse answer; it is one whose unsupported
  half could be anything.
* **The critic can only subtract.** `CritiqueResult.confidence_delta` is bounded
  at <= 0 by its contract, so this function cannot be used to talk confidence
  up, whatever a future prompt says.
"""

from __future__ import annotations

from src.contracts.thresholds import ScoringConfig, scoring_config


def coverage_ratio(*, claims_with_sources: int, total_claims: int) -> float:
    """Fraction of claims carrying at least one source id.

    An answer with no claims at all scores 0. That looks harsh, but an answer
    the claim extractor found nothing in is one nothing can be verified about,
    and treating "nothing to check" as "fully checked" is the failure mode this
    whole design exists to prevent.
    """
    if total_claims < 0 or claims_with_sources < 0:
        raise ValueError("claim counts must not be negative")
    if claims_with_sources > total_claims:
        raise ValueError(
            f"{claims_with_sources} sourced claims exceeds {total_claims} total claims"
        )
    if total_claims == 0:
        return 0.0
    return claims_with_sources / total_claims


def compute_confidence(
    *,
    primary_final_score: float,
    claims_with_sources: int,
    total_claims: int,
    critique_delta: float = 0.0,
    config: ScoringConfig | None = None,
) -> float:
    """The number that decides whether an answer needs a human.

    Below `confidence.escalation_threshold` the `DraftedAnswer` contract refuses
    to validate unless `needs_sme_review` is set, so this feeding a low value is
    not advisory — it forces the escalation.
    """
    resolved = config or scoring_config()
    if not 0.0 <= primary_final_score <= 1.0:
        raise ValueError(f"primary_final_score {primary_final_score} is outside [0, 1]")
    if critique_delta > 0:
        raise ValueError(
            f"critique_delta must not be positive, got {critique_delta}; "
            "the critic lowers confidence or leaves it alone"
        )

    ratio = coverage_ratio(claims_with_sources=claims_with_sources, total_claims=total_claims)
    raw = primary_final_score * ratio + critique_delta
    return min(resolved.confidence.clamp_max, max(resolved.confidence.clamp_min, raw))


def requires_sme_review(confidence: float, *, config: ScoringConfig | None = None) -> bool:
    """Whether this confidence obliges human review.

    The threshold is inclusive from above: exactly at the floor is acceptable,
    below it is not. Matches the `DraftedAnswer` validator, and the two are
    tested against the same config value so they cannot drift apart.
    """
    resolved = config or scoring_config()
    return confidence < resolved.confidence.escalation_threshold
