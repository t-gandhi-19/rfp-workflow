"""Typed contracts for every module boundary (CLAUDE.md rule 13).

No bare dict crosses a boundary in this system. Every crewAI task returns one of
these via `output_pydantic`, every MCP tool takes and returns one as JSON
Schema, and write-api accepts nothing else — so a malformed payload fails at the
edge with a readable error instead of halfway through a stage.

Import from this package root rather than the submodules:

    from src.contracts import DraftedAnswer, RunState
"""

from __future__ import annotations

from src.contracts.checks import (
    ComplianceResult,
    EntityCheckResult,
    ForbiddenContentHit,
    GuardrailVerdict,
)
from src.contracts.drafting import ConfidenceInputs, CritiqueResult, DraftedAnswer
from src.contracts.enums import (
    DocumentFormat,
    EntityType,
    EscalationTrigger,
    HaltReason,
    Outcome,
    QuestionStatus,
    QuestionType,
    RankingBasis,
    RetrievalStatus,
    RunStage,
    SelectionOutcome,
)
from src.contracts.escalation import EscalationRecord, EscalationsRecord
from src.contracts.retrieval import CandidateFlags, RetrievalResult, ScoredCandidate
from src.contracts.rfp import DocumentSection, ExtractedQuestion, RFPDocument, TriageResult
from src.contracts.run import ArtifactPaths, EvalScore, RunResult, RunState, RunTotals
from src.contracts.selection import CandidateSelection, RetrieverSelection
from src.contracts.thresholds import (
    ScoringConfig,
    confidence_escalation_threshold,
    reload_config,
    scoring_config,
)

__all__ = [
    "ArtifactPaths",
    "CandidateFlags",
    "CandidateSelection",
    "ComplianceResult",
    "ConfidenceInputs",
    "CritiqueResult",
    "DocumentFormat",
    "DocumentSection",
    "DraftedAnswer",
    "EntityCheckResult",
    "EntityType",
    "EscalationRecord",
    "EscalationTrigger",
    "EscalationsRecord",
    "EvalScore",
    "ExtractedQuestion",
    "ForbiddenContentHit",
    "GuardrailVerdict",
    "HaltReason",
    "Outcome",
    "QuestionStatus",
    "QuestionType",
    "RFPDocument",
    "RankingBasis",
    "RetrievalResult",
    "RetrievalStatus",
    "RetrieverSelection",
    "RunResult",
    "RunStage",
    "RunState",
    "RunTotals",
    "ScoredCandidate",
    "ScoringConfig",
    "SelectionOutcome",
    "TriageResult",
    # config accessors
    "confidence_escalation_threshold",
    "reload_config",
    "scoring_config",
]
