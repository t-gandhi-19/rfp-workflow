"""Run the retrieval eval and print the measurements. Phase 3 step 5.

    python -m scripts.eval_retrieval                  # config-default rerank
    python -m scripts.eval_retrieval --no-rerank      # rerank forced off
    python -m scripts.eval_retrieval --json out.json  # machine-readable too

Exits non-zero on any zero-tolerance violation, so it can gate. The numbers it
prints are the numbers of record quoted in the PR.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

from src.contracts.thresholds import ScoringConfig, scoring_config
from src.evals.retrieval import (
    RECALL_AT_5_THRESHOLD,
    RetrievalRun,
    run_retrieval_eval,
    to_category_result,
)
from src.gateway.fake_embedder import fake_embeddings_enabled
from src.graph.driver import close_driver, get_driver
from src.retrieval.calibration import load_for_current_corpus
from src.retrieval.retriever import Reranker


def rerank_disabled(config: ScoringConfig) -> ScoringConfig:
    """The same config with rerank off, for the ablation's second arm."""
    data = config.model_dump()
    data["rerank"]["enabled"] = False
    return ScoringConfig.model_validate(data)


def summary(run: RetrievalRun) -> str:
    result = to_category_result(run)
    lines = ["", "retrieval eval", ""]
    for metric in result.metrics:
        verdict = "" if metric.passed is None else ("  PASS" if metric.passed else "  FAIL")
        gate = "" if metric.threshold is None else f"  (threshold {metric.threshold})"
        lines.append(f"  {metric.label:<52} {metric.value:>8.4f}{gate}{verdict}")
        if metric.detail:
            lines.append(f"      {metric.detail}")
    lines.append("")

    if run.rank1_misses:
        lines.append("  RANK-1 PRIMARY-SOURCE MISSES (what Recall@5 cannot see):")
        for miss in run.rank1_misses:
            lines.append(
                f"    {miss.number}  expected {miss.expected_answer_id} at rank 1, "
                f"got {miss.rank1_with_preference} (rank {miss.rank_of_expected} without "
                f"preference: {miss.rank1_without_preference})"
            )
        lines.append("")

    lines.append("  per question:")
    lines.append(
        f"    {'no.':<5} {'kind':<13} {'status':<9} {'expected':<10} {'rank':<5} "
        f"{'cands':<6} pref-decisive"
    )
    for outcome in run.outcomes:
        rank = "-" if outcome.rank_of_expected is None else str(outcome.rank_of_expected)
        lines.append(
            f"    {outcome.number:<5} {outcome.kind:<13} {outcome.status.value:<9} "
            f"{outcome.expected_answer_id or '-':<10} {rank:<5} "
            f"{len(outcome.candidates):<6} {'YES' if outcome.preference_decisive else 'no'}"
        )
    lines.append("")

    lines.append("  attribution — top 3 per question (raw -> calibrated -> pref -> final):")
    for outcome in run.outcomes:
        lines.append(f"    {outcome.number}  floor {outcome.floor_used:.4f}")
        if not outcome.candidates:
            lines.append("      (no candidates)")
        for attribution in outcome.candidates[:3]:
            flag = "clears" if attribution.cleared_floor else "BELOW "
            lines.append(
                f"      {attribution.rank}. {attribution.answer_id:<10} "
                f"raw {attribution.raw_cosine:.4f}  cal {attribution.calibrated:.4f}  "
                f"rel {attribution.relevance:.4f}  pref {attribution.preference:.4f}  "
                f"final {attribution.final:.4f}  {flag}"
            )
    lines.append("")

    if result.violations:
        lines.append("  VIOLATIONS:")
        for violation in result.violations:
            where = f" [{violation.question_number}]" if violation.question_number else ""
            lines.append(f"    {violation.rule}{where}: {violation.detail}")
        lines.append("")
    else:
        lines.append("  no zero-tolerance violations.")
        lines.append("")
    return "\n".join(lines)


async def run(*, use_rerank: bool | None, json_path: Path | None) -> int:
    if fake_embeddings_enabled():
        sys.stderr.write(
            "WARNING: RFP_FAKE_EMBEDDINGS=1 — these vectors carry no semantics. "
            "This run exercises the machinery and says NOTHING about retrieval "
            "quality; the numbers of record come from real embeddings.\n"
        )

    config = scoring_config()
    if use_rerank is False:
        config = rerank_disabled(config)

    artifact = load_for_current_corpus()
    driver = get_driver()
    try:
        async with driver.session() as session:
            run_result = await run_retrieval_eval(
                session,
                calibration=artifact,
                reranker=Reranker(config=config),
                config=config,
            )
    finally:
        await close_driver()

    sys.stdout.write(summary(run_result))
    category = to_category_result(run_result)

    if json_path is not None:
        # ASYNC240 objects to blocking pathlib in an async function. This is a
        # one-shot CLI writing its own report after every await has completed;
        # there is no event loop left to starve.
        payload = {
            "rerank_enabled": config.rerank.enabled,
            "recall_at_5": run_result.recall_at_5,
            "mrr": run_result.mrr,
            "preference_decisive_rate": run_result.preference_decisive_rate,
            "no_match": sorted(run_result.no_match_numbers),
            "category": category.model_dump(mode="json"),
            "outcomes": [outcome.model_dump(mode="json") for outcome in run_result.outcomes],
        }
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(  # noqa: ASYNC240
            json.dumps(payload, indent=2) + "\n", encoding="utf-8", newline="\n"
        )
        sys.stdout.write(f"wrote {json_path}\n\n")

    if category.status.value != "PASS":
        sys.stderr.write(
            f"retrieval eval FAILED — Recall@5 {run_result.recall_at_5:.4f} against a "
            f"threshold of {RECALL_AT_5_THRESHOLD}, "
            f"{len(run_result.violations)} zero-tolerance violation(s). "
            "Constants do not move to make this pass.\n"
        )
        return 1
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the retrieval eval")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--rerank", dest="rerank", action="store_true", help="force rerank on")
    group.add_argument("--no-rerank", dest="rerank", action="store_false", help="force rerank off")
    parser.set_defaults(rerank=None)
    parser.add_argument("--json", type=Path, default=None, help="also write results as JSON")
    args = parser.parse_args()
    return asyncio.run(run(use_rerank=args.rerank, json_path=args.json))


if __name__ == "__main__":
    raise SystemExit(main())
