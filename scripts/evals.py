"""The eval harness. Phase 3 step 8 — the last build of the phase.

    python -m scripts.evals                  # config-default rerank
    python -m scripts.evals --rerank         # rerank FORCED ON: numbers of record
    python -m scripts.evals --ablation       # retrieval twice, on and off, with deltas
    python -m scripts.evals --no-persist     # report only, no Postgres write

Runs every implemented category, renders one HTML report to `out/`, writes the
gated metrics to Postgres keyed by git SHA, and prints the movement against the
previous SHA. Exits non-zero if any measured category fails, so it can gate.

WHAT IT DOES NOT DO. It does not touch a threshold, a calibration constant, or
`fixtures/answer_key_manual.json`. A failing eval is a finding about the system,
not a prompt to move the line it failed against.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import uuid
from pathlib import Path

from src.contracts.embedding import embedding_config
from src.contracts.thresholds import ScoringConfig, scoring_config
from src.evals.contracts import CategoryResult, CategoryStatus
from src.evals.extraction import run_extraction_eval
from src.evals.extraction import to_category_result as extraction_result
from src.evals.registry import (
    IMPLEMENTED,
    Category,
    all_placeholders,
    deselected,
    in_report_order,
)
from src.evals.report import DEFAULT_OUT, ReportContext, write
from src.evals.retrieval import run_retrieval_eval
from src.evals.retrieval import to_category_result as retrieval_result
from src.evals.store import (
    Delta,
    deltas,
    git_sha,
    persist,
    previous_scores,
    to_scores,
)
from src.gateway.fake_embedder import fake_embeddings_enabled
from src.graph.driver import close_driver, get_driver
from src.observability.logging import configure_app_logging
from src.retrieval.calibration import load_for_current_corpus
from src.retrieval.retriever import Reranker


def rerank_disabled(config: ScoringConfig) -> ScoringConfig:
    """The same config with rerank off, for the ablation's second arm."""
    data = config.model_dump()
    data["rerank"]["enabled"] = False
    return ScoringConfig.model_validate(data)


async def _retrieval(config: ScoringConfig) -> CategoryResult:
    artifact = load_for_current_corpus()
    driver = get_driver()
    try:
        async with driver.session() as session:
            run = await run_retrieval_eval(
                session,
                calibration=artifact,
                reranker=Reranker(config=config),
                config=config,
            )
    finally:
        await close_driver()
    return retrieval_result(run)


def _console(results: list[CategoryResult], movements: list[Delta]) -> str:
    lines: list[str] = ["", "eval summary", ""]
    for result in in_report_order(results):
        if result.status is CategoryStatus.NOT_IMPLEMENTED:
            lines.append(f"  {result.key:<14} NOT_IMPLEMENTED  {result.label}")
            continue
        lines.append(f"  {result.key:<14} {result.status.value:<16} {result.label}")
        for metric in result.metrics:
            verdict = "" if metric.passed is None else ("PASS" if metric.passed else "FAIL")
            gate = "" if metric.threshold is None else f" (>= {metric.threshold:g})"
            lines.append(f"      {metric.label:<58} {metric.value:>9.4f}{gate:<12} {verdict}")
        for violation in result.violations:
            where = f" [{violation.question_number}]" if violation.question_number else ""
            lines.append(f"      ! {violation.rule}{where}: {violation.detail}")
    lines.append("")

    if movements:
        lines.append("  deltas vs the previous SHA:")
        for delta in sorted(movements, key=lambda d: d.metric):
            lines.append(
                f"      {delta.metric:<44} {delta.previous:>9.4f} -> "
                f"{delta.current:>9.4f}  {delta.change:+.4f}"
            )
    else:
        lines.append("  deltas: none — no earlier SHA in eval_results to compare against.")
    lines.append("")
    return "\n".join(lines)


def _ablation_table(on: CategoryResult, off: CategoryResult) -> str:
    """Rerank ON vs OFF, metric by metric.

    Both arms are rendered whatever the verdict: the rerank-OFF arm is EXPECTED
    to fail rank-1 accuracy, because that is the golden-1.1 inversion, and a
    table that hid a failing arm would hide the finding the ablation exists to
    surface.
    """
    by_key = {metric.key: metric for metric in off.metrics}
    lines = [
        "",
        "rerank ablation — ON is the shipped configuration",
        "",
        f"  {'metric':<40} {'ON':>10} {'OFF':>10} {'delta':>10}",
    ]
    for metric in on.metrics:
        other = by_key.get(metric.key)
        if other is None:
            continue
        lines.append(
            f"  {metric.key:<40} {metric.value:>10.4f} {other.value:>10.4f} "
            f"{metric.value - other.value:>+10.4f}"
        )
    lines.append("")
    lines.append(f"  verdict ON:  {on.status.value}")
    lines.append(f"  verdict OFF: {off.status.value}")
    lines.append("")
    return "\n".join(lines)


async def run(
    *,
    use_rerank: bool | None,
    ablation: bool,
    do_persist: bool,
    out_path: Path,
    categories: set[str],
) -> int:
    # Same reason mcp-server does it: without a handler on the `rfp` tree, the
    # store's "baseline unavailable" warning goes nowhere, and a silent
    # degradation to "no previous SHA" is exactly the failure that warning exists
    # to make visible.
    configure_app_logging()

    if fake_embeddings_enabled():
        sys.stderr.write(
            "WARNING: RFP_FAKE_EMBEDDINGS=1 — these vectors carry no semantics. "
            "This run exercises the machinery and says NOTHING about retrieval "
            "quality; the numbers of record come from real embeddings.\n"
        )

    config = scoring_config()
    if use_rerank is True:
        data = config.model_dump()
        data["rerank"]["enabled"] = True
        config = ScoringConfig.model_validate(data)
    elif use_rerank is False:
        config = rerank_disabled(config)

    results: list[CategoryResult] = []

    # Extraction first: it needs no database and no models, so a broken stack
    # still produces the half of the report that does not depend on one.
    if Category.EXTRACTION in categories:
        results.append(extraction_result(run_extraction_eval()))

    if Category.RETRIEVAL in categories:
        retrieval_on = await _retrieval(config)
        results.append(retrieval_on)

        if ablation:
            other = rerank_disabled(config) if config.rerank.enabled else config
            retrieval_off = await _retrieval(other)
            sys.stdout.write(_ablation_table(retrieval_on, retrieval_off))

    # A category the caller excluded is reported as a placeholder rather than
    # dropped, and the reason says it was DESELECTED rather than unbuilt. The
    # two look identical in a report otherwise, and only one of them means the
    # code does not exist.
    for key in sorted(IMPLEMENTED - categories):
        results.append(deselected(key))

    results.extend(all_placeholders())

    sha = git_sha()
    run_id = f"evals-{uuid.uuid4().hex[:12]}"
    scores = to_scores(results, sha=sha, run_id=run_id)

    previous_sha, previous = await previous_scores(current_sha=sha)
    movements = deltas(scores, previous)

    rows_written: int | None = None
    if do_persist:
        rows_written = await persist(scores)

    report = write(
        results,
        ReportContext(
            git_sha=sha,
            run_id=run_id,
            rerank_enabled=config.rerank.enabled,
            embedder=embedding_config().model.alias,
            fake_embeddings=fake_embeddings_enabled(),
            previous_sha=previous_sha,
            deltas=movements,
            rows_written=rows_written,
        ),
        out_path,
    )

    sys.stdout.write(_console(results, movements))
    sys.stdout.write(f"  report: {report}\n")
    sys.stdout.write(
        f"  persisted: {'skipped' if rows_written is None else f'{rows_written} row(s)'} "
        f"at {sha}\n\n"
    )

    failed = [result for result in results if result.status is CategoryStatus.FAIL]
    if failed:
        sys.stderr.write(
            f"evals FAILED — {', '.join(result.key for result in failed)}. "
            "Thresholds, calibration constants and the manual answer key do not "
            "move to make this pass.\n"
        )
        return 1
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the eval harness")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--rerank", dest="rerank", action="store_true", help="force rerank on")
    group.add_argument("--no-rerank", dest="rerank", action="store_false", help="force rerank off")
    parser.set_defaults(rerank=None)
    parser.add_argument(
        "--ablation",
        action="store_true",
        help="also run retrieval with rerank flipped, and print the deltas",
    )
    parser.add_argument(
        "--no-persist",
        dest="persist",
        action="store_false",
        help="render the report without writing to Postgres",
    )
    parser.set_defaults(persist=True)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="where to write the report")
    parser.add_argument(
        "--categories",
        default=",".join(sorted(IMPLEMENTED)),
        help=(
            "comma-separated categories to run; default all implemented. "
            "CI uses --categories extraction, because retrieval measured on "
            "stand-in vectors is a number about nothing."
        ),
    )
    args = parser.parse_args()

    selected = {name.strip() for name in args.categories.split(",") if name.strip()}
    if unknown := sorted(selected - IMPLEMENTED):
        parser.error(
            f"not implemented (or not a category): {', '.join(unknown)}. "
            f"Implemented: {', '.join(sorted(IMPLEMENTED))}."
        )

    return asyncio.run(
        run(
            use_rerank=args.rerank,
            ablation=args.ablation,
            do_persist=args.persist,
            out_path=args.out,
            categories=selected,
        )
    )


if __name__ == "__main__":
    raise SystemExit(main())
