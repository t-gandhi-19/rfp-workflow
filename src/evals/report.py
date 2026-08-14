"""The HTML eval report written to `out/`.

Self-contained by construction: one file, inline CSS, no scripts and no network
requests. A report that fetches a stylesheet is a report that renders wrong in
six months, and this one is meant to be attachable to a PR and readable from a
filesystem with no server in front of it.

WHAT THE REPORT IS FOR. Not a dashboard — a record. It states what was measured,
at which commit, against which thresholds, and what moved since the last SHA. So
every number is rendered with the threshold it was judged against and the verdict
that was reached AT THE TIME, rather than recomputed at render: a threshold that
moved after the fact would otherwise silently rewrite history.

NOT_IMPLEMENTED categories are rendered greyed, named, with their reason. See
src/evals/registry.py for why they appear at all.
"""

from __future__ import annotations

import html
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from src.evals.contracts import CategoryResult, CategoryStatus, MetricDirection
from src.evals.registry import in_report_order
from src.evals.store import Delta

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = REPO_ROOT / "out" / "evals.html"

_CSS = """
:root { color-scheme: light dark; }
body { font: 15px/1.55 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
       margin: 0 auto; max-width: 60rem; padding: 2rem 1.25rem 4rem; }
h1 { font-size: 1.6rem; margin: 0 0 .25rem; }
h2 { font-size: 1.15rem; margin: 2.25rem 0 .5rem; }
.sub { color: #6b7280; font-size: .875rem; margin: 0 0 1.5rem; }
.meta { border-collapse: collapse; font-size: .875rem; margin: 0 0 1rem; }
.meta td { padding: .15rem .75rem .15rem 0; vertical-align: top; }
.meta td:first-child { color: #6b7280; white-space: nowrap; }
table.metrics { border-collapse: collapse; width: 100%; font-size: .9rem; }
table.metrics th, table.metrics td { border-bottom: 1px solid #e5e7eb; padding: .45rem .6rem;
                                     text-align: left; vertical-align: top; }
table.metrics th { font-weight: 600; font-size: .8rem; text-transform: uppercase;
                   letter-spacing: .04em; color: #6b7280; }
td.num { text-align: right; font-variant-numeric: tabular-nums; white-space: nowrap; }
.detail { color: #6b7280; font-size: .82rem; }
.badge { border-radius: .25rem; display: inline-block; font-size: .75rem; font-weight: 600;
         padding: .1rem .45rem; }
.badge.pass { background: #dcfce7; color: #166534; }
.badge.fail { background: #fee2e2; color: #991b1b; }
.badge.none { background: #f3f4f6; color: #6b7280; }
.badge.notimpl { background: #f3f4f6; color: #6b7280; }
section.greyed { opacity: .62; border-left: 3px solid #d1d5db; padding-left: 1rem; }
.reason { background: #f9fafb; border-radius: .35rem; font-size: .86rem; padding: .6rem .8rem; }
.viol { background: #fff7ed; border-left: 3px solid #f59e0b; font-size: .85rem;
        margin: .35rem 0; padding: .45rem .7rem; }
.viol code { font-weight: 600; }
.note { color: #4b5563; font-size: .85rem; margin: .35rem 0; }
.up { color: #166534; } .down { color: #991b1b; } .flat { color: #6b7280; }
footer { border-top: 1px solid #e5e7eb; color: #6b7280; font-size: .8rem;
         margin-top: 3rem; padding-top: 1rem; }
@media (prefers-color-scheme: dark) {
  body { background: #0b0f19; color: #e5e7eb; }
  table.metrics th, table.metrics td { border-bottom-color: #1f2937; }
  .reason { background: #111827; }
  .viol { background: #1c1917; }
  .badge.pass { background: #052e16; color: #86efac; }
  .badge.fail { background: #450a0a; color: #fca5a5; }
  .badge.none, .badge.notimpl { background: #1f2937; color: #9ca3af; }
  footer { border-top-color: #1f2937; }
  .up { color: #86efac; } .down { color: #fca5a5; }
}
"""


@dataclass(frozen=True)
class ReportContext:
    """Everything the report states about the run that produced it."""

    git_sha: str
    run_id: str
    rerank_enabled: bool
    embedder: str
    #: True when the run used the deterministic stand-in embedder.
    fake_embeddings: bool
    previous_sha: str | None
    deltas: list[Delta]
    #: None when persistence was skipped; otherwise the row count written.
    rows_written: int | None


def _e(value: object) -> str:
    return html.escape(str(value))


def _verdict_badge(passed: bool | None) -> str:
    if passed is None:
        return '<span class="badge none">reported</span>'
    return (
        '<span class="badge pass">PASS</span>' if passed else '<span class="badge fail">FAIL</span>'
    )


def _format_value(value: float, direction: MetricDirection) -> str:
    """Counts render as counts; rates render to four places.

    A MUST_BE_ZERO metric counts violations, and "0.0000 violations" reads as a
    rate that happens to be zero rather than as none at all.
    """
    if direction is MetricDirection.MUST_BE_ZERO:
        return f"{int(value)}"
    return f"{value:.4f}"


def _delta_cell(metric_key: str, deltas: list[Delta], direction: MetricDirection) -> str:
    match = next((delta for delta in deltas if delta.metric == metric_key), None)
    if match is None:
        return '<span class="flat">—</span>'
    change = match.change
    if abs(change) < 1e-9:
        return '<span class="flat">0.0000</span>'
    improved = change > 0 if direction is MetricDirection.HIGHER_IS_BETTER else change < 0
    css = "up" if improved else "down"
    return f'<span class="{css}">{change:+.4f}</span>'


def _category_section(result: CategoryResult, context: ReportContext) -> str:
    if result.status is CategoryStatus.NOT_IMPLEMENTED:
        return (
            f'<section class="greyed">\n'
            f"<h2>{_e(result.label)} "
            f'<span class="badge notimpl">NOT IMPLEMENTED</span></h2>\n'
            f'<p class="reason">{_e(result.not_implemented_reason or "")}</p>\n'
            f"</section>"
        )

    badge = (
        '<span class="badge pass">PASS</span>'
        if result.status is CategoryStatus.PASS
        else '<span class="badge fail">FAIL</span>'
    )
    rows = []
    for metric in result.metrics:
        threshold = "—" if metric.threshold is None else f"{metric.threshold:g}"
        detail = f'<div class="detail">{_e(metric.detail)}</div>' if metric.detail else ""
        rows.append(
            "<tr>"
            f"<td>{_e(metric.label)}{detail}</td>"
            f'<td class="num">{_format_value(metric.value, metric.direction)}</td>'
            f'<td class="num">{threshold}</td>'
            f'<td class="num">'
            f"{_delta_cell(f'{result.key}.{metric.key}', context.deltas, metric.direction)}</td>"
            f"<td>{_verdict_badge(metric.passed)}</td>"
            "</tr>"
        )

    violations = "".join(
        f'<div class="viol"><code>{_e(violation.rule)}</code>'
        + (f" [{_e(violation.question_number)}]" if violation.question_number else "")
        + f": {_e(violation.detail)}</div>"
        for violation in result.violations
    )
    notes = "".join(f'<p class="note">{_e(note)}</p>' for note in result.notes)

    return (
        f"<section>\n<h2>{_e(result.label)} {badge}</h2>\n"
        '<table class="metrics">\n'
        "<tr><th>Metric</th><th>Value</th><th>Threshold</th>"
        "<th>Δ vs prev</th><th>Verdict</th></tr>\n"
        + "\n".join(rows)
        + "\n</table>\n"
        + notes
        + violations
        + "</section>"
    )


def render(results: list[CategoryResult], context: ReportContext) -> str:
    """The whole report as one HTML string."""
    ordered = in_report_order(results)
    measured = [r for r in ordered if r.status is not CategoryStatus.NOT_IMPLEMENTED]
    failed = [r for r in measured if r.status is CategoryStatus.FAIL]

    overall = (
        '<span class="badge fail">FAIL</span>' if failed else '<span class="badge pass">PASS</span>'
    )

    baseline = (
        f"{_e(context.previous_sha)} ({len(context.deltas)} metric(s) compared)"
        if context.previous_sha
        else "none — no earlier SHA in eval_results"
    )
    persisted = (
        "skipped (--no-persist)"
        if context.rows_written is None
        else f"{context.rows_written} row(s) to eval_results"
    )

    # Stated on the report itself, not only in the console: a run on stand-in
    # vectors exercises the machinery and says nothing about retrieval quality,
    # and an HTML file outlives the terminal it was produced in.
    warning = (
        "<tr><td>embeddings</td><td><strong>STAND-IN (RFP_FAKE_EMBEDDINGS=1)</strong> — "
        "these vectors carry no semantics. This run exercises the machinery and says "
        "NOTHING about retrieval quality.</td></tr>"
        if context.fake_embeddings
        else f"<tr><td>embedder</td><td>{_e(context.embedder)}</td></tr>"
    )

    sections = "\n".join(_category_section(result, context) for result in ordered)
    generated = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S UTC")

    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>rfp-workflow eval report — {_e(context.git_sha[:12])}</title>
<style>{_CSS}</style></head><body>
<h1>Eval report {overall}</h1>
<p class="sub">Generated {generated}. Every number below was produced by
<code>make evals</code>; none of it is typed by hand.</p>
<table class="meta">
<tr><td>commit</td><td><code>{_e(context.git_sha)}</code></td></tr>
<tr><td>run id</td><td><code>{_e(context.run_id)}</code></td></tr>
<tr><td>rerank</td><td>{
        "ON — the shipped configuration"
        if context.rerank_enabled
        else "OFF — an ablation arm, not the shipped configuration"
    }</td></tr>
{warning}
<tr><td>baseline</td><td>{baseline}</td></tr>
<tr><td>persisted</td><td>{persisted}</td></tr>
<tr><td>categories</td><td>{len(measured)} measured,
    {len(ordered) - len(measured)} not implemented</td></tr>
</table>
{sections}
<footer>
Categories shown greyed are specified in §3 and not yet measured; each states the
input it is waiting on. The eval harness is built in Phase 3, before the agents in
Phase 4, by design — a harness written after the thing it grades tends to grade
what that thing happens to do.
</footer>
</body></html>
"""


def write(results: list[CategoryResult], context: ReportContext, path: Path = DEFAULT_OUT) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render(results, context), encoding="utf-8", newline="\n")
    return path
