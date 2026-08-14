"""`make run FILE=...` and `make resume RUN=...` — one RFP, end to end.

This is the composition root: the only place the real agent layer, the real
guardrails, the real compliance checker and the real assembler are wired to the
deterministic controller. Everything below it takes its collaborators as
arguments, which is what let the controller be built and tested before any of
them existed.

NOTHING HERE SUBMITS ANYTHING (rule 2). The run produces a draft response, an
escalation list and a set of checkpoints. Human review is the terminal stage,
and there is no code path past it.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import uuid
from datetime import UTC, datetime
from pathlib import Path

from src.agents.layer import CrewAgentLayer
from src.assembly import TemplateAssembler
from src.compliance import DeterministicComplianceChecker
from src.compliance.policy import domain_policy
from src.contracts import DocumentFormat, RFPDocument, RunResult
from src.controller import BudgetLedger, Checkpointer, RunController, load_run
from src.extraction.document import extract
from src.graph import driver as graph_driver
from src.graph import queries
from src.guardrails.suite import DeterministicGuardrails
from src.observability.logging import configure_logging
from src.observability.tracing import configure_tracing, instrument_httpx, tracing_status

logger = logging.getLogger("rfp.run")


def build_document(path: Path, *, customer: str, rfp_id: str) -> RFPDocument:
    """Read the file into the intake contract.

    `customer_name` is supplied rather than parsed. It decides what the graph
    query will let this run see, and inferring it from a document's prose would
    make confidentiality depend on a regex.
    """
    extracted = extract(path)
    return RFPDocument(
        id=rfp_id,
        source_filename=path.name,
        format=DocumentFormat(extracted.document_format.value),
        raw_text=extracted.text,
        customer_name=customer,
        domain=domain_policy().key,
        intake_timestamp=datetime.now(UTC),
    )


async def _sme_routing(session: object) -> dict[str, tuple[str, str]]:
    """capability id -> (sme_id, sme_name), read once before assembly.

    Resolved up front rather than per escalation: the mapping is small, fixed
    for the run, and reading it once means a graph hiccup cannot leave half the
    TODO blocks assigned and half not.
    """
    routing: dict[str, tuple[str, str]] = {}
    for capability in domain_policy().retrieval.capabilities:
        try:
            smes = await queries.get_sme_for_capability(session, capability_id=capability)  # type: ignore[arg-type]
        except Exception as exc:
            logger.warning("SME routing for '%s' unavailable: %s", capability, exc)
            continue
        if smes:
            routing[capability] = (smes[0].id, smes[0].name)
    return routing


async def execute(
    *, file: Path | None, customer: str, resume_run: str | None, out_dir: Path
) -> RunResult:
    configure_logging()
    status = configure_tracing("rfp-controller")
    # Instrumenting httpx is THE JOIN (§19): every outbound call — to the
    # gateway, to write-api, to mcp-server — carries a traceparent, so our spans
    # and LiteLLM's proxy-side spans land in one trace rather than being
    # correlated afterwards by run id or by timestamp.
    instrument_httpx()
    logger.info("%s", status.summary)

    async with graph_driver.session() as session:
        agents = CrewAgentLayer(session=session)  # type: ignore[arg-type]
        routing = await _sme_routing(session)
        controller = RunController(
            agents=agents,
            guardrails=DeterministicGuardrails(),
            compliance=DeterministicComplianceChecker(),
            assembler=TemplateAssembler(out_dir=out_dir, sme_for_capability=routing),
            checkpointer=Checkpointer(),
            ledger=BudgetLedger(),
        )

        if resume_run is not None:
            resumed = await load_run(resume_run)
            if file is None:
                raise SystemExit(
                    "make resume needs the source document too: "
                    "make resume RUN=<id> FILE=path/to/rfp.pdf"
                )
            document = build_document(file, customer=customer, rfp_id=resumed.state.rfp_id)
            return await controller.resume(document, resumed=resumed)

        if file is None:
            raise SystemExit("make run needs a document: make run FILE=path/to/rfp.pdf")
        rfp_id = f"rfp-{uuid.uuid4().hex[:12]}"
        document = build_document(file, customer=customer, rfp_id=rfp_id)
        return await controller.run(document, run_id=f"run-{uuid.uuid4().hex[:12]}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run one RFP through the workflow")
    parser.add_argument("--file", type=Path, default=None, help="Path to the RFP (PDF or DOCX)")
    parser.add_argument(
        "--customer",
        default="Meridian Insurance Group",
        help="Who the response is for. Decides what the graph will let this run see.",
    )
    parser.add_argument("--resume", dest="resume_run", default=None, help="Resume a run id")
    parser.add_argument("--out", type=Path, default=Path("out"), help="Artifact directory")
    args = parser.parse_args()

    result = asyncio.run(
        execute(
            file=args.file,
            customer=args.customer,
            resume_run=args.resume_run,
            out_dir=args.out,
        )
    )
    # `sys.stdout.write`, matching `scripts/ingest.py`: these scripts' stdout is
    # JSON that CI and `make demo` parse, so it stays a single deliberate write
    # rather than whatever `print` is configured to do.
    sys.stdout.write(
        json.dumps(
            {
                "run_id": result.run_id,
                "questions": result.totals.questions,
                "answered": result.totals.answered,
                "escalated": result.totals.escalated,
                "failed": result.totals.failed,
                "tokens_used": result.totals.tokens_used,
                "cost_usd": result.totals.cost_usd,
                "trace_url": result.trace_url,
                "tracing": tracing_status().summary,
                "artifacts": result.artifact_paths.model_dump(mode="json"),
            },
            indent=2,
        )
        + "\n"
    )
    # A run that answered nothing is not a successful run, whatever the stages
    # did — and `make demo` on a clean clone has to fail loudly if it happens.
    return 0 if result.totals.answered > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
