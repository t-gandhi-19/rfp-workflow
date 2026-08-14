"""mcp-server — the only path an agent has to the graph (§2, CLAUDE.md rule 4).

Six read tools, each wrapping a tested, parameterised function in
`src.graph.queries`. No tool accepts Cypher, and none writes.

**Auth is the write-api's, imported rather than reimplemented.** Same
`TokenVerifier`, same RS256/JWKS/audience/issuer checks, same `require_role`
factory — so 401 and 403 mean the same thing on both services because they are
the same code, not because two implementations were kept in step by hand. 401 is
"I do not know who you are"; 403 is "I know exactly who you are and you may not
do this".

**The caller's subject is logged on every call**, alongside the tool name and the
outcome. A read that surfaced confidential material is exactly as auditable as a
write, which matters because retrieval is where a confidentiality failure would
actually occur.

Read-only by construction: the session is opened in READ access mode, so a write
would be refused by Neo4j itself even if one were somehow reachable from here.
"""

from __future__ import annotations

import logging
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Annotated, Any

from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.responses import JSONResponse
from neo4j import READ_ACCESS
from pydantic import BaseModel, ConfigDict, ValidationError

from src.graph.driver import close_driver, get_driver
from src.mcp_server.tools import KG_READER, TOOLS, TOOLS_BY_NAME, ToolSpec
from src.observability.logging import configure_app_logging
from src.write_api.auth import Principal, build_verifier, require_role
from src.write_api.settings import get_settings

logger = logging.getLogger("rfp.mcp")


class ToolManifest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tools: list[dict[str, Any]]


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    # Before anything else: uvicorn leaves the root logger bare, so without this
    # every per-call audit line below is discarded in the container. See
    # src/observability/logging.py — this is the fix for a real, found defect,
    # not a precaution.
    configure_app_logging()
    app.state.verifier = build_verifier()
    app.state.settings = get_settings()
    yield
    await close_driver()


app = FastAPI(
    title="rfp mcp-server",
    version="1",
    description="Read-only graph tools. Every tool wraps a tested query function.",
    lifespan=lifespan,
)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok", "git_sha": get_settings().git_sha}


@app.get("/tools", response_model=ToolManifest)
async def list_tools(
    principal: Annotated[Principal, Depends(require_role(KG_READER))],
) -> ToolManifest:
    """The manifest. Behind the same role as invocation.

    Listing what a caller may not invoke is an information leak in miniature —
    it tells an unauthorised caller exactly what exists to be attacked.
    """
    logger.info(
        "mcp.tools.list subject=%s client=%s tools=%d",
        principal.subject,
        principal.client_id,
        len(TOOLS),
    )
    return ToolManifest(tools=[tool.schema() for tool in TOOLS])


def _resolve(name: str) -> ToolSpec:
    tool = TOOLS_BY_NAME.get(name)
    if tool is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=(
                f"No tool named '{name}'. Available: {', '.join(sorted(TOOLS_BY_NAME))}. "
                "There is no raw-query tool; graph access is only through these."
            ),
        )
    return tool


@app.post("/tools/{name}")
async def invoke_tool(
    name: str,
    payload: dict[str, Any],
    principal: Annotated[Principal, Depends(require_role(KG_READER))],
) -> JSONResponse:
    """Validate, run the wrapped query, log the caller, return the contract.

    The payload arrives as a bare dict and is validated against the tool's own
    input contract HERE rather than by FastAPI's signature, because the contract
    varies per tool. The 422 that results names the offending field, which is
    what makes a boundary failure readable instead of a stack trace.
    """
    tool = _resolve(name)

    try:
        parsed = tool.input_model.model_validate(payload)
    except ValidationError as exc:
        logger.info(
            "mcp.tool.invalid subject=%s client=%s tool=%s errors=%d",
            principal.subject,
            principal.client_id,
            name,
            len(exc.errors()),
        )
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={
                "tool": name,
                "message": f"input does not satisfy the contract for '{name}'",
                "errors": [
                    {
                        "field": ".".join(str(part) for part in error["loc"]) or "(root)",
                        "problem": error["msg"],
                    }
                    for error in exc.errors()
                ],
            },
        ) from exc

    started = time.perf_counter()
    driver = get_driver()
    try:
        # READ access mode: Neo4j itself refuses a write on this session, so
        # "these tools are read-only" is enforced by the database rather than by
        # the absence of a write tool.
        async with driver.session(default_access_mode=READ_ACCESS) as session:
            result = await tool.handler(session, parsed)
    except ValueError as exc:
        # A contract-shaped complaint raised by the handler (a wrong-width
        # embedding, an empty requesting_customer). Same 422 as a schema
        # failure, because to a caller it is the same class of mistake.
        logger.info(
            "mcp.tool.rejected subject=%s client=%s tool=%s reason=%s",
            principal.subject,
            principal.client_id,
            name,
            exc,
        )
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={"tool": name, "message": str(exc)},
        ) from exc

    elapsed_ms = (time.perf_counter() - started) * 1000
    logger.info(
        "mcp.tool.ok subject=%s client=%s tool=%s ms=%.1f",
        principal.subject,
        principal.client_id,
        name,
        elapsed_ms,
    )
    return JSONResponse(content=result.model_dump(mode="json"))


@app.exception_handler(Exception)
async def unhandled(request: Request, exc: Exception) -> JSONResponse:
    """Never leak an internal error's text to a tool caller.

    The detail goes to the log with the path; the caller gets a stable shape.
    An agent is not a debugger, and a stack trace in a tool response is both
    useless to it and a disclosure.
    """
    logger.exception("mcp.unhandled path=%s", request.url.path)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"message": "internal error; see the mcp-server log"},
    )
