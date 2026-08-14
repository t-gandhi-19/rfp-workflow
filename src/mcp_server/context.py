"""What a tool handler is given.

WHY A CONTEXT OBJECT RATHER THAN A SESSION. The six read tools needed exactly
one thing — a graph session. The two Phase 4 tools need different things:
`save_draft` needs the CALLER'S OWN TOKEN, because it writes through write-api
and the row's `written_by` must record who actually asked rather than recording
that "mcp-server" did. `get_run_state` needs neither a graph session nor a
token; it reads Postgres with the SELECT-only identity.

Threading all three through one object keeps the handler signature uniform, and
keeps the alternative — a second handler shape for the new tools — from
existing. Two shapes is how a boundary stops being a boundary.
"""

from __future__ import annotations

from dataclasses import dataclass

from neo4j import AsyncSession

from src.write_api.auth import Principal


@dataclass(frozen=True)
class ToolContext:
    """One tool invocation's surroundings."""

    principal: Principal
    #: None for tools that do not touch the graph. Opened in READ access mode,
    #: so Neo4j itself refuses a write on it.
    session: AsyncSession | None = None
    #: The caller's raw bearer token, forwarded when a tool writes through
    #: write-api. None for read tools, which have no reason to hold one.
    token: str | None = None

    def require_session(self) -> AsyncSession:
        if self.session is None:
            raise RuntimeError("this tool needs a graph session and was given none")
        return self.session

    def require_token(self) -> str:
        if self.token is None:
            raise RuntimeError("this tool writes through write-api and was given no token")
        return self.token
