"""mcp-server's client for write-api.

ONE FUNCTION, ONE ENDPOINT. This exists so `save_draft` can honour rule 4 —
Postgres writes go only through write-api — without the MCP server growing a
general-purpose HTTP client that a future tool could point anywhere.

THE CALLER'S TOKEN IS FORWARDED, not replaced. mcp-server holds no service
account of its own for this, deliberately: minting one would make every row's
`written_by` say "mcp-server", and the question an audit trail is asked is which
principal produced an artifact. Forwarding also means write-api's own role check
still applies — a caller who somehow reached this tool without `draft-writer`
is refused twice, by two services, for the same reason.
"""

from __future__ import annotations

import os
from typing import Any

import httpx

from src.contracts import DraftedAnswer


class WriteApiError(RuntimeError):
    """write-api refused or could not be reached."""


def write_api_base() -> str:
    return os.environ.get(
        "WRITE_API_BASE", f"http://localhost:{os.environ.get('WRITE_API_PORT', '8001')}"
    )


async def put_draft(*, run_id: str, answer: DraftedAnswer, token: str) -> dict[str, Any]:
    """PUT one draft. Idempotent on (run_id, question_id), like every write."""
    url = f"{write_api_base()}/v1/drafts/{run_id}/{answer.question_id}"
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.put(
                url,
                json=answer.model_dump(mode="json"),
                headers={"Authorization": f"Bearer {token}"},
            )
    except httpx.HTTPError as exc:
        raise WriteApiError(f"write-api unreachable at {url}: {exc}") from exc

    if response.status_code != 200:
        raise WriteApiError(
            f"write-api refused the draft: {response.status_code} {response.text[:300]}"
        )
    body: dict[str, Any] = response.json()
    return body
