"""CLI wrapper for the embedding preflight checks. Run via `make preflight`.

Exits non-zero on any failure so `make ingest` can depend on it and refuse to
proceed.
"""

from __future__ import annotations

import asyncio
import os
import sys

from src.gateway.preflight import format_report, run_preflight


async def _main() -> int:
    results = await run_preflight(dict(os.environ))
    sys.stdout.write(format_report(results))
    return 0 if all(result.ok for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(_main()))
