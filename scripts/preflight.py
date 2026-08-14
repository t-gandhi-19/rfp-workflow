"""CLI wrapper for the embedding preflight checks. Run via `make preflight`.

Exits non-zero on any failure so `make ingest` can depend on it and refuse to
proceed.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys

from src.gateway.preflight import format_report, run_preflight


async def _main(*, include_calibration: bool) -> int:
    results = await run_preflight(dict(os.environ), include_calibration=include_calibration)
    sys.stdout.write(format_report(results))
    return 0 if all(result.ok for result in results) else 1


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify the local model path")
    parser.add_argument(
        "--skip-calibration",
        action="store_true",
        help=(
            "omit the calibration freshness check. Used when preflight runs as part of "
            "ingest, where the corpus being calibrated does not exist yet."
        ),
    )
    args = parser.parse_args()
    return asyncio.run(_main(include_calibration=not args.skip_calibration))


if __name__ == "__main__":
    raise SystemExit(main())
