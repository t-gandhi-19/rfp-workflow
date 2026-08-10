"""Validate the hand-written answer key. Run via `make validate-manual-key`.

Checks structure and cross-references only. It deliberately does NOT compare the
key against `golden_rfp_source.json`: that file and the extractor share an
ancestor, so forcing agreement with it would convert the one independent
measurement in this repository back into a derived one. Disagreement between the
human key and the extractor is a finding for the eval report to surface, not
something this command should sand off.

    python -m scripts.validate_manual_key
    python -m scripts.validate_manual_key --emit-schema   # regenerate the schema
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from src.evals.manual_key import (
    MANUAL_KEY_PATH,
    MISSING_MESSAGE,
    SCHEMA_PATH,
    cross_reference_errors,
    json_schema,
    load_manual_key,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
QA_PAIRS = REPO_ROOT / "fixtures" / "qa_pairs.json"


def emit_schema() -> int:
    SCHEMA_PATH.write_text(
        json.dumps(json_schema(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    sys.stdout.write(f"wrote {SCHEMA_PATH.relative_to(REPO_ROOT)}\n")
    return 0


def validate() -> int:
    if not MANUAL_KEY_PATH.is_file():
        sys.stderr.write(MISSING_MESSAGE + "\n")
        return 1

    try:
        key = load_manual_key()
    except ValidationError as exc:
        sys.stderr.write(f"{MANUAL_KEY_PATH.relative_to(REPO_ROOT)} is not valid:\n\n{exc}\n")
        return 1
    except json.JSONDecodeError as exc:
        sys.stderr.write(f"{MANUAL_KEY_PATH.relative_to(REPO_ROOT)} is not valid JSON: {exc}\n")
        return 1

    with QA_PAIRS.open(encoding="utf-8") as handle:
        pairs: list[dict[str, Any]] = json.load(handle)

    errors = cross_reference_errors(key, pairs)
    if errors:
        sys.stderr.write("cross-reference errors:\n")
        for error in errors:
            sys.stderr.write(f"  - {error}\n")
        return 1

    matched = sum(1 for q in key.questions if q.expected_best_match_answer_id)
    sys.stdout.write(
        f"{MANUAL_KEY_PATH.relative_to(REPO_ROOT)} is valid\n"
        f"  source document : {key.source_document}\n"
        f"  authored by     : {key.authored_by} on {key.authored_on}\n"
        f"  questions        : {len(key.questions)}\n"
        f"  with expected match: {matched}\n"
        "\nStructure and cross-references only. Whether it agrees with the "
        "extractor is measured by the extraction evals, not here.\n"
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate the hand-written answer key")
    parser.add_argument(
        "--emit-schema",
        action="store_true",
        help="regenerate fixtures/answer_key_manual.schema.json from the model",
    )
    args = parser.parse_args()
    return emit_schema() if args.emit_schema else validate()


if __name__ == "__main__":
    raise SystemExit(main())
