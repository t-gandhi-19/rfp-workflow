"""Shared fixtures.

Kept deliberately small: contract tests should construct their own objects so a
reader can see exactly what is being asserted.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def now() -> datetime:
    """A timezone-aware timestamp. Contracts reject naive datetimes."""
    return datetime(2026, 6, 1, 12, 0, 0, tzinfo=UTC)


@pytest.fixture
def repo_root() -> Path:
    return REPO_ROOT
