"""Database URL construction and the async engine.

Two identities exist on purpose:

* ``app_writer`` — what write-api runs as. SELECT/INSERT/UPDATE, no DELETE.
* ``rfp_migrator`` — what Alembic runs as, and nothing else.

Keeping them apart means a bug in request handling cannot alter the schema, and
the migration identity is never available to a running request.
"""

from __future__ import annotations

import os
from functools import lru_cache
from urllib.parse import quote

from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine


def _require(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"Required environment variable {name} is not set")
    return value


def database_url(*, user: str, password: str, driver: str = "postgresql+psycopg") -> str:
    """Build a connection URL, percent-encoding credentials.

    Encoding matters: a password containing `@` or `/` silently produces a URL
    pointing somewhere else entirely.
    """
    host = os.environ.get("POSTGRES_HOST", "localhost")
    port = os.environ.get("POSTGRES_PORT", "5432")
    database = os.environ.get("POSTGRES_DB", "rfp")
    return f"{driver}://{quote(user)}:{quote(password)}@{host}:{port}/{database}"


def writer_url(*, driver: str = "postgresql+psycopg") -> str:
    """URL for the least-privileged application write identity."""
    return database_url(user="app_writer", password=_require("APP_WRITER_PASSWORD"), driver=driver)


def reader_url(*, driver: str = "postgresql+psycopg") -> str:
    """URL for the SELECT-only identity used by readers."""
    return database_url(user="app_reader", password=_require("APP_READER_PASSWORD"), driver=driver)


def migrator_url(*, driver: str = "postgresql+psycopg") -> str:
    """URL for the schema owner. Used by Alembic only."""
    return database_url(
        user="rfp_migrator", password=_require("RFP_MIGRATOR_PASSWORD"), driver=driver
    )


@lru_cache(maxsize=1)
def get_engine() -> AsyncEngine:
    """Process-wide async engine for write-api, created on first use."""
    return create_async_engine(
        writer_url(),
        pool_size=5,
        max_overflow=5,
        pool_pre_ping=True,
        echo=False,
    )


async def dispose_engine() -> None:
    """Close pooled connections on shutdown."""
    if get_engine.cache_info().currsize:
        await get_engine().dispose()
        get_engine.cache_clear()
