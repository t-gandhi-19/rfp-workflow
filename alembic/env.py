"""Alembic environment.

Runs as `rfp_migrator` — the only identity permitted to change the schema — with
the URL assembled from the environment so no password is ever committed.
"""

from __future__ import annotations

from logging.config import fileConfig

from alembic import context
from sqlalchemy import engine_from_config, pool

from src.state.db import migrator_url
from src.state.tables import metadata

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Migrations run synchronously; the async driver is only for request handling.
config.set_main_option("sqlalchemy.url", migrator_url(driver="postgresql+psycopg"))

target_metadata = metadata


def run_migrations_offline() -> None:
    """Emit SQL to stdout instead of executing it (`alembic upgrade --sql`)."""
    context.configure(
        url=config.get_main_option("sqlalchemy.url"),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            compare_type=True,
        )
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
