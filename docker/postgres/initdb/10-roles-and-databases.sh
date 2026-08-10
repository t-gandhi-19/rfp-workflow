#!/bin/bash
# Runs once, on first boot of an empty data directory. `make nuke` clears the
# volume so this runs again.
#
# Creates the three least-privileged roles required by build prompt §17 and the
# separate `langfuse` database, so Langfuse shares this Postgres server instead
# of needing a container of its own.
#
#   rfp_migrator  owns the schema and is the only role Alembic runs as
#   app_writer    what write-api connects as: SELECT/INSERT/UPDATE, no DELETE
#   app_reader    SELECT only — MCP server, dashboard, log interpreter
#
# app_writer has no DELETE deliberately. Every write in this system is an
# idempotent upsert; nothing in the application has any reason to remove a row,
# and a compromised write path should not be able to erase the audit record.
set -euo pipefail

psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" <<EOSQL
-- ---------------------------------------------------------------------------
-- Application roles
-- ---------------------------------------------------------------------------
CREATE ROLE rfp_migrator LOGIN PASSWORD '${RFP_MIGRATOR_PASSWORD}';
CREATE ROLE app_writer   LOGIN PASSWORD '${APP_WRITER_PASSWORD}';
CREATE ROLE app_reader   LOGIN PASSWORD '${APP_READER_PASSWORD}';

-- Postgres 15+ no longer grants CREATE on public to everyone, so the migrator
-- needs it explicitly.
GRANT ALL   ON SCHEMA public TO rfp_migrator;
GRANT USAGE ON SCHEMA public TO app_writer, app_reader;

-- The migrator creates every table, so default privileges on ITS objects are
-- what actually govern the app roles. Without this, each migration would need
-- to remember to grant, and one forgotten grant would break write-api at
-- runtime rather than at deploy time.
ALTER DEFAULT PRIVILEGES FOR ROLE rfp_migrator IN SCHEMA public
    GRANT SELECT, INSERT, UPDATE ON TABLES TO app_writer;
ALTER DEFAULT PRIVILEGES FOR ROLE rfp_migrator IN SCHEMA public
    GRANT SELECT ON TABLES TO app_reader;
ALTER DEFAULT PRIVILEGES FOR ROLE rfp_migrator IN SCHEMA public
    GRANT USAGE, SELECT ON SEQUENCES TO app_writer;

-- ---------------------------------------------------------------------------
-- Co-tenants on this server, each with its own role and database.
-- Keycloak needs durable storage so the audit event history the Phase 6
-- dashboard reads survives a restart; Langfuse needs a database of its own.
-- Both live here rather than in containers of their own, to stay inside the
-- resource budget of a single developer machine.
-- ---------------------------------------------------------------------------
CREATE ROLE langfuse LOGIN PASSWORD '${LANGFUSE_DB_PASSWORD}';
CREATE ROLE keycloak LOGIN PASSWORD '${KEYCLOAK_DB_PASSWORD}';
EOSQL

# CREATE DATABASE cannot run inside the transaction block above.
psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" \
    -c "CREATE DATABASE langfuse OWNER langfuse;"
psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" \
    -c "CREATE DATABASE keycloak OWNER keycloak;"

echo "rfp-workflow: roles (rfp_migrator, app_writer, app_reader) created; langfuse and keycloak databases created"
