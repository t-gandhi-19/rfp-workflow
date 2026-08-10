#!/bin/sh
# Apply migrations, then serve.
#
# Migrations run as `rfp_migrator` and the server then runs as `app_writer`
# (build prompt §17), so the identity that can change the schema is not the one
# handling requests. Alembic is idempotent, so a restart is a no-op.
set -eu

echo "write-api: applying migrations as rfp_migrator"
alembic upgrade head

echo "write-api: starting uvicorn as app_writer"
exec uvicorn src.write_api.app:app \
    --host 0.0.0.0 \
    --port 8001 \
    --log-level "$(echo "${LOG_LEVEL:-info}" | tr '[:upper:]' '[:lower:]')"
