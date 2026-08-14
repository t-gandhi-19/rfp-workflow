# mcp-server image.
#
# Deliberately the same shape as docker/api.Dockerfile — same pinned base, same
# two-stage split, same non-root uid, same lockfile install. The two services
# share their auth code (mcp-server imports write-api's TokenVerifier), so an
# image that drifted in Python version or dependency set would let 401/403 mean
# something subtly different on each, which is the exact failure that sharing the
# code was meant to rule out.
#
# There is NO entrypoint script here, unlike write-api. write-api runs migrations
# before serving; mcp-server owns no schema and writes nothing, so it goes
# straight to uvicorn. A read-only service acquiring a migration step would be a
# design change worth noticing, and its absence here is where that shows.

# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------
FROM python:3.12.8-slim-bookworm AS builder

COPY --from=ghcr.io/astral-sh/uv:0.5.11 /uv /usr/local/bin/uv

WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

# Dependencies first, so a source change does not invalidate the dependency
# layer. --no-install-project keeps our own code out of this step.
COPY pyproject.toml uv.lock README.md ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project --no-dev

# No alembic/ and no alembic.ini: this service has no migrations to run, and an
# image that cannot reach the migration tooling cannot be talked into running it.
COPY src/ ./src/
COPY config/ ./config/

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev

# ---------------------------------------------------------------------------
# Runtime
# ---------------------------------------------------------------------------
FROM python:3.12.8-slim-bookworm AS runtime

RUN groupadd --gid 10001 app \
    && useradd --uid 10001 --gid app --create-home --shell /usr/sbin/nologin app

WORKDIR /app
COPY --from=builder --chown=app:app /app /app

ENV PATH="/app/.venv/bin:${PATH}" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

USER app
EXPOSE 8002

# A shell is needed only to fold LOG_LEVEL to the lowercase uvicorn accepts —
# .env carries INFO, uvicorn wants info. `exec` hands the process over, so
# uvicorn still becomes pid 1 and receives SIGTERM directly; without it the
# shell would swallow the signal and `docker compose down` would wait out the
# ten-second kill timer on every stop.
CMD ["sh", "-c", "exec uvicorn src.mcp_server.app:app --host 0.0.0.0 --port 8002 --log-level $(echo ${LOG_LEVEL:-info} | tr '[:upper:]' '[:lower:]')"]
