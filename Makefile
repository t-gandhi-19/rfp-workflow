# rfp-workflow
#
# Targets belonging to a later phase exit with a message naming that phase,
# rather than failing obscurely. The list is the roadmap.

SHELL := /bin/bash
.DEFAULT_GOAL := help

COMPOSE      := docker compose
INFRA        := --profile infra
APP          := --profile app
UV           := uv
RUN          := $(UV) run
GIT_SHA      := $(shell git rev-parse --short HEAD 2>/dev/null || echo local-dev)

.PHONY: help
help: ## Show this help
	@echo "rfp-workflow — make targets"
	@echo
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'
	@echo

# ---------------------------------------------------------------------------
# Environment guard
# ---------------------------------------------------------------------------
.PHONY: check-env
check-env:
	@test -f .env || { \
		echo "ERROR: .env is missing. Run: cp .env.example .env  (then set GROQ_API_KEY)"; \
		exit 1; \
	}

# ---------------------------------------------------------------------------
# Stack
# ---------------------------------------------------------------------------
.PHONY: build
build: check-env ## Build application images
	$(COMPOSE) $(INFRA) $(APP) build

.PHONY: up
up: check-env ## Bring up the whole stack (infra + app) and wait for health
	GIT_SHA=$(GIT_SHA) $(COMPOSE) $(INFRA) $(APP) up -d --wait
	@echo
	@echo "Stack is up:"
	@echo "  Keycloak   http://localhost:8080  (realm: rfp)"
	@echo "  Neo4j      http://localhost:7474"
	@echo "  Langfuse   http://localhost:3000"
	@echo "  LiteLLM    http://localhost:4000"
	@echo "  write-api  http://localhost:$${WRITE_API_PORT:-8001}/health"

.PHONY: up-infra
up-infra: check-env ## Bring up infrastructure only, no application containers
	GIT_SHA=$(GIT_SHA) $(COMPOSE) $(INFRA) up -d --wait

.PHONY: down
down: ## Stop the stack, keeping volumes
	$(COMPOSE) $(INFRA) $(APP) down

.PHONY: nuke
nuke: ## Stop the stack and DELETE every named volume
	$(COMPOSE) $(INFRA) $(APP) down --volumes --remove-orphans
	@echo "All volumes removed. The next 'make up' re-runs database initialisation."

.PHONY: logs
logs: ## Tail logs from every service
	$(COMPOSE) $(INFRA) $(APP) logs -f --tail=100

.PHONY: ps
ps: ## Show service status
	$(COMPOSE) $(INFRA) $(APP) ps

# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------
# Run from the host, so Postgres is reached on the published port rather than
# by its compose service name.
HOST_PG = POSTGRES_HOST=localhost POSTGRES_PORT=$${POSTGRES_PORT_HOST:-5432}

.PHONY: migrate
migrate: check-env ## Apply Alembic migrations (idempotent)
	set -a && source .env && set +a && $(HOST_PG) $(RUN) alembic upgrade head

.PHONY: migrate-status
migrate-status: check-env ## Show the current migration revision
	set -a && source .env && set +a && $(HOST_PG) $(RUN) alembic current

# ---------------------------------------------------------------------------
# Development
# ---------------------------------------------------------------------------
.PHONY: install
install: ## Sync the virtualenv from uv.lock
	$(UV) sync --group dev

.PHONY: test
test: ## Run unit and security tests
	$(RUN) pytest tests/unit tests/security -q

.PHONY: test-integration
test-integration: check-env ## Run integration tests (requires 'make up')
	$(RUN) pytest tests/integration -q -m integration

.PHONY: test-all
test-all: test test-integration ## Run every test

.PHONY: lint
lint: ## ruff check + format check + mypy
	$(RUN) ruff check .
	$(RUN) ruff format --check .
	$(RUN) mypy src tests

.PHONY: fmt
fmt: ## Autoformat and autofix
	$(RUN) ruff check --fix .
	$(RUN) ruff format .

# ---------------------------------------------------------------------------
# Pipeline — arriving in later phases
# ---------------------------------------------------------------------------
define phase_gate
	@echo "'make $(1)' arrives in Phase $(2)."
	@echo "$(3)"
	@exit 1
endef

.PHONY: ingest
ingest: ## (Phase 2) Load synthetic fixtures into the graph
	$(call phase_gate,ingest,2,Needs the graph schema and the synthetic corpus.)

.PHONY: reembed
reembed: ## (Phase 2) Re-embed the corpus after an embedding-model change
	$(call phase_gate,reembed,2,Needs ingest and the pinned embedding model.)

.PHONY: evals
evals: ## (Phase 3) Run the eval harness and write the HTML report
	$(call phase_gate,evals,3,Needs retrieval scoring and the golden answer key.)

.PHONY: run
run: ## (Phase 4) Run one RFP end to end — make run FILE=path/to/rfp.pdf
	$(call phase_gate,run,4,Needs the agent crew and the deterministic controller.)

.PHONY: resume
resume: ## (Phase 4) Resume an interrupted run — make resume RUN=<run_id>
	$(call phase_gate,resume,4,Needs checkpointed run state from a real run.)

.PHONY: demo
demo: ## (Phase 5) End-to-end on the golden RFP, then open the report
	$(call phase_gate,demo,5,Needs the full pipeline and the adversarial suite.)

.PHONY: explain
explain: ## (Phase 6) Plain-English narrative of a run — make explain RUN=<run_id>
	$(call phase_gate,explain,6,Needs the log interpreter and a completed run.)
