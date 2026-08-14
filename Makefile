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
	# --build so a source change is never silently served by a stale image.
	# Layer caching makes the no-change case near-free.
	GIT_SHA=$(GIT_SHA) $(COMPOSE) $(INFRA) $(APP) up -d --wait --build
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
HOST_NEO4J = NEO4J_URI=bolt://localhost:$${NEO4J_BOLT_PORT_HOST:-7687} \
             LITELLM_BASE_URL_HOST=$${LITELLM_BASE_URL_HOST:-http://localhost:$${LITELLM_PORT_HOST:-4000}}

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

.PHONY: test-report
test-report: ## Per-suite verbatim pytest summaries + the SHA they were produced at
	@# Every PR quotes this output verbatim. Reproducing the numbers by hand
	@# invites remembering them wrong, so the report is generated, never typed.
	@echo "commit:    $$(git rev-parse HEAD)"
	@echo "short SHA: $$(git rev-parse --short HEAD)"
	@echo "worktree:  $$(git status --porcelain | wc -l | tr -d ' ') uncommitted path(s)"
	@# Amendment M. The review gate is the REMOTE, so the report states what is
	@# actually on it. A commit that exists only locally is not reviewable, and
	@# "landed" was once written about exactly that — this line makes the claim
	@# generated rather than remembered, like the pytest summaries below it.
	@#
	@# The remote ref is re-fetched first: a stale origin/<branch> would report a
	@# push that has not happened, which is the failure this exists to catch.
	@BRANCH=$$(git rev-parse --abbrev-ref HEAD); \
	git fetch --quiet origin "$$BRANCH" 2>/dev/null || true; \
	LOCAL=$$(git rev-parse HEAD); \
	REMOTE=$$(git rev-parse --verify --quiet "origin/$$BRANCH" || echo ""); \
	if [ -z "$$REMOTE" ]; then \
		echo "pushed:    UNPUSHED — local $$(git rev-parse --short HEAD), no origin/$$BRANCH"; \
	elif [ "$$LOCAL" = "$$REMOTE" ]; then \
		echo "pushed:    $$BRANCH @ $$LOCAL (origin verified)"; \
	else \
		echo "pushed:    UNPUSHED — local $$(git rev-parse --short $$LOCAL) ahead of origin $$(git rev-parse --short $$REMOTE)"; \
	fi
	@set -a; [ -f .env ] && . ./.env; set +a; \
	export KEYCLOAK_BASE=$${KEYCLOAK_BASE:-http://localhost:$${KEYCLOAK_PORT_HOST:-8080}}; \
	export WRITE_API_BASE=$${WRITE_API_BASE:-http://localhost:$${WRITE_API_PORT:-8001}}; \
	for suite in unit security integration; do \
		printf '\n### tests/%s\n' "$$suite"; \
		$(RUN) pytest tests/$$suite -q 2>&1 | tail -1; \
	done

.PHONY: tag-phase
tag-phase: ## Tag a phase — make tag-phase TAG=v0.3 (refuses unless main is clean and synced)
	@TAG="$(TAG)" bash scripts/tag_phase.sh

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

PREFLIGHT_ENV = OLLAMA_BASE_URL_HOST=$${OLLAMA_BASE_URL_HOST:-http://localhost:11434} \
                LITELLM_BASE_URL_HOST=$${LITELLM_BASE_URL_HOST:-http://localhost:$${LITELLM_PORT_HOST:-4000}}

.PHONY: preflight
preflight: check-env ## Verify the embedding path and that calibration is current
	@# HOST_NEO4J for the same reason apply-schema and ingest carry it: .env sets
	@# NEO4J_URI to the COMPOSE SERVICE NAME, which only resolves inside the
	@# network. Amendment Q's auth probe runs from the host, so without this it
	@# dials a name that does not exist — on any machine, including a clean clone.
	@set -a && source .env && set +a && \
		$(HOST_NEO4J) $(PREFLIGHT_ENV) $(RUN) python -m scripts.preflight

.PHONY: preflight-pre-ingest
preflight-pre-ingest: check-env ## Preflight without the calibration check (nothing to calibrate yet)
	@# `ingest` depends on this target, so a missing HOST_NEO4J here stopped
	@# `make ingest` at the gate before it reached any of its own work.
	@set -a && source .env && set +a && \
		$(HOST_NEO4J) $(PREFLIGHT_ENV) $(RUN) python -m scripts.preflight --skip-calibration

.PHONY: apply-schema
apply-schema: check-env ## Apply the Neo4j schema (idempotent)
	@set -a && source .env && set +a && $(HOST_NEO4J) $(RUN) python -m scripts.apply_schema

.PHONY: fixtures
fixtures: ## Regenerate the synthetic fixtures
	$(RUN) python -m scripts.generate_fixtures

.PHONY: fixtures-check
fixtures-check: ## Verify committed fixtures match a fresh generation
	$(RUN) python -m scripts.generate_fixtures --check

.PHONY: validate-manual-key
validate-manual-key: ## Validate the hand-written answer key (structure + cross-refs)
	$(RUN) python -m scripts.validate_manual_key

.PHONY: manual-key-schema
manual-key-schema: ## Regenerate the manual answer key's JSON Schema from the model
	$(RUN) python -m scripts.validate_manual_key --emit-schema

.PHONY: calibrate
calibrate: check-env ## Measure the retrieval calibration anchors and derive the match floor
	@# Retrieval is fail-closed on the artifact this produces (amendment J), so
	@# this is not an optional tuning step — without it nothing retrieves.
	@# HOST_NEO4J since amendment S: calibration now compares its own similarity
	@# against the vector index before writing an artifact, so it dials the graph.
	@set -a && source .env && set +a && \
		$(HOST_NEO4J) $(PREFLIGHT_ENV) WRITE_API_PORT=$${WRITE_API_PORT:-8001} \
		$(RUN) python -m scripts.calibrate

.PHONY: calibrate-dry
calibrate-dry: check-env ## Measure and print the anchors without writing anything
	@set -a && source .env && set +a && \
		$(HOST_NEO4J) $(PREFLIGHT_ENV) $(RUN) python -m scripts.calibrate --dry-run

.PHONY: calibrate-commission
calibrate-commission: check-env ## SET the separation guard baselines from a real measurement (D18)
	@# The one command that may move the baselines. Ordinary `make calibrate` is
	@# JUDGED against them — a ratchet that resets itself on every run never
	@# catches anything. Refuses to run against the stand-in embedder.
	@# HOST_NEO4J since amendment S: calibration now compares its own similarity
	@# against the vector index before writing an artifact, so it dials the graph.
	@set -a && source .env && set +a && \
		$(HOST_NEO4J) $(PREFLIGHT_ENV) WRITE_API_PORT=$${WRITE_API_PORT:-8001} \
		$(RUN) python -m scripts.calibrate --commission

.PHONY: ingest
ingest: preflight-pre-ingest apply-schema ## Load fixtures into the graph, then calibrate
	@# Calibration runs LAST and is part of ingest rather than a step someone
	@# remembers: the anchors are a property of the corpus, so a corpus that has
	@# just changed has a calibration that no longer describes it.
	@set -a && source .env && set +a && $(HOST_NEO4J) $(RUN) python -m scripts.ingest
	@$(MAKE) --no-print-directory calibrate

.PHONY: reembed
reembed: preflight-pre-ingest ## Recompute every embedding, then recalibrate
	@set -a && source .env && set +a && $(HOST_NEO4J) $(RUN) python -m scripts.ingest --reembed
	@$(MAKE) --no-print-directory calibrate

# The harness dials Neo4j (retrieval), the gateway (embeddings), Keycloak and
# write-api (persisting eval_results as evals-sa), and Postgres directly for the
# previous SHA's rows. HOST_PG as well as HOST_NEO4J, because that read is the
# only place the harness talks to Postgres without going through write-api.
.PHONY: evals
evals: check-env ## Run the eval harness with the config-default rerank setting (fast path)
	@set -a && source .env && set +a && \
		$(HOST_NEO4J) $(HOST_PG) $(PREFLIGHT_ENV) \
		KEYCLOAK_BASE=$${KEYCLOAK_BASE:-http://localhost:$${KEYCLOAK_PORT_HOST:-8080}} \
		WRITE_API_BASE=$${WRITE_API_BASE:-http://localhost:$${WRITE_API_PORT:-8001}} \
		$(RUN) python -m scripts.evals

.PHONY: evals-full
evals-full: check-env ## Run the eval harness with rerank FORCED ON — the official numbers
	@set -a && source .env && set +a && \
		$(HOST_NEO4J) $(HOST_PG) $(PREFLIGHT_ENV) \
		KEYCLOAK_BASE=$${KEYCLOAK_BASE:-http://localhost:$${KEYCLOAK_PORT_HOST:-8080}} \
		WRITE_API_BASE=$${WRITE_API_BASE:-http://localhost:$${WRITE_API_PORT:-8001}} \
		$(RUN) python -m scripts.evals --rerank

.PHONY: evals-ablation
evals-ablation: check-env ## Run retrieval twice, rerank on and off, and report the deltas
	@set -a && source .env && set +a && \
		$(HOST_NEO4J) $(HOST_PG) $(PREFLIGHT_ENV) \
		KEYCLOAK_BASE=$${KEYCLOAK_BASE:-http://localhost:$${KEYCLOAK_PORT_HOST:-8080}} \
		WRITE_API_BASE=$${WRITE_API_BASE:-http://localhost:$${WRITE_API_PORT:-8001}} \
		$(RUN) python -m scripts.evals --rerank --ablation

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
