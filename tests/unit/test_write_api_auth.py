"""write-api authentication and authorization.

These run against a locally generated RSA keypair and a stubbed database, so the
whole auth path — signature, issuer, audience, expiry, algorithm, role — is
exercised with no Keycloak and no Postgres. The live counterpart against the
real realm runs in the compose smoke job.

The distinction under test throughout is 401 versus 403: *who are you* versus
*you may not do this*. Collapsing them would hide a misconfigured role grant
behind what looks like a token problem.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from typing import Any

import httpx
import jwt
import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from sqlalchemy.dialects import postgresql

from src.contracts import RunStage, RunState
from src.write_api import app as app_module
from src.write_api.auth import TokenVerifier
from src.write_api.settings import WriteApiSettings

ISSUER = "http://keycloak.test/realms/rfp"
AUDIENCE = "rfp-api"
SUBJECT = "service-account-drafter-sa-uuid"

RUN_ID = "run-001"
QUESTION_ID = "q-001"


# ---------------------------------------------------------------------------
# Keys and tokens
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def keypair() -> tuple[Any, str]:
    """A throwaway RSA keypair standing in for the realm's signing key."""
    private = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    pem = private.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    ).decode()
    return private.public_key(), pem


@pytest.fixture(scope="module")
def other_keypair() -> tuple[Any, str]:
    """A second keypair — a validly-formed token signed by the wrong issuer."""
    private = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    pem = private.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    ).decode()
    return private.public_key(), pem


def make_token(
    private_pem: str,
    *,
    roles: list[str] | None = None,
    audience: str = AUDIENCE,
    issuer: str = ISSUER,
    expires_in: timedelta = timedelta(minutes=5),
    subject: str = SUBJECT,
    client_id: str = "drafter-sa",
) -> str:
    now = datetime.now(tz=UTC)
    claims: dict[str, Any] = {
        "sub": subject,
        "iss": issuer,
        "aud": audience,
        "azp": client_id,
        "iat": int(now.timestamp()),
        "exp": int((now + expires_in).timestamp()),
        "realm_access": {"roles": roles if roles is not None else ["draft-writer"]},
    }
    return jwt.encode(claims, private_pem, algorithm="RS256", headers={"kid": "test-key"})


class _StaticKeyResolver:
    """Stands in for PyJWKClient, returning one known public key."""

    def __init__(self, public_key: Any) -> None:
        self._public_key = public_key

    def get_signing_key_from_jwt(self, token: str) -> SimpleNamespace:
        return SimpleNamespace(key=self._public_key)


# ---------------------------------------------------------------------------
# Stub database
# ---------------------------------------------------------------------------


class _RecordingConnection:
    """Captures the statements a request would have executed."""

    def __init__(self) -> None:
        self.statements: list[Any] = []

    async def execute(self, statement: Any) -> None:
        self.statements.append(statement)


class _StubEngine:
    def __init__(self) -> None:
        self.connection = _RecordingConnection()

    def begin(self) -> Any:
        @asynccontextmanager
        async def _cm() -> Any:
            yield self.connection

        return _cm()

    def connect(self) -> Any:
        return self.begin()


@pytest.fixture
def engine(monkeypatch: pytest.MonkeyPatch) -> _StubEngine:
    stub = _StubEngine()
    monkeypatch.setattr(app_module, "get_engine", lambda: stub)
    return stub


@pytest.fixture
def client(keypair: tuple[Any, str], engine: _StubEngine) -> httpx.AsyncClient:
    public_key, _ = keypair
    settings = WriteApiSettings(
        keycloak_issuer=ISSUER,
        keycloak_audience=AUDIENCE,
    )
    app_module.app.state.verifier = TokenVerifier(
        settings, signing_key_resolver=_StaticKeyResolver(public_key)
    )
    transport = httpx.ASGITransport(app=app_module.app)
    return httpx.AsyncClient(transport=transport, base_url="http://write-api")


def _run_state_payload() -> dict[str, Any]:
    started = datetime(2026, 6, 1, 12, 0, tzinfo=UTC)
    return RunState(
        run_id=RUN_ID,
        rfp_id="rfp-001",
        stage=RunStage.DRAFTING,
        started_at=started,
        updated_at=started + timedelta(minutes=3),
    ).model_dump(mode="json")


def _draft_payload(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "question_id": QUESTION_ID,
        "answer_text": "We run a six-week assessment before any workload moves.",
        "source_ids": ["ans-014"],
        "confidence": 0.82,
        "needs_sme_review": False,
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestUnauthenticatedIs401:
    async def test_missing_token(self, client: httpx.AsyncClient) -> None:
        async with client:
            response = await client.put(f"/v1/runs/{RUN_ID}", json=_run_state_payload())
        assert response.status_code == 401
        assert response.headers["www-authenticate"] == "Bearer"

    async def test_garbage_token(self, client: httpx.AsyncClient) -> None:
        async with client:
            response = await client.put(
                f"/v1/runs/{RUN_ID}",
                json=_run_state_payload(),
                headers={"Authorization": "Bearer not-a-jwt"},
            )
        assert response.status_code == 401

    async def test_token_signed_by_an_unknown_key(
        self, client: httpx.AsyncClient, other_keypair: tuple[Any, str]
    ) -> None:
        """A well-formed token from the wrong issuer's key must not pass."""
        _, wrong_pem = other_keypair
        async with client:
            response = await client.put(
                f"/v1/runs/{RUN_ID}",
                json=_run_state_payload(),
                headers={"Authorization": f"Bearer {make_token(wrong_pem)}"},
            )
        assert response.status_code == 401

    async def test_expired_token(self, client: httpx.AsyncClient, keypair: tuple[Any, str]) -> None:
        _, pem = keypair
        token = make_token(pem, expires_in=timedelta(minutes=-30))
        async with client:
            response = await client.put(
                f"/v1/runs/{RUN_ID}",
                json=_run_state_payload(),
                headers={"Authorization": f"Bearer {token}"},
            )
        assert response.status_code == 401

    async def test_wrong_audience(
        self, client: httpx.AsyncClient, keypair: tuple[Any, str]
    ) -> None:
        """A token minted for another resource in the same realm is not ours."""
        _, pem = keypair
        token = make_token(pem, audience="some-other-api")
        async with client:
            response = await client.put(
                f"/v1/runs/{RUN_ID}",
                json=_run_state_payload(),
                headers={"Authorization": f"Bearer {token}"},
            )
        assert response.status_code == 401

    async def test_wrong_issuer(self, client: httpx.AsyncClient, keypair: tuple[Any, str]) -> None:
        _, pem = keypair
        token = make_token(pem, issuer="http://evil.test/realms/rfp")
        async with client:
            response = await client.put(
                f"/v1/runs/{RUN_ID}",
                json=_run_state_payload(),
                headers={"Authorization": f"Bearer {token}"},
            )
        assert response.status_code == 401

    async def test_unsigned_token_is_rejected(self, client: httpx.AsyncClient) -> None:
        """`alg: none` must never be accepted — the algorithm is pinned to RS256."""
        now = datetime.now(tz=UTC)
        unsigned = jwt.encode(
            {
                "sub": SUBJECT,
                "iss": ISSUER,
                "aud": AUDIENCE,
                "iat": int(now.timestamp()),
                "exp": int((now + timedelta(minutes=5)).timestamp()),
                "realm_access": {"roles": ["draft-writer"]},
            },
            key="",
            algorithm="none",
        )
        async with client:
            response = await client.put(
                f"/v1/runs/{RUN_ID}",
                json=_run_state_payload(),
                headers={"Authorization": f"Bearer {unsigned}"},
            )
        assert response.status_code == 401


class TestWrongRoleIs403:
    async def test_authenticated_but_unauthorized(
        self, client: httpx.AsyncClient, keypair: tuple[Any, str]
    ) -> None:
        """triage-sa holds rfp-reader, never draft-writer."""
        _, pem = keypair
        token = make_token(pem, roles=["rfp-reader"], client_id="triage-sa")
        async with client:
            response = await client.put(
                f"/v1/runs/{RUN_ID}",
                json=_run_state_payload(),
                headers={"Authorization": f"Bearer {token}"},
            )
        assert response.status_code == 403
        assert "draft-writer" in response.json()["detail"]

    async def test_no_roles_at_all(
        self, client: httpx.AsyncClient, keypair: tuple[Any, str]
    ) -> None:
        _, pem = keypair
        async with client:
            response = await client.put(
                f"/v1/runs/{RUN_ID}",
                json=_run_state_payload(),
                headers={"Authorization": f"Bearer {make_token(pem, roles=[])}"},
            )
        assert response.status_code == 403

    async def test_submitter_alone_grants_nothing(
        self, client: httpx.AsyncClient, keypair: tuple[Any, str]
    ) -> None:
        """Even if `submitter` were somehow granted, it authorizes no endpoint."""
        _, pem = keypair
        async with client:
            response = await client.put(
                f"/v1/runs/{RUN_ID}",
                json=_run_state_payload(),
                headers={"Authorization": f"Bearer {make_token(pem, roles=['submitter'])}"},
            )
        assert response.status_code == 403


class TestAuthorizedWrites:
    async def test_run_checkpoint_succeeds(
        self, client: httpx.AsyncClient, keypair: tuple[Any, str], engine: _StubEngine
    ) -> None:
        _, pem = keypair
        async with client:
            response = await client.put(
                f"/v1/runs/{RUN_ID}",
                json=_run_state_payload(),
                headers={"Authorization": f"Bearer {make_token(pem)}"},
            )
        assert response.status_code == 200
        assert response.json() == {
            "written": "run",
            "key": RUN_ID,
            "written_by": SUBJECT,
        }
        assert len(engine.connection.statements) == 1

    async def test_written_by_is_the_token_subject(
        self, client: httpx.AsyncClient, keypair: tuple[Any, str]
    ) -> None:
        """The audit trail is only meaningful if it records the real caller."""
        _, pem = keypair
        token = make_token(pem, subject="service-account-assembler-sa-uuid")
        async with client:
            response = await client.put(
                f"/v1/drafts/{RUN_ID}/{QUESTION_ID}",
                json=_draft_payload(),
                headers={"Authorization": f"Bearer {token}"},
            )
        assert response.status_code == 200
        assert response.json()["written_by"] == "service-account-assembler-sa-uuid"

    async def test_question_status_write(
        self, client: httpx.AsyncClient, keypair: tuple[Any, str]
    ) -> None:
        _, pem = keypair
        async with client:
            response = await client.put(
                f"/v1/runs/{RUN_ID}/questions/{QUESTION_ID}/status",
                json={"status": "escalated"},
                headers={"Authorization": f"Bearer {make_token(pem)}"},
            )
        assert response.status_code == 200
        assert response.json()["key"] == f"{RUN_ID}/{QUESTION_ID}"

    async def test_eval_results_batch(
        self, client: httpx.AsyncClient, keypair: tuple[Any, str], engine: _StubEngine
    ) -> None:
        _, pem = keypair
        payload = {
            "scores": [
                {
                    "git_sha": "abc123",
                    "run_id": RUN_ID,
                    "metric": "retrieval.recall_at_5",
                    "value": 0.86,
                    "threshold": 0.8,
                    "passed": True,
                },
                {
                    "git_sha": "abc123",
                    "run_id": RUN_ID,
                    "metric": "compliance.word_limit_violations",
                    "value": 0.0,
                    "threshold": 0.0,
                    "passed": True,
                },
            ]
        }
        async with client:
            response = await client.post(
                "/v1/eval-results",
                json=payload,
                headers={"Authorization": f"Bearer {make_token(pem)}"},
            )
        assert response.status_code == 200
        assert len(engine.connection.statements) == 2


class TestBoundaryValidation:
    async def test_path_and_body_must_agree(
        self, client: httpx.AsyncClient, keypair: tuple[Any, str]
    ) -> None:
        _, pem = keypair
        async with client:
            response = await client.put(
                "/v1/runs/run-999",
                json=_run_state_payload(),
                headers={"Authorization": f"Bearer {make_token(pem)}"},
            )
        assert response.status_code == 400

    async def test_an_ungrounded_draft_is_refused_at_the_boundary(
        self, client: httpx.AsyncClient, keypair: tuple[Any, str], engine: _StubEngine
    ) -> None:
        """The contract rejects it before a connection is ever opened."""
        _, pem = keypair
        async with client:
            response = await client.put(
                f"/v1/drafts/{RUN_ID}/{QUESTION_ID}",
                json=_draft_payload(source_ids=[]),
                headers={"Authorization": f"Bearer {make_token(pem)}"},
            )
        assert response.status_code == 422
        assert engine.connection.statements == []

    async def test_a_halted_run_without_a_reason_is_refused(
        self, client: httpx.AsyncClient, keypair: tuple[Any, str]
    ) -> None:
        _, pem = keypair
        payload = _run_state_payload() | {"stage": "halted"}
        async with client:
            response = await client.put(
                f"/v1/runs/{RUN_ID}",
                json=payload,
                headers={"Authorization": f"Bearer {make_token(pem)}"},
            )
        assert response.status_code == 422

    async def test_unknown_body_fields_are_refused(
        self, client: httpx.AsyncClient, keypair: tuple[Any, str]
    ) -> None:
        _, pem = keypair
        async with client:
            response = await client.put(
                f"/v1/drafts/{RUN_ID}/{QUESTION_ID}",
                json=_draft_payload(confidenc=0.9),
                headers={"Authorization": f"Bearer {make_token(pem)}"},
            )
        assert response.status_code == 422


class TestWritesArePreparedSafely:
    async def test_statements_bind_values_rather_than_inlining_them(
        self, client: httpx.AsyncClient, keypair: tuple[Any, str], engine: _StubEngine
    ) -> None:
        """Compiling without literal binds proves the values travel as parameters.

        A statement built by interpolation would carry the payload inside the SQL
        text; this one carries placeholders and a separate parameter map.
        """
        _, pem = keypair
        hostile = "'; DROP TABLE runs; --"
        async with client:
            response = await client.put(
                f"/v1/drafts/{RUN_ID}/{QUESTION_ID}",
                json=_draft_payload(answer_text=hostile),
                headers={"Authorization": f"Bearer {make_token(pem)}"},
            )
        assert response.status_code == 200

        statement = engine.connection.statements[0]
        compiled = statement.compile(dialect=postgresql.dialect())
        assert hostile not in str(compiled)
        assert hostile in compiled.params.values()
