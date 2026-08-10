"""End-to-end checks against the running stack.

Everything here uses real components: a real Keycloak issuing real
client-credentials tokens, a real Postgres with migrations applied, and
write-api enforcing the real realm's roles. The unit suite proves the auth logic
in isolation; this proves the wiring — that the realm actually imported, that
the issuer and audience the containers agree on match what write-api validates,
and that `submitter` is genuinely unheld in a live realm rather than only in the
JSON we ship.

Run with `make up && make test-integration`, or in CI where the compose job
starts postgres, keycloak, and write-api.
"""

from __future__ import annotations

import os
from typing import Any

import httpx
import pytest

pytestmark = pytest.mark.integration

KEYCLOAK_BASE = os.environ.get("KEYCLOAK_BASE", "http://localhost:8080")
WRITE_API_BASE = os.environ.get("WRITE_API_BASE", "http://localhost:8001")
REALM = os.environ.get("KEYCLOAK_REALM", "rfp")

TOKEN_URL = f"{KEYCLOAK_BASE}/realms/{REALM}/protocol/openid-connect/token"
ADMIN_TOKEN_URL = f"{KEYCLOAK_BASE}/realms/master/protocol/openid-connect/token"

RUN_ID = "run-integration-001"
QUESTION_ID = "q-integration-001"


async def _client_credentials_token(client: httpx.AsyncClient, client_id: str, secret: str) -> str:
    response = await client.post(
        TOKEN_URL,
        data={
            "grant_type": "client_credentials",
            "client_id": client_id,
            "client_secret": secret,
        },
    )
    response.raise_for_status()
    token: str = response.json()["access_token"]
    return token


async def _admin_token(client: httpx.AsyncClient) -> str:
    response = await client.post(
        ADMIN_TOKEN_URL,
        data={
            "grant_type": "password",
            "client_id": "admin-cli",
            "username": os.environ.get("KEYCLOAK_ADMIN", "admin"),
            "password": os.environ["KEYCLOAK_ADMIN_PASSWORD"],
        },
    )
    response.raise_for_status()
    token: str = response.json()["access_token"]
    return token


def _run_payload() -> dict[str, Any]:
    return {
        "run_id": RUN_ID,
        "rfp_id": "rfp-integration-001",
        "stage": "drafting",
        "per_question_status": {QUESTION_ID: "pending"},
        "tokens_used": 1234,
        "cost_usd": 0.0421,
        "started_at": "2026-06-01T12:00:00Z",
        "updated_at": "2026-06-01T12:03:00Z",
    }


@pytest.fixture
async def http() -> Any:
    async with httpx.AsyncClient(timeout=30.0) as client:
        yield client


class TestStackIsUp:
    async def test_write_api_is_healthy(self, http: httpx.AsyncClient) -> None:
        response = await http.get(f"{WRITE_API_BASE}/health")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"

    async def test_keycloak_published_its_jwks(self, http: httpx.AsyncClient) -> None:
        response = await http.get(f"{KEYCLOAK_BASE}/realms/{REALM}/protocol/openid-connect/certs")
        assert response.status_code == 200
        keys = response.json()["keys"]
        assert any(key.get("alg") == "RS256" for key in keys)


class TestRealTokensAgainstRealApi:
    async def test_drafter_can_write(self, http: httpx.AsyncClient) -> None:
        token = await _client_credentials_token(http, "drafter-sa", os.environ["DRAFTER_SA_SECRET"])
        response = await http.put(
            f"{WRITE_API_BASE}/v1/runs/{RUN_ID}",
            json=_run_payload(),
            headers={"Authorization": f"Bearer {token}"},
        )
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["written"] == "run"
        # The subject is the service account's own user id, not the client id.
        assert body["written_by"]

    async def test_the_write_is_idempotent(self, http: httpx.AsyncClient) -> None:
        """Sending the same checkpoint twice is what resume does."""
        token = await _client_credentials_token(http, "drafter-sa", os.environ["DRAFTER_SA_SECRET"])
        headers = {"Authorization": f"Bearer {token}"}
        first = await http.put(
            f"{WRITE_API_BASE}/v1/runs/{RUN_ID}", json=_run_payload(), headers=headers
        )
        second = await http.put(
            f"{WRITE_API_BASE}/v1/runs/{RUN_ID}", json=_run_payload(), headers=headers
        )
        assert first.status_code == second.status_code == 200
        assert first.json() == second.json()

    async def test_draft_write_round_trip(self, http: httpx.AsyncClient) -> None:
        token = await _client_credentials_token(http, "drafter-sa", os.environ["DRAFTER_SA_SECRET"])
        response = await http.put(
            f"{WRITE_API_BASE}/v1/drafts/{RUN_ID}/{QUESTION_ID}",
            json={
                "question_id": QUESTION_ID,
                "answer_text": "We stage every migration wave behind a rollback gate.",
                "source_ids": ["ans-014"],
                "confidence": 0.81,
                "needs_sme_review": False,
            },
            headers={"Authorization": f"Bearer {token}"},
        )
        assert response.status_code == 200, response.text

    async def test_no_token_is_401(self, http: httpx.AsyncClient) -> None:
        response = await http.put(f"{WRITE_API_BASE}/v1/runs/{RUN_ID}", json=_run_payload())
        assert response.status_code == 401

    async def test_wrong_role_is_403(self, http: httpx.AsyncClient) -> None:
        """triage-sa authenticates fine and is still refused — that is the point."""
        token = await _client_credentials_token(http, "triage-sa", os.environ["TRIAGE_SA_SECRET"])
        response = await http.put(
            f"{WRITE_API_BASE}/v1/runs/{RUN_ID}",
            json=_run_payload(),
            headers={"Authorization": f"Bearer {token}"},
        )
        assert response.status_code == 403

    async def test_evals_sa_can_write_eval_results(self, http: httpx.AsyncClient) -> None:
        token = await _client_credentials_token(http, "evals-sa", os.environ["EVALS_SA_SECRET"])
        response = await http.post(
            f"{WRITE_API_BASE}/v1/eval-results",
            json={
                "scores": [
                    {
                        "git_sha": "integration",
                        "run_id": RUN_ID,
                        "metric": "smoke.write_path",
                        "value": 1.0,
                        "threshold": 1.0,
                        "passed": True,
                    }
                ]
            },
            headers={"Authorization": f"Bearer {token}"},
        )
        assert response.status_code == 200, response.text


class TestTokenShape:
    async def test_tokens_carry_the_api_audience(self, http: httpx.AsyncClient) -> None:
        """If the audience mapper were missing, every request would 401."""
        import jwt

        token = await _client_credentials_token(http, "drafter-sa", os.environ["DRAFTER_SA_SECRET"])
        claims = jwt.decode(token, options={"verify_signature": False})
        audience = claims["aud"]
        audiences = audience if isinstance(audience, list) else [audience]
        assert "rfp-api" in audiences

    async def test_tokens_expire_in_five_minutes(self, http: httpx.AsyncClient) -> None:
        import jwt

        token = await _client_credentials_token(http, "drafter-sa", os.environ["DRAFTER_SA_SECRET"])
        claims = jwt.decode(token, options={"verify_signature": False})
        assert claims["exp"] - claims["iat"] == 300


class TestSubmitterIsUnheldLive:
    """The static check reads the file we ship; this reads the realm that ran."""

    async def test_the_role_exists_in_the_live_realm(self, http: httpx.AsyncClient) -> None:
        admin = await _admin_token(http)
        response = await http.get(
            f"{KEYCLOAK_BASE}/admin/realms/{REALM}/roles",
            headers={"Authorization": f"Bearer {admin}"},
        )
        response.raise_for_status()
        assert "submitter" in {role["name"] for role in response.json()}

    async def test_nobody_holds_it(self, http: httpx.AsyncClient) -> None:
        admin = await _admin_token(http)
        response = await http.get(
            f"{KEYCLOAK_BASE}/admin/realms/{REALM}/roles/submitter/users",
            headers={"Authorization": f"Bearer {admin}"},
        )
        response.raise_for_status()
        holders = [user.get("username") for user in response.json()]
        assert holders == [], f"submitter is held by: {holders}"

    async def test_audit_events_are_enabled_live(self, http: httpx.AsyncClient) -> None:
        admin = await _admin_token(http)
        response = await http.get(
            f"{KEYCLOAK_BASE}/admin/realms/{REALM}/events/config",
            headers={"Authorization": f"Bearer {admin}"},
        )
        response.raise_for_status()
        config = response.json()
        assert config["eventsEnabled"] is True
        assert config["adminEventsEnabled"] is True
