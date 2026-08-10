"""Nothing in this system can auto-submit an RFP response (CLAUDE.md rule 2).

The `submitter` role exists precisely so its emptiness is checkable. These tests
assert it statically against the checked-in realm export; the live counterpart
against a running Keycloak lives in tests/integration/.

The expected role map below is also the documentation of intent — a JSON file
cannot carry comments, so the rationale for who holds what lives here, where it
is executable.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

REALM_PATH = Path(__file__).resolve().parents[2] / "docker" / "keycloak" / "realm-rfp.json"

SUBMITTER = "submitter"

# Every service account and the realm roles it is entitled to. Derived from the
# tool-to-role table (build prompt §11) and the role holders in §16. A role
# appears here only if some tool or endpoint the account actually calls needs it.
#
# Two grants were narrowed after the Phase 1 review:
#   - evals-sa holds `eval-writer`, not `draft-writer`. The harness records
#     metrics; it has no business writing an answer.
#   - assembler-sa holds no `kg-reader`. The assembler is deterministic and
#     reads nothing from the graph. If a later phase needs it, that phase
#     argues for it.
EXPECTED_REALM_ROLES: dict[str, set[str]] = {
    "triage-sa": {"rfp-reader"},
    "extractor-sa": {"rfp-reader"},
    "retriever-sa": {"rfp-reader", "kg-reader"},
    "drafter-sa": {"rfp-reader", "kg-reader", "draft-writer"},
    "critic-sa": {"rfp-reader", "kg-reader"},
    "assembler-sa": {"rfp-reader", "draft-writer"},
    "ingest-sa": {"kg-reader", "kg-writer"},
    "evals-sa": {"rfp-reader", "kg-reader", "eval-writer"},
    "dashboard-sa": {"rfp-reader", "log-reader", "view-events"},
    "loginterp-sa": {"rfp-reader", "log-reader"},
}


@pytest.fixture(scope="module")
def realm() -> dict[str, Any]:
    with REALM_PATH.open(encoding="utf-8") as handle:
        data: dict[str, Any] = json.load(handle)
    return data


def _users(realm: dict[str, Any]) -> list[dict[str, Any]]:
    users: list[dict[str, Any]] = realm.get("users", [])
    return users


class TestSubmitterIsDefinedButUnheld:
    def test_the_role_exists(self, realm: dict[str, Any]) -> None:
        """It must be defined — an undefined role would make the check vacuous."""
        names = {role["name"] for role in realm["roles"]["realm"]}
        assert SUBMITTER in names

    def test_no_user_holds_it_directly(self, realm: dict[str, Any]) -> None:
        holders = [
            user["username"] for user in _users(realm) if SUBMITTER in user.get("realmRoles", [])
        ]
        assert holders == [], f"submitter is granted to: {holders}"

    def test_no_user_holds_it_as_a_client_role(self, realm: dict[str, Any]) -> None:
        holders = [
            user["username"]
            for user in _users(realm)
            for roles in user.get("clientRoles", {}).values()
            if SUBMITTER in roles
        ]
        assert holders == [], f"submitter granted as a client role to: {holders}"

    def test_no_composite_role_smuggles_it_in(self, realm: dict[str, Any]) -> None:
        """A composite containing submitter would grant it without naming it."""
        offenders = []
        for role in realm["roles"]["realm"]:
            composites = role.get("composites", {})
            if SUBMITTER in composites.get("realm", []):
                offenders.append(role["name"])
        assert offenders == [], f"composite roles including submitter: {offenders}"

    def test_it_is_not_a_default_role(self, realm: dict[str, Any]) -> None:
        """A default role is granted to every principal automatically."""
        default_role = realm.get("defaultRole", {})
        composites = default_role.get("composites", {})
        assert SUBMITTER not in composites.get("realm", [])
        assert SUBMITTER not in realm.get("defaultRoles", [])

    def test_no_client_defines_a_shadow_submitter_role(self, realm: dict[str, Any]) -> None:
        """A client-scoped role of the same name would read as submitter in a token."""
        client_roles = realm.get("roles", {}).get("client", {})
        offenders = [
            client
            for client, roles in client_roles.items()
            if any(role.get("name") == SUBMITTER for role in roles)
        ]
        assert offenders == [], f"clients defining a submitter role: {offenders}"


class TestServiceAccounts:
    def test_every_expected_account_exists(self, realm: dict[str, Any]) -> None:
        client_ids = {client["clientId"] for client in realm["clients"]}
        missing = set(EXPECTED_REALM_ROLES) - client_ids
        assert missing == set(), f"missing service-account clients: {sorted(missing)}"

    def test_role_assignments_match_intent(self, realm: dict[str, Any]) -> None:
        """Least privilege is asserted, not assumed — extra grants fail here."""
        actual = {
            user["serviceAccountClientId"]: set(user.get("realmRoles", []))
            for user in _users(realm)
            if "serviceAccountClientId" in user
        }
        assert actual == EXPECTED_REALM_ROLES

    def _holders_of(self, realm: dict[str, Any], role: str) -> set[str]:
        return {
            user["serviceAccountClientId"]
            for user in _users(realm)
            if role in user.get("realmRoles", [])
        }

    def test_kg_writer_is_held_only_by_ingest(self, realm: dict[str, Any]) -> None:
        assert self._holders_of(realm, "kg-writer") == {"ingest-sa"}

    def test_eval_writer_is_held_only_by_the_harness(self, realm: dict[str, Any]) -> None:
        assert self._holders_of(realm, "eval-writer") == {"evals-sa"}

    def test_the_harness_cannot_write_drafts(self, realm: dict[str, Any]) -> None:
        """The whole point of splitting eval-writer out of draft-writer."""
        assert "evals-sa" not in self._holders_of(realm, "draft-writer")

    def test_draft_writer_is_confined_to_who_produces_answers(self, realm: dict[str, Any]) -> None:
        assert self._holders_of(realm, "draft-writer") == {"drafter-sa", "assembler-sa"}

    def test_the_assembler_reads_nothing_from_the_graph(self, realm: dict[str, Any]) -> None:
        """The assembler is deterministic template filling — it needs no graph access."""
        assert "assembler-sa" not in self._holders_of(realm, "kg-reader")

    @pytest.mark.parametrize("client_id", sorted(EXPECTED_REALM_ROLES))
    def test_accounts_are_client_credentials_only(
        self, realm: dict[str, Any], client_id: str
    ) -> None:
        """No browser flow, no password grant — a leaked secret is the only risk surface."""
        client = next(c for c in realm["clients"] if c["clientId"] == client_id)
        assert client["serviceAccountsEnabled"] is True
        assert client["publicClient"] is False
        assert client["standardFlowEnabled"] is False
        assert client["implicitFlowEnabled"] is False
        assert client["directAccessGrantsEnabled"] is False

    @pytest.mark.parametrize("client_id", sorted(EXPECTED_REALM_ROLES))
    def test_every_token_carries_the_api_audience(
        self, realm: dict[str, Any], client_id: str
    ) -> None:
        """write-api and the MCP server validate `aud`; without this they reject everything."""
        client = next(c for c in realm["clients"] if c["clientId"] == client_id)
        mappers = client.get("protocolMappers", [])
        audiences = [
            mapper["config"].get("included.custom.audience")
            for mapper in mappers
            if mapper.get("protocolMapper") == "oidc-audience-mapper"
        ]
        assert "rfp-api" in audiences


class TestRealmHardening:
    def test_audit_events_are_on(self, realm: dict[str, Any]) -> None:
        """The Phase 6 audit table needs history from the first run, not from later."""
        assert realm["eventsEnabled"] is True
        assert realm["adminEventsEnabled"] is True
        assert realm["adminEventsDetailsEnabled"] is True
        assert "CLIENT_LOGIN" in realm["enabledEventTypes"]

    def test_tokens_are_short_lived(self, realm: dict[str, Any]) -> None:
        assert realm["accessTokenLifespan"] == 300

    def test_signature_algorithm_is_rs256(self, realm: dict[str, Any]) -> None:
        assert realm["defaultSignatureAlgorithm"] == "RS256"

    def test_descriptions_fit_keycloaks_column(self, realm: dict[str, Any]) -> None:
        """Keycloak stores descriptions in varchar(255).

        An over-long one does not truncate — it aborts the whole realm import
        and the container never becomes healthy, which surfaces as a confusing
        startup failure rather than as a validation error. Cheaper to catch here.
        """
        too_long = [
            (role["name"], len(role["description"]))
            for role in realm["roles"]["realm"]
            if len(role.get("description", "")) > 255
        ] + [
            (client["clientId"], len(client["description"]))
            for client in realm["clients"]
            if len(client.get("description", "")) > 255
        ]
        assert too_long == [], f"descriptions exceeding 255 chars: {too_long}"
