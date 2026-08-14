"""A dropped keep-alive connection is retried once. Nothing else is.

WHY THIS IS NOT A GENERAL RETRY POLICY. A run holds one HTTP client across a
fan-out that can take many minutes, and an idle keep-alive connection gets
closed by the server or an intermediary in that time. httpx surfaces that as
`Server disconnected without sending a response` on the next use — a TRANSPORT
failure, not a refusal, and one that succeeds immediately on a fresh connection.

A 4xx is a refusal that will be refused again, and retrying it would spend the
budget to be told the same thing twice. So only `TransportError` is retried, and
only once.

Re-sending is safe because every write here is an idempotent upsert on a natural
key: a PUT that may already have landed cannot double-write.
"""

from __future__ import annotations

from datetime import UTC, datetime

import httpx
import pytest

from src.contracts import QuestionStatus, RunStage, RunState
from src.controller.checkpoint import Checkpointer, CheckpointError

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


def checkpointer() -> Checkpointer:
    """Pre-authenticated, so the tests exercise the PUT rather than Keycloak."""
    return Checkpointer(base_url="http://write-api", _token="a-token")


def client_for(handler: object) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))  # type: ignore[arg-type]


def _run_state() -> RunState:
    now = datetime.now(UTC)
    return RunState(
        run_id="run-1", rfp_id="rfp-1", stage=RunStage.DRAFTING, started_at=now, updated_at=now
    )


class TestADroppedConnectionIsRetried:
    async def test_the_second_attempt_succeeds(self) -> None:
        attempts = {"n": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            attempts["n"] += 1
            if attempts["n"] == 1:
                raise httpx.ReadError("Server disconnected without sending a response.")
            return httpx.Response(200, json={"written": "x", "key": "k", "written_by": "s"})

        async with client_for(handler) as client:
            await checkpointer().save_question_status(
                run_id="run-1",
                question_id="GQ-001",
                status=QuestionStatus.ESCALATED,
                client=client,
            )
        assert attempts["n"] == 2

    async def test_a_second_drop_gives_up_with_a_readable_error(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ReadError("Server disconnected without sending a response.")

        async with client_for(handler) as client:
            with pytest.raises(CheckpointError, match="unreachable"):
                await checkpointer().save_question_status(
                    run_id="run-1",
                    question_id="GQ-001",
                    status=QuestionStatus.ESCALATED,
                    client=client,
                )

    async def test_it_does_not_retry_forever(self) -> None:
        attempts = {"n": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            attempts["n"] += 1
            raise httpx.ConnectError("connection refused")

        async with client_for(handler) as client:
            with pytest.raises(CheckpointError):
                await checkpointer().save_question_status(
                    run_id="run-1",
                    question_id="GQ-001",
                    status=QuestionStatus.ESCALATED,
                    client=client,
                )
        assert attempts["n"] == 2


class TestAnExpiredTokenIsReMinted:
    """A Keycloak access token lives minutes; a run lives as long as its fan-out.

    Caching one for the whole run guaranteed a 401 on any run slower than the
    token lifetime — which is every real run. A 14-minute local-tier run died on
    `Invalid token: Signature has expired`, and because the halt path
    checkpoints too, the halt died with it.

    This is NOT "retrying a refusal": the credential was valid and went stale,
    and the second attempt carries a NEW one rather than the same one again.
    """

    def keycloak_and_writes(self, tokens: list[str]) -> tuple[object, dict[str, list[str]]]:
        """A transport that mints from `tokens` and 401s anything but the last."""
        seen: dict[str, list[str]] = {"tokens_issued": [], "auth_headers": []}

        def handler(request: httpx.Request) -> httpx.Response:
            if "openid-connect/token" in str(request.url):
                issued = tokens[len(seen["tokens_issued"])]
                seen["tokens_issued"].append(issued)
                return httpx.Response(200, json={"access_token": issued})
            header = request.headers.get("authorization", "")
            seen["auth_headers"].append(header)
            if header != f"Bearer {tokens[-1]}":
                return httpx.Response(401, json={"detail": "Invalid token: Signature has expired"})
            return httpx.Response(200, json={"written": "x", "key": "k", "written_by": "s"})

        return handler, seen

    async def test_a_401_re_authenticates_and_succeeds(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("DRAFTER_SA_SECRET", "s")
        handler, seen = self.keycloak_and_writes(["stale-token", "fresh-token"])
        checkpointer = Checkpointer(base_url="http://write-api")
        async with client_for(handler) as client:
            await checkpointer.save_question_status(
                run_id="run-1",
                question_id="GQ-001",
                status=QuestionStatus.COMPLETE,
                client=client,
            )
        assert seen["tokens_issued"] == ["stale-token", "fresh-token"]
        assert seen["auth_headers"] == ["Bearer stale-token", "Bearer fresh-token"]

    async def test_the_second_attempt_carries_a_different_token(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The point. Retrying with the SAME stale token would 401 again and
        would be indistinguishable from retrying a refusal."""
        monkeypatch.setenv("DRAFTER_SA_SECRET", "s")
        handler, seen = self.keycloak_and_writes(["stale-token", "fresh-token"])
        checkpointer = Checkpointer(base_url="http://write-api")
        async with client_for(handler) as client:
            await checkpointer.save_run(_run_state(), client=client)
        assert len(set(seen["auth_headers"])) == 2

    async def test_a_persistent_401_gives_up_rather_than_looping(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("DRAFTER_SA_SECRET", "s")
        attempts = {"n": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            if "openid-connect/token" in str(request.url):
                return httpx.Response(200, json={"access_token": "never-good"})
            attempts["n"] += 1
            return httpx.Response(401, json={"detail": "Invalid token"})

        async with client_for(handler) as client:
            with pytest.raises(CheckpointError, match="refused"):
                await Checkpointer(base_url="http://write-api").save_question_status(
                    run_id="run-1",
                    question_id="GQ-001",
                    status=QuestionStatus.COMPLETE,
                    client=client,
                )
        assert attempts["n"] == 2


class TestARefusalIsNotRetried:
    @pytest.mark.parametrize("status_code", [400, 403, 422, 500])
    async def test_a_status_code_is_returned_on_the_first_attempt(self, status_code: int) -> None:
        """A refusal will be refused again. Retrying spends the budget to be
        told the same thing twice."""
        attempts = {"n": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            attempts["n"] += 1
            return httpx.Response(status_code, text="no")

        async with client_for(handler) as client:
            with pytest.raises(CheckpointError, match="refused"):
                await checkpointer().save_question_status(
                    run_id="run-1",
                    question_id="GQ-001",
                    status=QuestionStatus.ESCALATED,
                    client=client,
                )
        assert attempts["n"] == 1


class TestTheHappyPathIsUnchanged:
    async def test_one_attempt_when_nothing_drops(self) -> None:
        attempts = {"n": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            attempts["n"] += 1
            return httpx.Response(200, json={"written": "x", "key": "k", "written_by": "s"})

        async with client_for(handler) as client:
            await checkpointer().save_question_status(
                run_id="run-1",
                question_id="GQ-001",
                status=QuestionStatus.COMPLETE,
                client=client,
            )
        assert attempts["n"] == 1
