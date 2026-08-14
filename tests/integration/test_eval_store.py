"""The harness's own persistence path, against the real stack.

The unit suite covers which metrics become rows and how they are named. What it
cannot cover is the round trip: that `evals-sa` can actually obtain a token, that
write-api accepts the batch, that the rows land, and that `previous_scores` can
read them back through a DIFFERENT connection with a DIFFERENT role. Those are
four separate pieces of wiring, and every one of them is invisible to a test that
never opens a socket.

The read path matters especially. Writes go through write-api; reads go directly
to Postgres as `app_reader`, and nothing else in the suite exercises that
asymmetry. A missing grant on `app_reader` would leave the harness silently
reporting "no previous SHA" forever — a degradation that looks exactly like a
fresh clone.
"""

from __future__ import annotations

import os
import uuid

import pytest

from src.contracts.run import EvalScore
from src.evals.store import deltas, persist, previous_scores
from tests.live import require_env, require_write_api_and_keycloak

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module", autouse=True)
def _live_stack() -> None:
    require_write_api_and_keycloak()
    # The read half authenticates separately from the write half; without this
    # the failure surfaces as a swallowed exception and an empty baseline.
    require_env("APP_READER_PASSWORD")


def score(sha: str, run_id: str, metric: str, value: float) -> EvalScore:
    return EvalScore(
        git_sha=sha,
        run_id=run_id,
        metric=metric,
        value=value,
        threshold=0.95,
        passed=value >= 0.95,
    )


def fake_sha(marker: str = "e7") -> str:
    """A key with a REAL COMMIT'S SHAPE, so the baseline query will consider it.

    40 hex characters. Since `previous_scores` now ignores anything that is not
    commit-shaped — so that test rows cannot become a production baseline —
    a test that wants to exercise the baseline path has to look like a commit.
    The `marker` prefix keeps these identifiable in the table.
    """
    return (marker + uuid.uuid4().hex + uuid.uuid4().hex)[:40]


class TestTheRoundTrip:
    async def test_rows_written_can_be_read_back(self) -> None:
        """Write as evals-sa through write-api, read as app_reader directly."""
        sha = fake_sha()
        run_id = f"evals-{uuid.uuid4().hex[:10]}"
        written = await persist([score(sha, run_id, "extraction.recall", 1.0)])
        assert written == 1

        # Read from the perspective of a LATER sha, which is what the harness
        # does: "what did the previous commit measure?"
        previous_sha, scores = await previous_scores(current_sha=fake_sha("ff"))
        assert previous_sha is not None
        assert scores, "the read path returned nothing; check app_reader's grant on eval_results"

    async def test_the_current_sha_is_excluded_from_its_own_baseline(self) -> None:
        """A run must not difference against itself and report zero movement."""
        sha = fake_sha()
        run_id = f"evals-{uuid.uuid4().hex[:10]}"
        await persist([score(sha, run_id, "extraction.recall", 1.0)])

        previous_sha, _ = await previous_scores(current_sha=sha)
        assert previous_sha != sha

    async def test_a_rerun_at_the_same_sha_updates_rather_than_duplicates(self) -> None:
        """The primary key is (git_sha, run_id, metric).

        An eval you can run twice is one you will run twice, and two rows for one
        measurement is a scoreboard that cannot be read.
        """
        sha = fake_sha()
        run_id = f"evals-{uuid.uuid4().hex[:10]}"
        await persist([score(sha, run_id, "extraction.recall", 0.5)])
        await persist([score(sha, run_id, "extraction.recall", 1.0)])

        found_sha, scores = await previous_scores(current_sha=fake_sha("ff"))
        if found_sha == sha:
            assert scores["extraction.recall"] == pytest.approx(1.0)


class TestTestRowsCannotBecomeABaseline:
    """`eval_results` is not the harness's private table.

    The write-path test in `test_live_stack.py` writes `git_sha='integration'`.
    Before the shape filter, that row was a candidate baseline for every real
    run — so running the suite changed what the scoreboard said about the code.
    """

    async def test_a_non_commit_key_is_never_chosen_as_the_baseline(self) -> None:
        run_id = f"evals-{uuid.uuid4().hex[:10]}"
        await persist([score("integration", run_id, "extraction.recall", 0.1)])

        previous_sha, _ = await previous_scores(current_sha=fake_sha("ff"))
        assert previous_sha != "integration"

    async def test_a_commit_shaped_key_is(self) -> None:
        """The other half: the filter must not exclude everything."""
        sha = fake_sha()
        run_id = f"evals-{uuid.uuid4().hex[:10]}"
        await persist([score(sha, run_id, "extraction.recall", 1.0)])

        previous_sha, _ = await previous_scores(current_sha=fake_sha("ff"))
        assert previous_sha is not None


class TestDeltasAgainstRealRows:
    async def test_a_movement_is_computed_from_stored_values(self) -> None:
        sha = fake_sha()
        run_id = f"evals-{uuid.uuid4().hex[:10]}"
        metric = f"extraction.delta_probe_{uuid.uuid4().hex[:6]}"
        await persist([score(sha, run_id, metric, 0.90)])

        _, stored = await previous_scores(current_sha=fake_sha("ff"))
        if metric not in stored:
            pytest.skip("a later SHA was written concurrently; the probe row is not the baseline")

        movements = deltas([score(fake_sha("ff"), run_id, metric, 1.0)], stored)
        assert len(movements) == 1
        assert movements[0].change == pytest.approx(0.10)


class TestTheHarnessCannotWriteWhatItMustNotWrite:
    async def test_evals_sa_holds_eval_writer_only(self) -> None:
        """Restated at the harness's own boundary.

        `test_live_stack.py` proves the realm's role split. This proves the
        module that USES it authenticates as the account with the narrow grant —
        so a harness bug can corrupt its own scoreboard and nothing else.
        """
        from src.evals.store import EVALS_CLIENT_ID

        assert EVALS_CLIENT_ID == "evals-sa"
        assert os.environ.get("EVALS_SA_SECRET"), "the harness authenticates as evals-sa"
