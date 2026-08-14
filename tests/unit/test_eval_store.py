"""Flattening categories into rows, and differencing against a previous SHA.

The database itself is not exercised here — that is
`tests/integration/test_live_stack.py`, which writes through the real write-api
as `evals-sa`. What these cover is the layer above it, where the decisions live:
which metrics become rows, how they are named, and what a delta means when one
side is missing.
"""

from __future__ import annotations

import subprocess

import pytest

from src.evals.contracts import (
    CategoryResult,
    CategoryStatus,
    EvalMetric,
    MetricDirection,
)
from src.evals.registry import all_placeholders
from src.evals.store import (
    Delta,
    EvalStoreError,
    deltas,
    git_sha,
    previous_scores,
    to_scores,
)


def category(key: str = "extraction", **overrides: object) -> CategoryResult:
    metrics = [
        EvalMetric(
            key="recall",
            label="Recall",
            value=1.0,
            direction=MetricDirection.HIGHER_IS_BETTER,
            threshold=0.95,
            passed=True,
        ),
        EvalMetric(
            key="mrr",
            label="Mean reciprocal rank",
            value=0.9333,
            direction=MetricDirection.HIGHER_IS_BETTER,
        ),
    ]
    base: dict[str, object] = {
        "key": key,
        "label": key.title(),
        "status": CategoryStatus.PASS,
        "metrics": metrics,
    }
    base.update(overrides)
    return CategoryResult(**base)  # type: ignore[arg-type]


class TestOnlyGatedMetricsBecomeRows:
    def test_a_gated_metric_is_persisted(self) -> None:
        scores = to_scores([category()], sha="abc", run_id="r1")
        assert [score.metric for score in scores] == ["extraction.recall"]

    def test_an_ungated_metric_is_not(self) -> None:
        """MRR has no threshold. Writing it with an invented 0.0 would make it
        look gated forever after, to everyone reading the table."""
        scores = to_scores([category()], sha="abc", run_id="r1")
        assert all("mrr" not in score.metric for score in scores)

    def test_placeholders_contribute_nothing(self) -> None:
        """A placeholder row would be differenced against the next SHA and turn
        'not measured' into an apparent regression the day it is measured."""
        assert to_scores(all_placeholders(), sha="abc", run_id="r1") == []


class TestMetricNamesAreNamespaced:
    def test_the_category_prefixes_the_metric(self) -> None:
        scores = to_scores([category()], sha="abc", run_id="r1")
        assert scores[0].metric == "extraction.recall"

    def test_two_categories_with_the_same_metric_key_do_not_collide(self) -> None:
        """The primary key is (git_sha, run_id, metric), so an unqualified
        `recall` from two categories would overwrite one with the other."""
        scores = to_scores([category("extraction"), category("retrieval")], sha="abc", run_id="r1")
        assert {score.metric for score in scores} == {
            "extraction.recall",
            "retrieval.recall",
        }
        assert len(scores) == 2


class TestRowsCarryTheirProvenance:
    def test_every_row_carries_the_sha(self) -> None:
        scores = to_scores([category()], sha="abc123", run_id="r1")
        assert all(score.git_sha == "abc123" for score in scores)

    def test_every_row_carries_the_run_id(self) -> None:
        scores = to_scores([category()], sha="abc123", run_id="evals-xyz")
        assert all(score.run_id == "evals-xyz" for score in scores)

    def test_the_verdict_is_carried_not_recomputed(self) -> None:
        """`passed` travels with the row so a threshold that moves later cannot
        silently reinterpret a measurement already taken."""
        failing = category(
            metrics=[
                EvalMetric(
                    key="recall",
                    label="Recall",
                    value=0.5,
                    direction=MetricDirection.HIGHER_IS_BETTER,
                    threshold=0.95,
                    passed=False,
                )
            ]
        )
        assert to_scores([failing], sha="abc", run_id="r1")[0].passed is False


def git_can_answer() -> str | None:
    """HEAD as git sees it from the test's cwd, or None if git cannot say.

    Not a formality. `git rev-parse` genuinely fails in some environments this
    suite must stay green in — a `git worktree` created by Windows git and run
    under WSL has a `.git` file pointing at a `C:/…` gitdir the Linux git cannot
    resolve, and a source tarball has no repository at all. Where git cannot
    answer, "prefer git over the environment variable" is not a property that
    exists to be tested.
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],  # noqa: S607
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()


class TestTheShaComesFromGitInACheckout:
    """One env var name, two different questions.

    `.env` sets `GIT_SHA=local-dev` so a CONTAINER can report its build. The
    harness asks a different question — which commit produced these numbers —
    and the first version answered it with the container's variable, keying
    every local run to `local-dev` and making the SHA-keyed scoreboard useless.
    """

    @pytest.fixture
    def head(self) -> str:
        resolved = git_can_answer()
        if resolved is None:
            pytest.skip("git cannot resolve HEAD here; the precedence has nothing to rank")
        return resolved

    def test_the_container_variable_does_not_override_the_real_commit(
        self, monkeypatch: pytest.MonkeyPatch, head: str
    ) -> None:
        monkeypatch.setenv("GIT_SHA", "local-dev")
        assert git_sha() != "local-dev"

    def test_it_returns_the_head_this_checkout_is_on(
        self, monkeypatch: pytest.MonkeyPatch, head: str
    ) -> None:
        monkeypatch.delenv("GIT_SHA", raising=False)
        assert git_sha().startswith(head)

    def test_a_dirty_worktree_is_marked(self, monkeypatch: pytest.MonkeyPatch, head: str) -> None:
        """A number measured against uncommitted code is not a number about that
        commit, and a scoreboard attributing it to one is worse than a gap."""
        monkeypatch.delenv("GIT_SHA", raising=False)
        dirty = subprocess.run(
            ["git", "status", "--porcelain"],  # noqa: S607
            capture_output=True,
            text=True,
            check=False,
        ).stdout.strip()
        assert git_sha().endswith("-dirty") is bool(dirty)


class TestTheFallbackWhenGitCannotAnswer:
    """The other branch, forced rather than waited for.

    These run everywhere, including where git works, because the fallback is
    reached in containers and source tarballs — environments this suite does not
    execute in. Making the failure happen is the only way to cover it.
    """

    @pytest.fixture
    def git_is_broken(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def explode(*_args: object, **_kwargs: object) -> object:
            raise OSError("git is not available here")

        monkeypatch.setattr("src.evals.store.subprocess.run", explode)

    def test_the_env_var_is_used_when_git_cannot_answer(
        self, monkeypatch: pytest.MonkeyPatch, git_is_broken: None
    ) -> None:
        """Outside a checkout GIT_SHA is the only answer available — and there
        it is the right one. The precedence is about which to prefer, not about
        refusing the other."""
        monkeypatch.setenv("GIT_SHA", "deadbeef")
        assert git_sha() == "deadbeef"

    def test_it_refuses_when_neither_is_available(
        self, monkeypatch: pytest.MonkeyPatch, git_is_broken: None
    ) -> None:
        """Rows are keyed by SHA, so an unknown SHA is not something to guess at."""
        monkeypatch.delenv("GIT_SHA", raising=False)
        with pytest.raises(EvalStoreError, match="cannot determine the git SHA"):
            git_sha()


class TestAnUnreachableBaselineDegradesRatherThanCrashes:
    """The lookup runs AFTER every expensive measurement.

    A wrong driver name in the connection URL once took down a 35-minute
    rerank-ON run at its final step, with all the work done and nothing written
    — because the engine was constructed above the handler rather than inside
    it. Anything that can go wrong reaching the database belongs to the "report
    says no baseline" path, not to the "lose the run" path.
    """

    async def test_an_unusable_url_returns_no_baseline(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("POSTGRES_HOST", "no-such-host.invalid")
        monkeypatch.setenv("POSTGRES_PORT", "1")
        monkeypatch.setenv("APP_READER_PASSWORD", "irrelevant")
        assert await previous_scores(current_sha="a" * 40) == (None, {})

    async def test_an_unknown_driver_returns_no_baseline(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The exact shape of the original defect: a driver that cannot import.

        Raised past the handler before the fix, because `create_async_engine`
        resolves the dialect at construction and construction was outside it.
        """
        monkeypatch.setattr(
            "src.state.db.reader_url",
            lambda **_: "postgresql+notadriver://u:p@localhost:1/db",
        )
        assert await previous_scores(current_sha="a" * 40) == (None, {})

    async def test_the_degradation_is_logged(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Quietly is fine; invisibly is not."""
        monkeypatch.setattr(
            "src.state.db.reader_url",
            lambda **_: "postgresql+notadriver://u:p@localhost:1/db",
        )
        with caplog.at_level("WARNING", logger="rfp.evals.store"):
            await previous_scores(current_sha="a" * 40)
        assert any("baseline unavailable" in record.message for record in caplog.records)


class TestDeltas:
    def test_a_metric_in_both_runs_gets_a_delta(self) -> None:
        scores = to_scores([category()], sha="abc", run_id="r1")
        movements = deltas(scores, {"extraction.recall": 0.9})
        assert len(movements) == 1
        assert movements[0].change == pytest.approx(0.1)

    def test_a_new_metric_gets_no_delta(self) -> None:
        """Differencing against an implied zero would show every new metric as a
        large improvement on the day it was introduced."""
        scores = to_scores([category()], sha="abc", run_id="r1")
        assert deltas(scores, {}) == []

    def test_a_dropped_metric_is_not_invented(self) -> None:
        scores = to_scores([category()], sha="abc", run_id="r1")
        movements = deltas(scores, {"extraction.recall": 0.9, "extraction.gone": 1.0})
        assert [m.metric for m in movements] == ["extraction.recall"]

    def test_change_is_current_minus_previous(self) -> None:
        assert Delta(metric="m", previous=1.0, current=0.75).change == pytest.approx(-0.25)
