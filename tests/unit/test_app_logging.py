"""The handler wiring that makes `logger.info` reach a container's stdout.

These assert the properties the found defect violated, one per test. None of
them can prove the deployed service logs — that is
`tests/integration/test_mcp_server.py::TestTheCallerIsInTheLog`, and the split is
deliberate: an in-process test of logging configuration passes just as happily
when uvicorn discards the output, which is precisely how the defect survived.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator

import pytest

from src.observability.logging import APP_LOGGER, configure_app_logging


@pytest.fixture(autouse=True)
def _fresh_app_logger() -> Iterator[None]:
    """Reset the `rfp` tree around every test.

    `configure_app_logging` is idempotent BY DESIGN — a second call must not
    double every line — and a `StreamHandler` binds its stream at construction.
    Both are right for a long-lived process and wrong for a suite where `capsys`
    swaps `sys.stdout` per test: without this reset, the handler built in the
    first test keeps writing to the first test's captured stream.

    Restoring rather than merely clearing, so this module cannot leave the app
    logger in a state that changes how another module's tests behave.
    """
    logger = logging.getLogger(APP_LOGGER)
    saved_handlers = list(logger.handlers)
    saved_level, saved_propagate = logger.level, logger.propagate
    logger.handlers.clear()
    yield
    logger.handlers[:] = saved_handlers
    logger.setLevel(saved_level)
    logger.propagate = saved_propagate


class TestItAttachesAHandler:
    def test_the_app_logger_gains_one(self) -> None:
        logger = configure_app_logging("INFO")
        assert logger.handlers, "an application logger with no handler drops INFO records"

    def test_calling_twice_does_not_double_it(self) -> None:
        """A reload, or a second service in one process, must not duplicate lines."""
        first = configure_app_logging("INFO")
        before = len(first.handlers)
        second = configure_app_logging("INFO")
        assert len(second.handlers) == before

    def test_it_does_not_propagate_to_the_root(self) -> None:
        """The tree owns a handler, so propagation as well would print twice."""
        assert configure_app_logging("INFO").propagate is False

    def test_it_configures_the_app_tree_not_the_root_logger(self) -> None:
        """Configuring the actual root would capture every library's output too."""
        logger = configure_app_logging("INFO")
        assert logger.name == APP_LOGGER
        assert logger is not logging.getLogger()


class TestItRespectsTheConfiguredLevel:
    def test_info_is_emitted_at_info(self, capsys: object) -> None:
        configure_app_logging("INFO")
        logging.getLogger("rfp.test").info("audit-line-marker")
        captured = capsys.readouterr()  # type: ignore[attr-defined]
        assert "audit-line-marker" in captured.out

    def test_info_is_suppressed_at_warning(self, capsys: object) -> None:
        configure_app_logging("WARNING")
        logging.getLogger("rfp.test").info("should-not-appear")
        captured = capsys.readouterr()  # type: ignore[attr-defined]
        assert "should-not-appear" not in captured.out

    def test_the_level_is_read_case_insensitively(self) -> None:
        """`.env` carries INFO; a caller passing `info` means the same thing."""
        assert configure_app_logging("info").level == logging.INFO

    def test_it_falls_back_to_the_environment(self, monkeypatch: object) -> None:
        monkeypatch.setenv("LOG_LEVEL", "DEBUG")  # type: ignore[attr-defined]
        assert configure_app_logging().level == logging.DEBUG


class TestItWritesToStdout:
    def test_records_go_to_stdout_not_stderr(self, capsys: object) -> None:
        """A collector treating stderr as errors would flag every successful call."""
        configure_app_logging("INFO")
        logging.getLogger("rfp.test").info("stream-check")
        captured = capsys.readouterr()  # type: ignore[attr-defined]
        assert "stream-check" in captured.out
        assert "stream-check" not in captured.err


class TestTheMcpLoggerIsUnderTheConfiguredTree:
    def test_the_mcp_logger_name_is_covered(self) -> None:
        """`rfp.mcp` must sit under `rfp`, or the handler covers nothing.

        The one assertion connecting this module to the audit line it exists
        for: renaming either side breaks the wiring silently otherwise.
        """
        from src.mcp_server.app import logger as mcp_logger

        assert mcp_logger.name.startswith(f"{APP_LOGGER}.")
