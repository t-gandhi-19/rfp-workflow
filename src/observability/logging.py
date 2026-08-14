"""Application logging that actually reaches a handler under uvicorn.

WHY THIS EXISTS. `logging.getLogger("rfp.mcp").info(...)` writes nothing in a
uvicorn container. Uvicorn installs its own dictConfig, which attaches handlers
to the `uvicorn*` loggers and leaves the ROOT logger bare; an application logger
propagates to a root with no handlers, and Python's last-resort handler passes
only WARNING and above. So every `logger.info` in the codebase is discarded, in
silence, in exactly the environment it was written for.

That is not a cosmetic loss here. mcp-server's per-call line is the audit trail
for graph reads — the record of which caller asked for what, which is the only
evidence available after the fact that a retrieval did or did not surface
confidential material. An audit trail that exists in the source and not in the
container is worse than none, because it reads as though the question has been
answered.

Found by `TestTheCallerIsInTheLog`, which reads the container's own log rather
than an in-process capture. A `caplog` test would have passed throughout: the
logging call is present and correct, and it is the *handler wiring around it*
that was missing. That is the whole argument for asserting against the deployed
service.
"""

from __future__ import annotations

import logging
import os
import sys

#: The root of the application logger tree. Every module logs under `rfp.*`, so
#: one handler here covers all of them and nothing else — configuring the actual
#: root logger instead would also capture every library's output.
APP_LOGGER = "rfp"

#: Matches uvicorn's default so a container's log reads as one stream rather than
#: two interleaved formats.
FORMAT = "%(levelname)-8s %(name)s: %(message)s"


def configure_app_logging(level: str | None = None) -> logging.Logger:
    """Attach a stdout handler to the `rfp` logger tree.

    Call from a service's lifespan, not at import: uvicorn configures logging
    when it starts the server, which is *after* the app module is imported, and
    a handler installed at import time is one `dictConfig` away from being
    dropped.

    Idempotent — a reload, or a second service in the same process, must not
    double every line.
    """
    resolved = (level or os.environ.get("LOG_LEVEL") or "INFO").upper()
    logger = logging.getLogger(APP_LOGGER)
    logger.setLevel(resolved)

    # stdout, not stderr: these are ordinary operational records, and a log
    # collector that treats stderr as errors would flag every successful call.
    if not any(getattr(h, "_rfp_app_handler", False) for h in logger.handlers):
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(FORMAT))
        handler._rfp_app_handler = True  # type: ignore[attr-defined]
        logger.addHandler(handler)

    # The `rfp` tree carries its own handler, so propagating as well would print
    # each record twice wherever a root handler does exist — under pytest, for
    # one, which is where the duplication would be noticed last.
    logger.propagate = False
    return logger
