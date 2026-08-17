"""Structured logging.

Driver logs on a job cluster are shipped to CloudWatch/Datadog and parsed
there, so records are emitted as single-line JSON. A human tailing the log
still gets something readable; a log aggregator gets fields it can index.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from contextlib import contextmanager
from typing import Any

_RESERVED = set(logging.LogRecord("", 0, "", 0, "", (), None).__dict__) | {
    "asctime",
    "message",
    "taskName",
}


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        extras = {k: v for k, v in record.__dict__.items() if k not in _RESERVED}
        payload.update(extras)
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def configure_logging(level: str = "INFO", *, json_output: bool = True) -> logging.Logger:
    """Install the platform log configuration and return the root logger."""
    root = logging.getLogger()
    root.setLevel(level.upper())
    for handler in list(root.handlers):
        root.removeHandler(handler)

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        JsonFormatter()
        if json_output
        else logging.Formatter("%(asctime)s %(levelname)-8s %(name)s :: %(message)s")
    )
    root.addHandler(handler)

    # py4j logs every JVM call at INFO. It is never what you are looking for.
    logging.getLogger("py4j").setLevel(logging.WARNING)
    logging.getLogger("py4j.java_gateway").setLevel(logging.ERROR)
    return root


@contextmanager
def stage(logger: logging.Logger, name: str, **context: Any):
    """Log the start, duration and outcome of a pipeline stage."""
    logger.info("stage.start", extra={"stage": name, **context})
    started = time.monotonic()
    try:
        yield
    except Exception:
        logger.exception(
            "stage.failed",
            extra={"stage": name, "duration_s": round(time.monotonic() - started, 3), **context},
        )
        raise
    logger.info(
        "stage.complete",
        extra={"stage": name, "duration_s": round(time.monotonic() - started, 3), **context},
    )
