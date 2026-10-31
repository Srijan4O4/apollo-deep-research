"""Structured logging configuration using *structlog*.

Provides a single :func:`get_logger` entry-point that returns a *bound*
logger pre-stamped with the caller's module name.  Two rendering modes are
supported:

* **Development** — coloured, human-readable console output.
* **Production** — machine-parseable JSON lines (activated when
  ``LOG_LEVEL`` is *not* ``DEBUG``).

Context-binding helpers :func:`bind_task_context` and
:func:`bind_agent_context` make it trivial to attach ``task_id`` and
``agent_name`` to every subsequent log line emitted by a logger.
"""

from __future__ import annotations

import logging
import sys

import structlog

from src.config import get_settings


def _configure_structlog() -> None:
    """Set up structlog processors and the stdlib integration once."""

    settings = get_settings()
    is_debug = settings.LOG_LEVEL.upper() == "DEBUG"

    shared_processors: list[structlog.types.Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.CallsiteParameterAdder(
            [
                structlog.processors.CallsiteParameter.MODULE,
                structlog.processors.CallsiteParameter.FUNC_NAME,
                structlog.processors.CallsiteParameter.LINENO,
            ],
        ),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.UnicodeDecoder(),
    ]

    if is_debug:
        renderer: structlog.types.Processor = structlog.dev.ConsoleRenderer(
            colors=True,
        )
    else:
        renderer = structlog.processors.JSONRenderer()

    structlog.configure(
        processors=[
            *shared_processors,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    # Wire a stdlib handler so that structlog events are actually emitted.
    formatter = structlog.stdlib.ProcessorFormatter(
        processor=renderer,
        foreign_pre_chain=shared_processors,
    )

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.addHandler(handler)
    root_logger.setLevel(settings.LOG_LEVEL.upper())


# Run configuration on first import.
_configure_structlog()


# ── Public API ─────────────────────────────────────────────────────────────


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """Return a :class:`structlog.stdlib.BoundLogger` bound to *name*.

    Parameters
    ----------
    name:
        Typically ``__name__`` of the calling module.

    Returns
    -------
    structlog.stdlib.BoundLogger
        A logger with ``module=name`` already bound in context.
    """
    return structlog.get_logger(name)


def bind_task_context(logger: structlog.stdlib.BoundLogger, task_id: str) -> structlog.stdlib.BoundLogger:
    """Return a new logger with ``task_id`` bound.

    Parameters
    ----------
    logger:
        The existing bound logger to extend.
    task_id:
        Unique identifier for the research task.

    Returns
    -------
    structlog.stdlib.BoundLogger
        A child logger carrying the ``task_id`` key.
    """
    return logger.bind(task_id=task_id)


def bind_agent_context(logger: structlog.stdlib.BoundLogger, agent_name: str) -> structlog.stdlib.BoundLogger:
    """Return a new logger with ``agent_name`` bound.

    Parameters
    ----------
    logger:
        The existing bound logger to extend.
    agent_name:
        Human-readable name of the agent (e.g. ``"planner"``, ``"searcher"``).

    Returns
    -------
    structlog.stdlib.BoundLogger
        A child logger carrying the ``agent_name`` key.
    """
    return logger.bind(agent_name=agent_name)
