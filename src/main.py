"""FastAPI application entrypoint for the Apollo Research Agent."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator

import structlog
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.routes import router as api_router
from src.config import get_settings


def _configure_logging() -> None:
    """Configure structlog with processors for structured JSON logging."""
    settings = get_settings()

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    raw_level = getattr(settings, "LOG_LEVEL", getattr(settings, "log_level", "INFO"))
    log_level = getattr(logging, str(raw_level).upper(), logging.INFO)
    logging.basicConfig(
        format="%(message)s",
        level=log_level,
        force=True,
    )


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan handler for startup and shutdown events.

    Configures structured logging on startup.

    Args:
        app: The FastAPI application instance.
    """
    _configure_logging()
    logger = structlog.get_logger()
    logger.info("Apollo Research Agent starting up", version="1.0.0")
    yield
    logger.info("Apollo Research Agent shutting down")


app = FastAPI(
    title="Apollo Research Agent",
    version="1.0.0",
    description=(
        "Apollo Deep Research Agent — an AI-powered multi-agent system that "
        "produces comprehensive, citation-backed research reports using "
        "iterative planning, search, analysis, and synthesis."
    ),
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router)


@app.get("/health")
async def health_check() -> dict[str, str]:
    """Health check endpoint.

    Returns:
        A JSON object with status and version information.
    """
    return {"status": "healthy", "version": "1.0.0"}


if __name__ == "__main__":
    uvicorn.run("src.main:app", host="0.0.0.0", port=8000, reload=True)
