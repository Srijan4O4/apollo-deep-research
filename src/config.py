"""Centralized configuration loaded from environment / ``.env`` file.

Uses *pydantic-settings* so that every value can be overridden via an
environment variable of the same name.  A cached singleton is exposed
through :func:`get_settings` to avoid re-reading the file on every call.
"""

from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application-wide configuration for the Apollo research system."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── LLM provider keys ──────────────────────────────────────────────
    GEMINI_API_KEY: str = ""
    OPENAI_API_KEY: str = ""

    # ── Search provider keys ───────────────────────────────────────────
    TAVILY_API_KEY: str = ""

    # ── Model selection ────────────────────────────────────────────────
    PRIMARY_MODEL: str = "gemini-flash-lite-latest"
    FALLBACK_MODEL: str = "gemini-flash-latest"

    # ── Workflow limits ────────────────────────────────────────────────
    MAX_ITERATIONS_PER_SECTION: int = 5
    AGENT_TIMEOUT_SECONDS: int = 120
    MAX_TOKEN_BUDGET: int = 100_000

    # ── Quality thresholds ─────────────────────────────────────────────
    COVERAGE_THRESHOLD: float = 0.75
    COVERAGE_IMPROVEMENT_MIN: float = 0.05

    # ── Observability ──────────────────────────────────────────────────
    LOG_LEVEL: str = "INFO"
    LANGSMITH_API_KEY: str = ""
    LANGSMITH_PROJECT: str = "apollo-research"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the cached singleton :class:`Settings` instance.

    The first call reads from the environment / ``.env``; subsequent calls
    return the same object without any I/O.
    """
    return Settings()  # type: ignore[call-arg]
