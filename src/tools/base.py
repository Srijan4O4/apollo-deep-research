"""Base utilities, shared helpers, and common types for Apollo tool layer."""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any, Callable, Coroutine, TypeVar

from pydantic import BaseModel, Field

from src.models.state import Evidence

logger = logging.getLogger("apollo.tools")

T = TypeVar("T")


# ---------------------------------------------------------------------------
# Custom exception
# ---------------------------------------------------------------------------


class ToolError(Exception):
    """Raised when a tool encounters a recoverable or permanent failure.

    Attributes:
        tool_name: Identifier of the tool that failed.
        original_error: The underlying exception (if any).
    """

    def __init__(
        self,
        message: str,
        *,
        tool_name: str,
        original_error: Exception | None = None,
    ) -> None:
        super().__init__(message)
        self.tool_name = tool_name
        self.original_error = original_error


# ---------------------------------------------------------------------------
# Intermediate search result model
# ---------------------------------------------------------------------------


class SearchResult(BaseModel):
    """Normalised intermediate result returned by raw API calls before
    conversion to the canonical :class:`Evidence` model.
    """

    title: str
    url: str
    content: str
    score: float = 0.0
    metadata: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Retry helper
# ---------------------------------------------------------------------------


async def retry_with_backoff(
    func: Callable[..., Coroutine[Any, Any, T]],
    *args: Any,
    max_retries: int = 3,
    base_delay: float = 1.0,
    **kwargs: Any,
) -> T:
    """Execute *func* with exponential back-off on failure.

    Parameters
    ----------
    func:
        An async callable to invoke.
    *args:
        Positional arguments forwarded to *func*.
    max_retries:
        Maximum number of retry attempts (excluding the initial call).
    base_delay:
        Delay in seconds for the first retry; doubles on each subsequent
        attempt.
    **kwargs:
        Keyword arguments forwarded to *func*.

    Returns
    -------
    T
        The return value of *func* on success.

    Raises
    ------
    ToolError
        If all attempts are exhausted.
    """

    last_error: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            return await func(*args, **kwargs)
        except Exception as exc:  # noqa: BLE001 – intentionally broad
            last_error = exc
            if attempt < max_retries:
                delay = base_delay * (2 ** attempt)
                logger.warning(
                    "Retry %d/%d for %s after %.1fs – %s",
                    attempt + 1,
                    max_retries,
                    getattr(func, "__name__", str(func)),
                    delay,
                    exc,
                )
                await asyncio.sleep(delay)
            else:
                logger.error(
                    "All %d retries exhausted for %s: %s",
                    max_retries,
                    getattr(func, "__name__", str(func)),
                    exc,
                )

    raise ToolError(
        f"All {max_retries} retries exhausted.",
        tool_name=getattr(func, "__name__", str(func)),
        original_error=last_error,
    )


# ---------------------------------------------------------------------------
# Normalisation helper
# ---------------------------------------------------------------------------


def normalize_evidence(
    results: list[SearchResult],
    source_type: str,
) -> list[Evidence]:
    """Convert a list of :class:`SearchResult` into canonical :class:`Evidence` objects.

    Parameters
    ----------
    results:
        Raw search results from any search back-end.
    source_type:
        One of ``"web"``, ``"academic"``, ``"financial"``.

    Returns
    -------
    list[Evidence]
        Deduplicated, sorted (by score descending) evidence items.
    """

    seen_urls: set[str] = set()
    evidence_list: list[Evidence] = []

    for result in results:
        if result.url in seen_urls:
            continue
        seen_urls.add(result.url)

        evidence_list.append(
            Evidence(
                content=result.content,
                source_url=result.url,
                source_type=source_type,  # type: ignore[arg-type]
                title=result.title,
                relevance_score=max(0.0, min(1.0, result.score)),
                retrieved_at=datetime.now(tz=timezone.utc),
                metadata=result.metadata,
            )
        )

    # Return highest-relevance items first
    evidence_list.sort(key=lambda e: e.relevance_score, reverse=True)
    return evidence_list
