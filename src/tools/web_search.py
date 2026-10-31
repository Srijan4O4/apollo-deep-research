"""Tavily web search integration for Apollo Deep Research."""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

import httpx
from langchain_core.tools import tool
from pydantic import BaseModel, Field

from src.config import Settings
from src.models.state import Evidence
from src.tools.base import (
    SearchResult,
    ToolError,
    normalize_evidence,
    retry_with_backoff,
)

logger = logging.getLogger("apollo.tools.web_search")

_TAVILY_SEARCH_URL = "https://api.tavily.com/search"
_HTTP_TIMEOUT = 30.0


# ---------------------------------------------------------------------------
# Input schema
# ---------------------------------------------------------------------------


class WebSearchInput(BaseModel):
    """Validated parameters for a Tavily web search."""

    query: str = Field(..., max_length=500, description="Search query string.")
    max_results: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Number of results to return.",
    )
    search_depth: str = Field(
        default="advanced",
        pattern="^(basic|advanced)$",
        description="Tavily search depth: 'basic' or 'advanced'.",
    )
    include_domains: list[str] = Field(
        default_factory=list,
        description="Restrict results to these domains.",
    )
    exclude_domains: list[str] = Field(
        default_factory=list,
        description="Exclude results from these domains.",
    )


# ---------------------------------------------------------------------------
# Core async implementation
# ---------------------------------------------------------------------------


async def _raw_tavily_search(params: WebSearchInput) -> list[SearchResult]:
    """Call the Tavily Search API and return intermediate :class:`SearchResult` objects."""

    settings = Settings()  # type: ignore[call-arg]
    api_key = settings.TAVILY_API_KEY

    payload: dict[str, Any] = {
        "api_key": api_key,
        "query": params.query,
        "max_results": params.max_results,
        "search_depth": params.search_depth,
    }
    if params.include_domains:
        payload["include_domains"] = params.include_domains
    if params.exclude_domains:
        payload["exclude_domains"] = params.exclude_domains

    async with httpx.AsyncClient(timeout=_HTTP_TIMEOUT) as client:
        response = await client.post(_TAVILY_SEARCH_URL, json=payload)
        response.raise_for_status()

    data = response.json()
    results: list[SearchResult] = []
    for item in data.get("results", []):
        results.append(
            SearchResult(
                title=item.get("title", ""),
                url=item.get("url", ""),
                content=item.get("content", ""),
                score=float(item.get("score", 0.0)),
                metadata={
                    "raw_content_length": len(item.get("raw_content", "") or ""),
                },
            )
        )

    return results


async def web_search(params: WebSearchInput) -> list[Evidence]:
    """Execute a web search via Tavily and return :class:`Evidence` items.

    Parameters
    ----------
    params:
        Validated search parameters.

    Returns
    -------
    list[Evidence]
        Evidence objects with ``source_type="web"``.

    Raises
    ------
    ToolError
        If the search fails after retries.
    """

    logger.info("Web search: query=%r  depth=%s  max=%d", params.query, params.search_depth, params.max_results)

    try:
        raw_results: list[SearchResult] = await retry_with_backoff(
            _raw_tavily_search,
            params,
            max_retries=3,
            base_delay=1.0,
        )
    except Exception as exc:
        raise ToolError(
            f"Tavily web search failed for query: {params.query!r}",
            tool_name="web_search",
            original_error=exc if isinstance(exc, Exception) else None,
        ) from exc

    evidence = normalize_evidence(raw_results, source_type="web")
    logger.info("Web search returned %d evidence items.", len(evidence))
    return evidence


# ---------------------------------------------------------------------------
# LangChain-compatible tool wrapper
# ---------------------------------------------------------------------------


def _format_evidence_for_llm(evidence_list: list[Evidence]) -> str:
    """Render evidence items as structured JSON for LLM and agent parsing."""
    if not evidence_list:
        return json.dumps([])

    data = [
        {
            "title": ev.title,
            "author": ev.author if ev.author and ev.author != "Unknown" else "Web Source",
            "source_url": ev.source_url,
            "content": ev.content[:1500],
            "source_type": "web",
            "relevance_score": ev.relevance_score,
            "date": ev.date,
        }
        for ev in evidence_list
    ]
    return json.dumps(data)


@tool
async def web_search_tool(query: str, max_results: int = 5) -> str:
    """Search the web using Tavily and return structured JSON results.

    Args:
        query: The search query string (max 500 chars).
        max_results: Number of results to return (1-20, default 5).

    Returns:
        Structured JSON string of search results with title, url, snippet, author.
    """
    params = WebSearchInput(query=query, max_results=max_results)
    try:
        evidence = await web_search(params)
        return _format_evidence_for_llm(evidence)
    except Exception as exc:
        logger.warning("Web search call failed gracefully: %s", exc)
        return json.dumps([])
