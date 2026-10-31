"""Semantic Scholar academic paper search integration for Apollo Deep Research."""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

import httpx
from langchain_core.tools import tool
from pydantic import BaseModel, Field

from src.models.state import Evidence
from src.tools.base import (
    SearchResult,
    ToolError,
    normalize_evidence,
    retry_with_backoff,
)

logger = logging.getLogger("apollo.tools.academic_search")

_SEMANTIC_SCHOLAR_URL = "https://api.semanticscholar.org/graph/v1/paper/search"
_FIELDS = "title,abstract,url,year,authors,citationCount,openAccessPdf"
_HTTP_TIMEOUT = 30.0


# ---------------------------------------------------------------------------
# Input schema
# ---------------------------------------------------------------------------


class AcademicSearchInput(BaseModel):
    """Validated parameters for a Semantic Scholar paper search."""

    query: str = Field(..., max_length=300, description="Academic search query.")
    max_results: int = Field(
        default=10,
        ge=1,
        le=50,
        description="Maximum number of papers to return.",
    )
    year_range: tuple[int, int] | None = Field(
        default=None,
        description="Optional (start_year, end_year) filter.",
    )
    fields_of_study: list[str] = Field(
        default_factory=list,
        description="Restrict to specific fields of study (e.g. 'Computer Science').",
    )
    open_access_only: bool = Field(
        default=False,
        description="If True, return only papers with open-access PDFs.",
    )


# ---------------------------------------------------------------------------
# Core async implementation
# ---------------------------------------------------------------------------


async def _raw_academic_search(params: AcademicSearchInput) -> list[SearchResult]:
    """Call the Semantic Scholar API and return intermediate :class:`SearchResult` objects."""

    query_params: dict[str, Any] = {
        "query": params.query,
        "limit": params.max_results,
        "fields": _FIELDS,
    }

    if params.year_range is not None:
        start_year, end_year = params.year_range
        query_params["year"] = f"{start_year}-{end_year}"

    if params.fields_of_study:
        query_params["fieldsOfStudy"] = ",".join(params.fields_of_study)

    async with httpx.AsyncClient(timeout=_HTTP_TIMEOUT) as client:
        response = await client.get(_SEMANTIC_SCHOLAR_URL, params=query_params)
        response.raise_for_status()

    data = response.json()
    raw_papers: list[dict[str, Any]] = data.get("data", [])

    results: list[SearchResult] = []
    for paper in raw_papers:
        # Respect open-access filter
        open_access_pdf = paper.get("openAccessPdf")
        if params.open_access_only and not open_access_pdf:
            continue

        authors = paper.get("authors", [])
        author_names = [a.get("name", "") for a in authors] if authors else []
        citation_count = paper.get("citationCount", 0) or 0

        # Build a usable URL — prefer open-access PDF, fall back to S2 page
        paper_url: str = ""
        if open_access_pdf and isinstance(open_access_pdf, dict):
            paper_url = open_access_pdf.get("url", "")
        if not paper_url:
            paper_url = paper.get("url", f"https://www.semanticscholar.org/paper/{paper.get('paperId', '')}")

        # Compute a simple relevance proxy from citation count (log-scaled, capped)
        import math

        relevance = min(1.0, math.log1p(citation_count) / 10.0) if citation_count else 0.1

        abstract = paper.get("abstract") or ""

        results.append(
            SearchResult(
                title=paper.get("title", "Untitled"),
                url=paper_url,
                content=abstract,
                score=relevance,
                metadata={
                    "year": paper.get("year"),
                    "authors": author_names,
                    "citation_count": citation_count,
                    "paper_id": paper.get("paperId", ""),
                    "has_open_access_pdf": bool(open_access_pdf),
                },
            )
        )

    return results


async def academic_search(params: AcademicSearchInput) -> list[Evidence]:
    """Execute an academic paper search via Semantic Scholar.

    Parameters
    ----------
    params:
        Validated search parameters.

    Returns
    -------
    list[Evidence]
        Evidence objects with ``source_type="academic"``.

    Raises
    ------
    ToolError
        If the search fails after retries.
    """

    logger.info(
        "Academic search: query=%r  max=%d  year_range=%s  open_access=%s",
        params.query,
        params.max_results,
        params.year_range,
        params.open_access_only,
    )

    try:
        raw_results: list[SearchResult] = await retry_with_backoff(
            _raw_academic_search,
            params,
            max_retries=1,
            base_delay=0.5,
        )
        evidence = normalize_evidence(raw_results, source_type="academic")
    except Exception as exc:
        logger.info("Semantic Scholar error (%s). Falling back to direct arXiv search.", exc)
        from src.tools.arxiv_search import ArxivSearchInput, arxiv_search
        try:
            evidence = await arxiv_search(
                ArxivSearchInput(query=params.query, max_results=params.max_results)
            )
        except Exception as arxiv_err:
            logger.info("Direct arXiv search failed (%s). Falling back to Tavily arXiv search.", arxiv_err)
            from src.tools.web_search import WebSearchInput, web_search
            try:
                tavily_res = await web_search(
                    WebSearchInput(
                        query=f"site:arxiv.org OR site:semanticscholar.org {params.query}",
                        max_results=min(params.max_results, 5),
                    )
                )
                raw_results = [
                    SearchResult(
                        title=item.title,
                        url=item.source_url,
                        content=item.content,
                        score=item.relevance_score,
                        metadata={"source": "arxiv_tavily", **item.metadata},
                    )
                    for item in tavily_res
                ]
                evidence = normalize_evidence(raw_results, source_type="academic")
            except Exception:
                raise ToolError(
                    f"Academic search failed for query: {params.query!r}",
                    tool_name="academic_search",
                    original_error=exc if isinstance(exc, Exception) else None,
                ) from exc

    logger.info("Academic search returned %d evidence items.", len(evidence))
    return evidence


# ---------------------------------------------------------------------------
# LangChain-compatible tool wrapper
# ---------------------------------------------------------------------------


def _format_academic_evidence(evidence_list: list[Evidence]) -> str:
    """Render academic evidence items as structured JSON for LLM and Athena agent parsing."""
    if not evidence_list:
        return json.dumps([])

    data = []
    for ev in evidence_list:
        authors = ev.metadata.get("authors", [])
        author_str = ev.author if ev.author and ev.author != "Unknown" else (
            ", ".join(authors[:3]) + (" et al." if len(authors) > 3 else "") if authors else "Unknown"
        )
        data.append(
            {
                "title": ev.title,
                "author": author_str,
                "authors": authors,
                "year": ev.date or str(ev.metadata.get("year", "")),
                "arxiv_id": str(ev.metadata.get("arxiv_id", "")),
                "source_url": ev.source_url,
                "content": ev.content[:1500],
                "source_type": "academic",
                "relevance_score": ev.relevance_score,
            }
        )
    return json.dumps(data)


@tool
async def academic_search_tool(query: str, max_results: int = 10) -> str:
    """Search academic papers on Semantic Scholar and arXiv and return structured JSON results.

    Args:
        query: The academic search query (max 300 chars).
        max_results: Number of papers to return (1-50, default 10).

    Returns:
        Structured JSON string of academic paper results with author, title, year, arxiv_id, url.
    """
    params = AcademicSearchInput(query=query, max_results=max_results)
    try:
        evidence = await academic_search(params)
        return _format_academic_evidence(evidence)
    except Exception as exc:
        logger.warning("Academic search call failed gracefully: %s", exc)
        return json.dumps([])
