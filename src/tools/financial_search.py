"""SEC EDGAR full-text search integration for Apollo Deep Research."""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Literal

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

logger = logging.getLogger("apollo.tools.financial_search")

_EDGAR_SEARCH_URL = "https://efts.sec.gov/LATEST/search-index"
_EDGAR_FILING_BASE = "https://www.sec.gov/Archives/edgar/data"
_USER_AGENT = "Apollo Research Agent (apollo@research.dev)"
_HTTP_TIMEOUT = 30.0


# ---------------------------------------------------------------------------
# Input schema
# ---------------------------------------------------------------------------


class FinancialSearchInput(BaseModel):
    """Validated parameters for an SEC EDGAR full-text search."""

    query: str = Field(..., description="Full-text search query.")
    company_name: str | None = Field(
        default=None,
        description="Optional company name to narrow results.",
    )
    ticker: str | None = Field(
        default=None,
        description="Optional stock ticker symbol.",
    )
    filing_type: Literal["10-K", "10-Q", "8-K", "all"] = Field(
        default="all",
        description="SEC filing type filter.",
    )
    date_range: tuple[str, str] | None = Field(
        default=None,
        description="Optional (start_date, end_date) in YYYY-MM-DD format.",
    )
    max_results: int = Field(
        default=5,
        ge=1,
        le=50,
        description="Maximum number of filings to return.",
    )


# ---------------------------------------------------------------------------
# Core async implementation
# ---------------------------------------------------------------------------


async def _raw_financial_search(params: FinancialSearchInput) -> list[SearchResult]:
    """Call the SEC EDGAR full-text search API and return :class:`SearchResult` objects."""

    # Build the search query — combine free-text query with company/ticker
    search_query = params.query
    if params.ticker:
        search_query = f"{params.ticker} {search_query}"
    if params.company_name:
        search_query = f'"{params.company_name}" {search_query}'

    query_params: dict[str, Any] = {
        "q": search_query,
    }

    if params.filing_type != "all":
        query_params["forms"] = params.filing_type

    if params.date_range is not None:
        start_date, end_date = params.date_range
        query_params["dateRange"] = "custom"
        query_params["startdt"] = start_date
        query_params["enddt"] = end_date

    headers = {
        "User-Agent": _USER_AGENT,
        "Accept": "application/json",
    }

    async with httpx.AsyncClient(timeout=_HTTP_TIMEOUT) as client:
        response = await client.get(
            _EDGAR_SEARCH_URL,
            params=query_params,
            headers=headers,
        )
        response.raise_for_status()

    data = response.json()
    hits: list[dict[str, Any]] = data.get("hits", {}).get("hits", [])

    results: list[SearchResult] = []
    for hit in hits[: params.max_results]:
        source = hit.get("_source", {})

        # Extract fields from the EDGAR response
        filing_type = source.get("form_type", source.get("file_type", ""))
        company = source.get("entity_name", source.get("display_names", [""])[0] if source.get("display_names") else "")
        filed_date = source.get("file_date", source.get("period_of_report", ""))
        file_num = source.get("file_num", "")

        # Build the filing URL
        file_path = source.get("file_path", "")
        if file_path:
            filing_url = f"https://www.sec.gov/Archives/{file_path}"
        else:
            # Fallback: construct from accession number
            accession = source.get("accession_no", hit.get("_id", ""))
            filing_url = f"https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&accession={accession}"

        # Use snippet or description as content
        content_snippet = ""
        if "highlight" in hit:
            highlights = hit["highlight"]
            for field_values in highlights.values():
                if isinstance(field_values, list):
                    content_snippet = " ... ".join(field_values)
                    break
        if not content_snippet:
            content_snippet = source.get("title", source.get("entity_name", params.query))

        title = f"{company} – {filing_type}" if company and filing_type else (company or filing_type or "SEC Filing")

        results.append(
            SearchResult(
                title=title,
                url=filing_url,
                content=content_snippet,
                score=float(hit.get("_score", 0.0)) / 100.0,  # normalise EDGAR score
                metadata={
                    "filing_type": filing_type,
                    "company_name": company,
                    "filed_date": filed_date,
                    "file_number": file_num,
                    "accession_number": source.get("accession_no", ""),
                },
            )
        )

    return results


async def financial_search(params: FinancialSearchInput) -> list[Evidence]:
    """Execute a financial filing search on SEC EDGAR.

    Parameters
    ----------
    params:
        Validated search parameters.

    Returns
    -------
    list[Evidence]
        Evidence objects with ``source_type="financial"``.

    Raises
    ------
    ToolError
        If the search fails after retries.
    """

    logger.info(
        "Financial search: query=%r  ticker=%s  filing_type=%s  max=%d",
        params.query,
        params.ticker,
        params.filing_type,
        params.max_results,
    )

    try:
        raw_results: list[SearchResult] = await retry_with_backoff(
            _raw_financial_search,
            params,
            max_retries=3,
            base_delay=2.0,
        )
    except Exception as exc:
        raise ToolError(
            f"SEC EDGAR search failed for query: {params.query!r}",
            tool_name="financial_search",
            original_error=exc if isinstance(exc, Exception) else None,
        ) from exc

    evidence = normalize_evidence(raw_results, source_type="financial")
    logger.info("Financial search returned %d evidence items.", len(evidence))
    return evidence


# ---------------------------------------------------------------------------
# LangChain-compatible tool wrapper
# ---------------------------------------------------------------------------


def _format_financial_evidence(evidence_list: list[Evidence]) -> str:
    """Render financial evidence items as a readable string for LLM consumption."""

    if not evidence_list:
        return "No SEC filings found."

    parts: list[str] = []
    for idx, ev in enumerate(evidence_list, start=1):
        filing_type = ev.metadata.get("filing_type", "N/A")
        company = ev.metadata.get("company_name", "Unknown")
        filed_date = ev.metadata.get("filed_date", "N/A")

        parts.append(
            f"[{idx}] {ev.title}\n"
            f"    Company: {company}\n"
            f"    Filing: {filing_type}  |  Date: {filed_date}\n"
            f"    URL: {ev.source_url}\n"
            f"    Relevance: {ev.relevance_score:.2f}\n"
            f"    Excerpt: {ev.content[:400]}"
        )
    return "\n\n".join(parts)


@tool
async def financial_search_tool(
    query: str,
    ticker: str = "",
    filing_type: str = "all",
) -> str:
    """Search SEC EDGAR for financial filings and return formatted results.

    Args:
        query: The search query for SEC filings.
        ticker: Optional stock ticker symbol to narrow results.
        filing_type: Filing type filter — one of '10-K', '10-Q', '8-K', or 'all'.

    Returns:
        Formatted string of SEC filing results for the LLM.
    """
    valid_types = {"10-K", "10-Q", "8-K", "all"}
    if filing_type not in valid_types:
        filing_type = "all"

    params = FinancialSearchInput(
        query=query,
        ticker=ticker or None,
        filing_type=filing_type,  # type: ignore[arg-type]
    )

    try:
        evidence = await financial_search(params)
        return _format_financial_evidence(evidence)
    except Exception as exc:
        logger.warning("Financial search call failed gracefully: %s", exc)
        return f"Financial search unavailable ({exc}). Continuing with other sources."
