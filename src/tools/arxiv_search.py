"""Direct arXiv API search integration for Apollo Deep Research."""

from __future__ import annotations

import asyncio
import json
import logging
import urllib.parse
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from typing import Any

import httpx
from langchain_core.tools import tool
from pydantic import BaseModel, Field

from src.models.state import Evidence
from src.tools.base import SearchResult, ToolError, normalize_evidence, retry_with_backoff

logger = logging.getLogger("apollo.tools.arxiv_search")

_ARXIV_API_URL = "https://export.arxiv.org/api/query"
_ATOM_NS = {"atom": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}
_HTTP_TIMEOUT = 25.0


class ArxivSearchInput(BaseModel):
    """Validated parameters for an arXiv academic paper search."""

    query: str = Field(..., max_length=300, description="Academic research query.")
    max_results: int = Field(
        default=10,
        ge=1,
        le=30,
        description="Maximum number of papers to return.",
    )


async def _raw_arxiv_search(params: ArxivSearchInput) -> list[SearchResult]:
    """Query arXiv Atom API and return intermediate :class:`SearchResult` objects."""
    clean_query = params.query.replace('"', "").strip()
    encoded_query = urllib.parse.quote(clean_query)
    url = f"{_ARXIV_API_URL}?search_query=all:{encoded_query}&start=0&max_results={params.max_results}&sortBy=relevance&sortOrder=descending"

    async with httpx.AsyncClient(timeout=_HTTP_TIMEOUT, follow_redirects=True) as client:
        response = await client.get(url)
        response.raise_for_status()

    root = ET.fromstring(response.text)
    results: list[SearchResult] = []

    for entry in root.findall("atom:entry", _ATOM_NS):
        id_elem = entry.find("atom:id", _ATOM_NS)
        title_elem = entry.find("atom:title", _ATOM_NS)
        summary_elem = entry.find("atom:summary", _ATOM_NS)
        published_elem = entry.find("atom:published", _ATOM_NS)

        if title_elem is None or summary_elem is None:
            continue

        title = " ".join(title_elem.text.strip().split()) if title_elem.text else "Untitled"
        summary = " ".join(summary_elem.text.strip().split()) if summary_elem.text else ""
        raw_url = id_elem.text.strip() if id_elem is not None and id_elem.text else ""

        # Extract arXiv ID (e.g. 2404.14219)
        arxiv_id = raw_url.split("/abs/")[-1] if "/abs/" in raw_url else raw_url

        # Authors
        authors = []
        for author in entry.findall("atom:author", _ATOM_NS):
            name_elem = author.find("atom:name", _ATOM_NS)
            if name_elem is not None and name_elem.text:
                authors.append(name_elem.text.strip())

        # Year
        pub_date = published_elem.text.strip() if published_elem is not None and published_elem.text else ""
        year = pub_date[:4] if len(pub_date) >= 4 else str(datetime.now(timezone.utc).year)

        # PDF Link
        pdf_url = ""
        for link in entry.findall("atom:link", _ATOM_NS):
            if link.attrib.get("title") == "pdf":
                pdf_url = link.attrib.get("href", "")
                break

        canonical_url = f"https://arxiv.org/abs/{arxiv_id}" if arxiv_id else raw_url

        results.append(
            SearchResult(
                title=title,
                url=canonical_url,
                content=summary,
                score=0.9,
                metadata={
                    "authors": authors,
                    "year": year,
                    "published": pub_date,
                    "arxiv_id": arxiv_id,
                    "pdf_url": pdf_url,
                    "source": "arxiv",
                },
            )
        )

    return results


async def arxiv_search(params: ArxivSearchInput) -> list[Evidence]:
    """Execute paper search directly against arXiv API."""
    logger.info("Executing arXiv search: query=%r max=%d", params.query, params.max_results)
    try:
        raw_results = await retry_with_backoff(
            _raw_arxiv_search,
            params,
            max_retries=2,
            base_delay=1.0,
        )
    except Exception as exc:
        raise ToolError(
            f"arXiv search failed for query: {params.query!r}",
            tool_name="arxiv_search",
            original_error=exc if isinstance(exc, Exception) else None,
        ) from exc

    evidence = normalize_evidence(raw_results, source_type="academic")
    for item in evidence:
        authors = item.metadata.get("authors", [])
        if authors:
            item.author = ", ".join(authors[:3]) + (" et al." if len(authors) > 3 else "")
        item.date = item.metadata.get("year", "")
    return evidence


@tool
async def arxiv_search_tool(query: str, max_results: int = 10) -> str:
    """Search academic papers directly on arXiv and return structured JSON results.

    Args:
        query: Academic search query (e.g. 'compact language models SLM distillation').
        max_results: Number of papers to retrieve (default 10).

    Returns:
        Structured JSON string containing paper details (title, authors, year, arxiv_id, abstract, url).
    """
    params = ArxivSearchInput(query=query, max_results=max_results)
    try:
        evidence = await arxiv_search(params)
        data = [
            {
                "title": ev.title,
                "authors": ev.metadata.get("authors", [ev.author]),
                "author": ev.author,
                "year": ev.date or ev.metadata.get("year", ""),
                "arxiv_id": ev.metadata.get("arxiv_id", ""),
                "source_url": ev.source_url,
                "content": ev.content[:1500],
                "source_type": "academic",
                "relevance_score": ev.relevance_score,
            }
            for ev in evidence
        ]
        return json.dumps(data)
    except Exception as exc:
        logger.warning("arXiv search tool error: %s", exc)
        return json.dumps([])
