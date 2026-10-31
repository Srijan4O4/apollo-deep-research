"""Tests for Apollo search tools and utilities."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.tools.base import SearchResult, ToolError, normalize_evidence
from src.tools.web_search import WebSearchInput
from src.tools.academic_search import AcademicSearchInput
from src.tools.financial_search import FinancialSearchInput


def test_search_result_and_normalize_evidence() -> None:
    results = [
        SearchResult(
            title="Result 1",
            url="https://example.com/item1",
            content="Summary content 1",
            score=0.95,
        ),
        SearchResult(
            title="Result 2",
            url="https://example.com/item2",
            content="Summary content 2",
            score=0.85,
        ),
        # Duplicate URL should be deduplicated by normalize_evidence
        SearchResult(
            title="Result 1 duplicate",
            url="https://example.com/item1",
            content="Summary content 1 duplicate",
            score=0.90,
        ),
    ]

    evidence = normalize_evidence(results, source_type="web")
    assert len(evidence) == 2
    assert evidence[0].source_url == "https://example.com/item1"
    assert evidence[0].relevance_score == 0.95
    assert evidence[0].source_type == "web"


def test_tool_error() -> None:
    original = ValueError("API timed out")
    err = ToolError("Search failed", tool_name="web_search", original_error=original)
    assert err.tool_name == "web_search"
    assert err.original_error is original
    assert "Search failed" in str(err)


def test_tool_input_validation() -> None:
    # Valid input
    web_in = WebSearchInput(query="Quantum AI breakthroughs", max_results=5)
    assert web_in.max_results == 5

    # Out of range max_results
    with pytest.raises(ValidationError):
        WebSearchInput(query="test", max_results=50)

    # Academic search
    acad_in = AcademicSearchInput(query="Attention is all you need", max_results=10)
    assert acad_in.query == "Attention is all you need"

    # Financial search
    fin_in = FinancialSearchInput(query="Revenue and margins", ticker="NVDA", filing_type="10-K")
    assert fin_in.ticker == "NVDA"
    assert fin_in.filing_type == "10-K"
