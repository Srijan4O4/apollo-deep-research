"""Apollo tools layer — re-export all tool functions and input schemas."""

from __future__ import annotations

from src.tools.base import (
    SearchResult,
    ToolError,
    normalize_evidence,
    retry_with_backoff,
)
from src.tools.web_search import (
    WebSearchInput,
    web_search,
    web_search_tool,
)
from src.tools.academic_search import (
    AcademicSearchInput,
    academic_search,
    academic_search_tool,
)
from src.tools.arxiv_search import (
    ArxivSearchInput,
    arxiv_search,
    arxiv_search_tool,
)
from src.tools.financial_search import (
    FinancialSearchInput,
    financial_search,
    financial_search_tool,
)

__all__ = [
    # Base utilities
    "SearchResult",
    "ToolError",
    "normalize_evidence",
    "retry_with_backoff",
    # Web search
    "WebSearchInput",
    "web_search",
    "web_search_tool",
    # Academic search
    "AcademicSearchInput",
    "academic_search",
    "academic_search_tool",
    "ArxivSearchInput",
    "arxiv_search",
    "arxiv_search_tool",
    # Financial search
    "FinancialSearchInput",
    "financial_search",
    "financial_search_tool",
]
