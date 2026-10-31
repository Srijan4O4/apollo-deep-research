"""Tests for Hermes analyzer agent node with mocked LLM calls."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, patch

import pytest

from src.agents.hermes import hermes_analyze
from src.models.state import Evidence, ResearchState, SectionOutline


@pytest.mark.anyio
async def test_hermes_analyze_mocked() -> None:
    mock_gap_report = {
        "section_id": "sec1",
        "coverage_score": 0.85,
        "missing_topics": ["long-term scalability"],
        "suggested_queries": ["scalability benchmarks in production"],
        "reasoning": "Good coverage of basics, minor gap in long-term scalability data.",
    }
    mock_response = json.dumps(mock_gap_report)

    with patch("src.agents.hermes._invoke_with_fallback", new_callable=AsyncMock) as mock_invoke:
        mock_invoke.return_value = mock_response

        outline = [
            SectionOutline(
                section_id="sec1",
                title="System Architecture",
                description="Architecture analysis",
                key_questions=["How does it scale?"],
            )
        ]
        evidence = [
            Evidence(
                content="Architecture details",
                source_url="https://example.com/arch",
                source_type="web",
                title="Arch 2025",
                relevance_score=0.9,
            )
        ]

        state: ResearchState = {
            "query": "System Architecture Review",
            "outline": outline,
            "evidence": evidence,
            "gap_reports": [],
            "section_drafts": [],
            "current_section_idx": 0,
            "iteration_count": 0,
            "max_iterations": 5,
            "status": "searching",
            "messages": [],
            "error_log": [],
            "token_usage": 0,
        }

        result = await hermes_analyze(state)

        assert "gap_reports" in result
        assert len(result["gap_reports"]) == 1
        assert result["gap_reports"][0].coverage_score == 0.85
        assert result["iteration_count"] == 1
