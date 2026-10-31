"""Tests for Apollo agent nodes with mocked LLM calls."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, patch

import pytest
from langchain_core.messages import AIMessage

from src.agents.apollo import (
    _extract_json,
    apollo_assemble_report,
    apollo_plan,
    apollo_synthesize_section,
)
from src.models.state import Evidence, ResearchState, SectionDraft, SectionOutline


def test_extract_json() -> None:
    # Bare JSON
    data = {"key": "value"}
    assert _extract_json(json.dumps(data)) == data

    # Markdown fenced JSON
    fenced = "```json\n" + json.dumps(data) + "\n```"
    assert _extract_json(fenced) == data


@pytest.mark.anyio
async def test_apollo_plan_mocked() -> None:
    mock_outline = [
        {
            "section_id": "sec1",
            "title": "Introduction",
            "description": "Background info",
            "key_questions": ["What is this?"],
        }
    ]
    mock_response = json.dumps(mock_outline)

    with patch("src.agents.apollo._invoke_with_fallback", new_callable=AsyncMock) as mock_invoke:
        mock_invoke.return_value = mock_response

        state: ResearchState = {
            "query": "Deep Research Agents",
            "outline": [],
            "evidence": [],
            "gap_reports": [],
            "section_drafts": [],
            "current_section_idx": 0,
            "iteration_count": 0,
            "max_iterations": 5,
            "status": "planning",
            "messages": [],
            "error_log": [],
            "token_usage": 0,
        }

        result = await apollo_plan(state)

        assert result["status"] == "searching"
        assert len(result["outline"]) == 1
        assert result["outline"][0].section_id == "sec1"


@pytest.mark.anyio
async def test_apollo_synthesize_section_mocked() -> None:
    mock_draft = {
        "title": "Market Trends",
        "content": "Strong market momentum [1].",
        "citations": [
            {
                "citation_id": 1,
                "source_url": "https://example.com/trend",
                "title": "Trend 2025",
                "source_type": "web",
                "accessed_at": "2025-01-01T00:00:00Z",
            }
        ],
    }
    mock_response = json.dumps(mock_draft)

    with patch("src.agents.apollo._invoke_with_fallback", new_callable=AsyncMock) as mock_invoke:
        mock_invoke.return_value = mock_response

        outline = [
            SectionOutline(
                section_id="sec1",
                title="Market Trends",
                description="Analysis of trends",
                key_questions=["What are the trends?"],
            )
        ]
        evidence = [
            Evidence(
                content="Trend data",
                source_url="https://example.com/trend",
                source_type="web",
                title="Trend 2025",
                relevance_score=0.9,
            )
        ]

        state: ResearchState = {
            "query": "Market analysis",
            "outline": outline,
            "evidence": evidence,
            "gap_reports": [],
            "section_drafts": [],
            "current_section_idx": 0,
            "iteration_count": 1,
            "max_iterations": 5,
            "status": "searching",
            "messages": [],
            "error_log": [],
            "token_usage": 0,
        }

        result = await apollo_synthesize_section(state)

        assert "section_drafts" in result
        assert len(result["section_drafts"]) == 1
        assert result["section_drafts"][0].title == "Market Trends"
