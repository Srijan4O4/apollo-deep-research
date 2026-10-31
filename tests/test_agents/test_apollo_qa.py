"""Unit tests for Apollo QA Node."""

from __future__ import annotations

import pytest

from src.agents.apollo_qa import (
    _clean_redundant_headers,
    _normalize_text_anomalies,
    apollo_qa_validate,
)
from src.models.state import Citation, ResearchState, SectionDraft


def test_normalize_text_anomalies() -> None:
    text = "An LLM exceeding 70 parameters in FP16 precision requires multi-GPU nodes."
    fixed = _normalize_text_anomalies(text)
    assert "70B parameters" in fixed

    text2 = "Models under 14 parameters and 8 parameters offer great efficiency."
    fixed2 = _normalize_text_anomalies(text2)
    assert "14B parameters" in fixed2
    assert "8B parameters" in fixed2


def test_clean_redundant_headers() -> None:
    title = "Comparative Performance Analysis"
    content = "### Comparative Performance Analysis\n\nEvaluating the functional boundaries between SLMs and LLMs..."
    cleaned = _clean_redundant_headers(title, content)
    assert not cleaned.startswith("### Comparative Performance Analysis")
    assert cleaned.startswith("Evaluating the functional boundaries")


@pytest.mark.anyio
async def test_apollo_qa_validate() -> None:
    drafts = [
        SectionDraft(
            section_id="sec_1",
            title="Benchmark Comparison",
            content="### Benchmark Comparison\n\n| Model | GSM8K |\n| --- | --- |\n| Phi-4 | 84.8% |\n\nEvaluated using [1].",
            citations=[
                Citation(number=1, source_url="https://arxiv.org/abs/2412.08905", title="Phi-4 Technical Report")
            ],
        ),
        # Fallback dummy draft that should be stripped
        SectionDraft(
            section_id="__final_report__",
            title="Research Report (unassembled)",
            content="Duplicate report text...",
        ),
    ]

    state: ResearchState = {
        "query": "SLMs vs LLMs",
        "outline": [],
        "evidence": [],
        "gap_reports": [],
        "section_drafts": drafts,
        "current_section_idx": 1,
        "iteration_count": 1,
        "max_iterations": 3,
        "status": "assembling",
        "messages": [],
        "error_log": [],
        "token_usage": 500,
    }

    result = await apollo_qa_validate(state)

    assert result["status"] == "complete"
    assert len(result["section_drafts"]) == 1
    assert result["section_drafts"][0].section_id == "sec_1"
    assert not result["section_drafts"][0].content.startswith("### Benchmark Comparison")
    assert "| Model | GSM8K |" in result["section_drafts"][0].content
