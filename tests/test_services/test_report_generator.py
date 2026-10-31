"""Tests for ReportGenerator service."""

from __future__ import annotations

import json
import pytest

from src.models.state import Citation, Evidence, ResearchState, SectionDraft, SectionOutline
from src.services.citation_manager import CitationManager
from src.services.report_generator import ReportGenerator


@pytest.mark.anyio
async def test_report_generator_generate_report() -> None:
    citation_manager = CitationManager()
    generator = ReportGenerator(citation_manager)

    evidence = [
        Evidence(
            content="Agent architectures allow self-correction.",
            source_url="https://arxiv.org/abs/2303.11366",
            source_type="academic",
            title="Reflexion: Language Agents with Verbal Reinforcement Learning",
            relevance_score=0.95,
            author="Noah Shinn et al.",
            date="2023",
        )
    ]
    sections = [
        SectionDraft(
            section_id="intro",
            title="Introduction to Agentic AI",
            content="Reflexion provides iterative memory [1].",
            citations=[],
        )
    ]
    outline = [
        SectionOutline(
            section_id="intro",
            title="Introduction to Agentic AI",
            description="Overview of reasoning loops",
            key_questions=["How does self-correction work?"],
        )
    ]

    state: ResearchState = {
        "query": "Recent advances in agentic AI",
        "outline": outline,
        "evidence": evidence,
        "gap_reports": [],
        "section_drafts": sections,
        "current_section_idx": 0,
        "iteration_count": 1,
        "max_iterations": 5,
        "status": "synthesizing",
        "messages": [],
        "error_log": [],
        "token_usage": 1500,
    }

    report = await generator.generate_report(state)

    assert report.query == "Recent advances in agentic AI"
    assert len(report.sections) == 1
    assert len(report.bibliography) >= 1
    assert report.total_sources >= 1

    # Test markdown formatting
    md = generator.format_as_markdown(report)
    assert "# " in md
    assert "Introduction to Agentic AI" in md
    assert "Bibliography" in md
    assert "Research Methodology & Evidence Standards" in md


@pytest.mark.anyio
async def test_report_generator_assembly_metadata_and_no_duplication() -> None:
    citation_manager = CitationManager()
    generator = ReportGenerator(citation_manager)

    evidence = [
        Evidence(
            content="Qwen2.5 models achieve strong scores on HumanEval.",
            source_url="https://arxiv.org/abs/2409.12190",
            source_type="academic",
            title="Qwen2.5 Technical Report",
            relevance_score=0.95,
            author="Qwen Team",
            date="2024",
        )
    ]
    sections = [
        SectionDraft(
            section_id="code_gen",
            title="Code Generation Benchmarks",
            content="### Code Generation Benchmarks\n\nQwen2.5-Coder scores 85% on HumanEval [1].",
            citations=[],
        ),
        # Assembly metadata draft
        SectionDraft(
            section_id="__assembly_metadata__",
            title="SLMs in Enterprise: An Empirical Evaluation",
            content=json.dumps({
                "title": "SLMs in Enterprise: An Empirical Evaluation",
                "executive_summary": "Compact models achieve competitive parity in domain-specific tasks.",
                "key_findings": ["Qwen2.5 achieves 85% on HumanEval [1]."],
                "methodology": "Systematic review across arXiv and benchmark datasets.",
            }),
            citations=[],
        ),
        # Legacy fallback that must be omitted
        SectionDraft(
            section_id="__final_report__",
            title="Research Report (unassembled)",
            content="Duplicate text...",
            citations=[],
        ),
    ]

    state: ResearchState = {
        "query": "Can SLMs replace LLMs?",
        "outline": [],
        "evidence": evidence,
        "gap_reports": [],
        "section_drafts": sections,
        "current_section_idx": 1,
        "iteration_count": 1,
        "max_iterations": 3,
        "status": "complete",
        "messages": [],
        "error_log": [],
        "token_usage": 2000,
    }

    report = await generator.generate_report(state)

    assert report.title == "SLMs in Enterprise: An Empirical Evaluation"
    assert "Compact models achieve competitive parity" in report.executive_summary
    assert len(report.sections) == 1
    assert report.sections[0].section_id == "code_gen"
    # Ensure redundant leading header was stripped
    assert not report.sections[0].content.startswith("### Code Generation Benchmarks")

    md = generator.format_as_markdown(report)
    assert "Research Report (unassembled)" not in md
    assert "Systematic review across arXiv" in md
    assert "Bibliography" in md
    assert "Qwen Team" in md or "arxiv.org/abs/2409.12190" in md
