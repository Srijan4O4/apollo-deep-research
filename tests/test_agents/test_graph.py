"""Tests for LangGraph State Machine topology and conditional edges."""

from __future__ import annotations

from src.agents.graph import (
    advance_section,
    build_research_graph,
    should_continue_searching,
    should_continue_sections,
)
from src.models.state import GapReport, ResearchState, SectionOutline


def test_should_continue_searching_first_iteration() -> None:
    outline = [
        SectionOutline(section_id="intro", title="Intro", description="Intro section", key_questions=["Q1"])
    ]
    state: ResearchState = {
        "query": "Test query",
        "outline": outline,
        "evidence": [],
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

    assert should_continue_searching(state) == "search"


def test_should_continue_searching_threshold_met() -> None:
    outline = [
        SectionOutline(section_id="intro", title="Intro", description="Intro section", key_questions=["Q1"])
    ]
    gap = GapReport(section_id="intro", coverage_score=0.9, reasoning="Well covered")
    state: ResearchState = {
        "query": "Test query",
        "outline": outline,
        "evidence": [],
        "gap_reports": [gap],
        "section_drafts": [],
        "current_section_idx": 0,
        "iteration_count": 1,
        "max_iterations": 5,
        "status": "analyzing",
        "messages": [],
        "error_log": [],
        "token_usage": 0,
    }

    assert should_continue_searching(state) == "synthesize"


def test_should_continue_searching_max_iterations() -> None:
    outline = [
        SectionOutline(section_id="intro", title="Intro", description="Intro section", key_questions=["Q1"])
    ]
    gap = GapReport(section_id="intro", coverage_score=0.4, reasoning="Gaps remain")
    state: ResearchState = {
        "query": "Test query",
        "outline": outline,
        "evidence": [],
        "gap_reports": [gap],
        "section_drafts": [],
        "current_section_idx": 0,
        "iteration_count": 5,
        "max_iterations": 5,
        "status": "analyzing",
        "messages": [],
        "error_log": [],
        "token_usage": 0,
    }

    assert should_continue_searching(state) == "synthesize"


def test_should_continue_sections_navigation() -> None:
    outline = [
        SectionOutline(section_id="s1", title="S1", description="D1", key_questions=[]),
        SectionOutline(section_id="s2", title="S2", description="D2", key_questions=[]),
    ]

    state_sec1: ResearchState = {
        "query": "Test",
        "outline": outline,
        "evidence": [],
        "gap_reports": [],
        "section_drafts": [],
        "current_section_idx": 0,
        "iteration_count": 1,
        "max_iterations": 5,
        "status": "synthesizing",
        "messages": [],
        "error_log": [],
        "token_usage": 0,
    }
    assert should_continue_sections(state_sec1) == "next_section"

    state_sec2: ResearchState = {
        "query": "Test",
        "outline": outline,
        "evidence": [],
        "gap_reports": [],
        "section_drafts": [],
        "current_section_idx": 1,
        "iteration_count": 1,
        "max_iterations": 5,
        "status": "synthesizing",
        "messages": [],
        "error_log": [],
        "token_usage": 0,
    }
    assert should_continue_sections(state_sec2) == "assemble"


def test_advance_section() -> None:
    outline = [
        SectionOutline(section_id="s1", title="S1", description="D1", key_questions=[]),
        SectionOutline(section_id="s2", title="S2", description="D2", key_questions=[]),
    ]
    state: ResearchState = {
        "query": "Test",
        "outline": outline,
        "evidence": [],
        "gap_reports": [],
        "section_drafts": [],
        "current_section_idx": 0,
        "iteration_count": 3,
        "max_iterations": 5,
        "status": "synthesizing",
        "messages": [],
        "error_log": [],
        "token_usage": 0,
    }
    result = advance_section(state)
    assert result["current_section_idx"] == 1
    assert result["iteration_count"] == 0
    assert result["status"] == "searching"


def test_build_research_graph() -> None:
    graph = build_research_graph()
    assert graph is not None
