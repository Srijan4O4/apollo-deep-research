"""Tests for Pydantic models: Evidence, SectionOutline, GapReport, ResearchRequest, TaskStatus."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.models.requests import ResearchRequest
from src.models.responses import TaskStatus
from src.models.state import Evidence, GapReport, SectionOutline


class TestEvidence:
    """Tests for the Evidence model."""

    def test_evidence_creation_valid(self) -> None:
        """Test creating an Evidence instance with valid data."""
        evidence = Evidence(
            source_url="https://example.com/article",
            title="Test Article",
            content="This is the content of the article discussing important findings.",
            relevance_score=0.85,
            source_type="web",
        )

        assert evidence.source_url == "https://example.com/article"
        assert evidence.title == "Test Article"
        assert evidence.content == "This is the content of the article discussing important findings."
        assert evidence.relevance_score == 0.85
        assert evidence.source_type == "web"

    def test_evidence_creation_minimal(self) -> None:
        """Test creating an Evidence instance with minimal required fields."""
        evidence = Evidence(
            source_url="https://example.com",
            title="Minimal",
            content="Content",
            relevance_score=0.5,
            source_type="web",
        )

        assert evidence.source_url == "https://example.com"
        assert evidence.relevance_score == 0.5

    def test_evidence_relevance_score_zero(self) -> None:
        """Test that relevance_score can be exactly 0."""
        evidence = Evidence(
            source_url="https://example.com",
            title="Low Relevance",
            content="Not very relevant content",
            relevance_score=0.0,
            source_type="web",
        )

        assert evidence.relevance_score == 0.0

    def test_evidence_relevance_score_one(self) -> None:
        """Test that relevance_score can be exactly 1."""
        evidence = Evidence(
            source_url="https://example.com",
            title="High Relevance",
            content="Highly relevant content",
            relevance_score=1.0,
            source_type="web",
        )

        assert evidence.relevance_score == 1.0

    def test_evidence_relevance_score_above_one_rejected(self) -> None:
        """Test that relevance_score > 1 is rejected by validation."""
        with pytest.raises(ValidationError):
            Evidence(
                source_url="https://example.com",
                title="Invalid",
                content="Content",
                relevance_score=1.5,
                source_type="web",
            )

    def test_evidence_relevance_score_negative_rejected(self) -> None:
        """Test that negative relevance_score is rejected by validation."""
        with pytest.raises(ValidationError):
            Evidence(
                source_url="https://example.com",
                title="Invalid",
                content="Content",
                relevance_score=-0.1,
                source_type="web",
            )


class TestSectionOutline:
    """Tests for the SectionOutline model."""

    def test_section_outline_creation(self) -> None:
        """Test creating a SectionOutline with valid data."""
        outline = SectionOutline(
            title="Introduction",
            description="Overview of the research topic and methodology",
            key_questions=["What is the main question?", "Why is this important?"],
            search_queries=["research topic overview", "topic methodology"],
        )

        assert outline.title == "Introduction"
        assert outline.description == "Overview of the research topic and methodology"
        assert len(outline.key_questions) == 2
        assert len(outline.search_queries) == 2

    def test_section_outline_empty_lists(self) -> None:
        """Test creating a SectionOutline with empty question/query lists."""
        outline = SectionOutline(
            title="Conclusion",
            description="Summary of findings",
            key_questions=[],
            search_queries=[],
        )

        assert outline.title == "Conclusion"
        assert outline.key_questions == []
        assert outline.search_queries == []


class TestGapReport:
    """Tests for the GapReport model."""

    def test_gap_report_creation(self) -> None:
        """Test creating a GapReport with valid data."""
        gap_report = GapReport(
            section_title="Market Analysis",
            gaps=["Missing competitor data", "No pricing information"],
            suggested_queries=["competitor analysis 2024", "market pricing trends"],
            coverage_score=0.45,
        )

        assert gap_report.section_title == "Market Analysis"
        assert len(gap_report.gaps) == 2
        assert len(gap_report.suggested_queries) == 2
        assert gap_report.coverage_score == 0.45

    def test_gap_report_full_coverage(self) -> None:
        """Test creating a GapReport with full coverage (no gaps)."""
        gap_report = GapReport(
            section_title="Well-Researched Section",
            gaps=[],
            suggested_queries=[],
            coverage_score=1.0,
        )

        assert gap_report.gaps == []
        assert gap_report.coverage_score == 1.0


class TestResearchRequest:
    """Tests for the ResearchRequest model."""

    def test_research_request_defaults(self) -> None:
        """Test that ResearchRequest has correct default values."""
        request = ResearchRequest(query="What is quantum computing?")

        assert request.query == "What is quantum computing?"
        assert request.max_sections == 8
        assert request.max_iterations == 5
        assert request.include_financial is True
        assert request.include_academic is True

    def test_research_request_custom_values(self) -> None:
        """Test creating a ResearchRequest with custom values."""
        request = ResearchRequest(
            query="Analyze climate change impacts",
            max_sections=5,
            max_iterations=3,
            include_financial=False,
            include_academic=True,
        )

        assert request.query == "Analyze climate change impacts"
        assert request.max_sections == 5
        assert request.max_iterations == 3
        assert request.include_financial is False
        assert request.include_academic is True

    def test_research_request_requires_query(self) -> None:
        """Test that ResearchRequest requires a query field."""
        with pytest.raises(ValidationError):
            ResearchRequest()  # type: ignore[call-arg]


class TestTaskStatus:
    """Tests for the TaskStatus enum."""

    def test_task_status_values(self) -> None:
        """Test that all expected TaskStatus enum values exist."""
        assert TaskStatus.PENDING == "pending"
        assert TaskStatus.PLANNING == "planning"
        assert TaskStatus.SEARCHING == "searching"
        assert TaskStatus.ANALYZING == "analyzing"
        assert TaskStatus.SYNTHESIZING == "synthesizing"
        assert TaskStatus.ASSEMBLING == "assembling"
        assert TaskStatus.COMPLETE == "complete"
        assert TaskStatus.ERROR == "error"

    def test_task_status_is_string(self) -> None:
        """Test that TaskStatus values are strings."""
        for status in TaskStatus:
            assert isinstance(status.value, str)

    def test_task_status_count(self) -> None:
        """Test that there are exactly 8 TaskStatus values."""
        assert len(TaskStatus) == 8
