"""Tests for CitationManager."""

from __future__ import annotations

from datetime import datetime, timezone

from src.models.state import Citation, Evidence
from src.services.citation_manager import CitationManager


def test_register_evidence_and_deduplicate() -> None:
    manager = CitationManager()
    evidence_list = [
        Evidence(
            content="Evidence 1",
            source_url="https://example.com/1",
            source_type="web",
            title="Source 1",
            relevance_score=0.9,
        ),
        Evidence(
            content="Evidence 2",
            source_url="https://example.com/2",
            source_type="academic",
            title="Source 2",
            relevance_score=0.8,
        ),
        Evidence(
            content="Evidence 1 duplicate",
            source_url="https://example.com/1",
            source_type="web",
            title="Source 1",
            relevance_score=0.7,
        ),
    ]

    manager.register_evidence(evidence_list)
    assert len(manager._registry) == 2


def test_resolve_citations() -> None:
    manager = CitationManager()
    evidence_list = [
        Evidence(
            content="Content A",
            source_url="https://example.com/a",
            source_type="web",
            title="Source A",
            relevance_score=0.9,
        ),
        Evidence(
            content="Content B",
            source_url="https://example.com/b",
            source_type="web",
            title="Source B",
            relevance_score=0.8,
        ),
    ]

    text = "Recent breakthroughs indicate progress [1], though challenges remain [2]."
    resolved_text, citations = manager.resolve_citations(text, evidence_list)

    assert "[1]" in resolved_text
    assert "[2]" in resolved_text
    assert len(citations) == 2


def test_build_bibliography() -> None:
    manager = CitationManager()
    citations = [
        Citation(
            citation_id=1,
            number=1,
            source_url="https://example.com/report",
            title="Deep Research Report",
            author="Jane Doe",
            date="2025",
        ),
    ]

    bib = manager.build_bibliography(citations)
    assert "## Bibliography" in bib
    assert "Deep Research Report" in bib
    assert "https://example.com/report" in bib
