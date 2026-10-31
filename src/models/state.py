"""Core domain models and the LangGraph workflow state.

Every data structure that flows through the research pipeline is defined
here so that the rest of the codebase has a single source of truth for
serialisation, validation and type-checking.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Annotated, Literal, TypedDict

from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field


# ── Value Objects ──────────────────────────────────────────────────────────


class Citation(BaseModel):
    """A citation reference linking a claim to its source."""

    citation_id: int = 0
    number: int = 0
    source_url: str = ""
    title: str = ""
    source_type: Literal["web", "academic", "financial"] = "web"
    accessed_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    author: str = "Unknown"
    date: str = ""
    snippet: str = ""

    def __init__(self, **data):
        if "number" in data and "citation_id" not in data:
            data["citation_id"] = data["number"]
        elif "citation_id" in data and "number" not in data:
            data["number"] = data["citation_id"]
        super().__init__(**data)


class Evidence(BaseModel):
    """A piece of evidence gathered from search."""

    content: str
    source_url: str
    source_type: Literal["web", "academic", "financial"]
    title: str
    relevance_score: float = Field(ge=0.0, le=1.0)
    retrieved_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict = Field(default_factory=dict)
    author: str = "Unknown"
    date: str = ""


class SectionOutline(BaseModel):
    """A section in the research report outline."""

    section_id: str = ""
    title: str = ""
    description: str = ""
    key_questions: list[str] = Field(default_factory=list)
    search_queries: list[str] = Field(default_factory=list)

    def __init__(self, **data):
        if not data.get("section_id") and data.get("title"):
            data["section_id"] = data["title"].lower().replace(" ", "_")
        super().__init__(**data)


class GapReport(BaseModel):
    """Analysis of knowledge gaps for a section."""

    section_id: str = ""
    section_title: str = ""
    coverage_score: float = Field(default=0.0, ge=0.0, le=1.0)
    missing_topics: list[str] = Field(default_factory=list)
    gaps: list[str] = Field(default_factory=list)
    suggested_queries: list[str] = Field(default_factory=list)
    reasoning: str = ""

    def __init__(self, **data):
        if "section_title" in data and not data.get("section_id"):
            data["section_id"] = data["section_title"].lower().replace(" ", "_")
        if "gaps" in data and not data.get("missing_topics"):
            data["missing_topics"] = list(data["gaps"])
        elif "missing_topics" in data and not data.get("gaps"):
            data["gaps"] = list(data["missing_topics"])
        super().__init__(**data)


class SectionDraft(BaseModel):
    """A drafted section with inline citations."""

    section_id: str = ""
    title: str = ""
    section_title: str = ""
    content: str = ""  # markdown content
    citations: list[Citation] = Field(default_factory=list)
    quality_score: float = 1.0

    def __init__(self, **data):
        if "section_title" in data and not data.get("title"):
            data["title"] = data["section_title"]
        elif "title" in data and not data.get("section_title"):
            data["section_title"] = data["title"]
        super().__init__(**data)


# ── LangGraph State ───────────────────────────────────────────────────────


class ResearchState(TypedDict):
    """The central state for the LangGraph research workflow.

    Each key is a *channel* in the LangGraph state graph.  The ``messages``
    channel uses the built-in ``add_messages`` reducer so that appended
    messages are merged rather than replaced.
    """

    query: str
    outline: list[SectionOutline]
    evidence: list[Evidence]
    gap_reports: list[GapReport]
    section_drafts: list[SectionDraft]
    current_section_idx: int
    iteration_count: int
    max_iterations: int
    status: str  # "planning" | "searching" | "analyzing" | "synthesizing" | "assembling" | "complete" | "error"
    messages: Annotated[list, add_messages]
    error_log: list[str]
    token_usage: int
