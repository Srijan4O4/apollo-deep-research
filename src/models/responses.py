"""Outbound API response models.

Pydantic models that define the JSON shape of every response returned by
the Apollo API — task confirmations, progress updates, and completed
research reports.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum

from pydantic import BaseModel, Field

from src.models.state import Citation, SectionDraft


# ── Enums ──────────────────────────────────────────────────────────────────


class TaskStatus(str, Enum):
    """Lifecycle states for a research task."""

    PENDING = "pending"
    PLANNING = "planning"
    SEARCHING = "searching"
    ANALYZING = "analyzing"
    SYNTHESIZING = "synthesizing"
    ASSEMBLING = "assembling"
    COMPLETE = "complete"
    ERROR = "error"


# ── Response Bodies ────────────────────────────────────────────────────────


class ResearchTaskResponse(BaseModel):
    """Acknowledgement returned when a research task is created."""

    task_id: str
    status: TaskStatus
    message: str


class ResearchProgress(BaseModel):
    """Snapshot of a running task's progress (used for polling / SSE)."""

    task_id: str
    status: TaskStatus
    current_section: str | None = None
    sections_completed: int = 0
    total_sections: int = 0
    iteration: int = 0
    evidence_count: int = 0


class ResearchReport(BaseModel):
    """The completed research report delivered to the client."""

    task_id: str
    query: str
    title: str
    executive_summary: str
    sections: list[SectionDraft]
    bibliography: list[Citation]
    generated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    total_sources: int = 0
    metadata: dict = Field(default_factory=dict)
