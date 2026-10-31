"""Inbound API request models.

These Pydantic models validate and document the JSON payloads accepted by
the Apollo REST/WebSocket endpoints.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class ResearchRequest(BaseModel):
    """Payload for initiating a new deep-research task."""

    query: str = Field(
        ...,
        min_length=3,
        max_length=2000,
        description="The research question or topic to investigate.",
    )
    max_sections: int = Field(
        default=8,
        ge=1,
        le=20,
        description="Maximum number of report sections to generate.",
    )
    max_iterations: int = Field(
        default=5,
        ge=1,
        le=15,
        description="Maximum search/analyze iterations per section.",
    )
    include_financial: bool = Field(
        default=True,
        description="Whether to include financial data sources.",
    )
    include_academic: bool = Field(
        default=True,
        description="Whether to include academic / scholarly sources.",
    )
