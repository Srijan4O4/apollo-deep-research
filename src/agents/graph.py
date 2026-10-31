"""LangGraph state machine — the Apollo research workflow graph.

This is the central orchestration module.  It wires together all agent nodes
(Apollo, Athena, Hermes) into a compiled LangGraph ``StateGraph`` that
implements the iterative research loop:

    START → plan → search → analyze → (search | synthesize)
                                          ↓
                                   (next_section | assemble) → END
"""

from __future__ import annotations

import logging
from typing import Any

from langgraph.graph import END, START, StateGraph

from src.agents.apollo import (
    apollo_assemble_report,
    apollo_plan,
    apollo_synthesize_section,
)
from src.agents.apollo_qa import apollo_qa_validate
from src.agents.athena import athena_search
from src.agents.hermes import hermes_analyze
from src.config import get_settings
from src.models.state import ResearchState

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Conditional-edge functions
# ---------------------------------------------------------------------------


def should_continue_searching(state: ResearchState) -> str:
    """Decide whether the current section needs more evidence or is ready for synthesis.

    Returns
    -------
    ``"search"``
        Another search–analyse cycle is needed.
    ``"synthesize"``
        Evidence is sufficient (or iteration limit reached); proceed to writing.
    """
    settings = get_settings()
    current_idx: int = state.get("current_section_idx", 0)
    iteration: int = state.get("iteration_count", 0)
    max_iter: int = state.get("max_iterations", settings.MAX_ITERATIONS_PER_SECTION)
    gap_reports = state.get("gap_reports", [])
    outline = state.get("outline", [])

    if not outline or current_idx >= len(outline):
        return "synthesize"

    # Find gap reports for the current section.
    current_section_id = outline[current_idx].section_id
    current_gaps = [
        g for g in gap_reports if g.section_id == current_section_id
    ]

    # First iteration — no gap report yet, must search first.
    if not current_gaps:
        logger.info(
            "Section '%s': no gap report yet — routing to search.",
            current_section_id,
        )
        return "search"

    latest_gap = current_gaps[-1]

    # Hard iteration limit.
    if iteration >= max_iter:
        logger.info(
            "Section '%s': iteration limit (%d) reached — routing to synthesize.",
            current_section_id,
            max_iter,
        )
        return "synthesize"

    # Coverage threshold met.
    if latest_gap.coverage_score >= settings.COVERAGE_THRESHOLD:
        logger.info(
            "Section '%s': coverage %.2f >= threshold %.2f — routing to synthesize.",
            current_section_id,
            latest_gap.coverage_score,
            settings.COVERAGE_THRESHOLD,
        )
        return "synthesize"

    # Stagnation check — if improvement between the last two analyses is tiny,
    # stop early to avoid wasting tokens.
    if len(current_gaps) >= 2:
        improvement = current_gaps[-1].coverage_score - current_gaps[-2].coverage_score
        if improvement < settings.COVERAGE_IMPROVEMENT_MIN:
            logger.info(
                "Section '%s': improvement %.3f < min %.3f — routing to synthesize.",
                current_section_id,
                improvement,
                settings.COVERAGE_IMPROVEMENT_MIN,
            )
            return "synthesize"

    # Check global token budget.
    token_usage: int = state.get("token_usage", 0)
    if token_usage >= settings.MAX_TOKEN_BUDGET:
        logger.warning(
            "Token budget exhausted (%d / %d) — routing to synthesize.",
            token_usage,
            settings.MAX_TOKEN_BUDGET,
        )
        return "synthesize"

    logger.info(
        "Section '%s': coverage %.2f — routing to search for more evidence.",
        current_section_id,
        latest_gap.coverage_score,
    )
    return "search"


def should_continue_sections(state: ResearchState) -> str:
    """Decide whether to advance to the next section or assemble the final report.

    Returns
    -------
    ``"next_section"``
        More sections remain to be processed.
    ``"assemble"``
        All sections have been synthesised; assemble the final report.
    """
    current_idx: int = state.get("current_section_idx", 0)
    outline = state.get("outline", [])
    total_sections = len(outline)

    if total_sections > 0 and current_idx < total_sections - 1:
        logger.info(
            "Section %d/%d complete — advancing to next section.",
            current_idx + 1,
            total_sections,
        )
        return "next_section"

    logger.info("All sections processed — routing to assemble.")
    return "assemble"


def check_plan_result(state: ResearchState) -> str:
    """Validate that planning produced a valid outline before proceeding to search."""
    if state.get("status") == "error" or not state.get("outline"):
        logger.error("Planning failed or produced no outline sections. Routing to assemble.")
        return "assemble"
    return "search"


# ---------------------------------------------------------------------------
# Helper node
# ---------------------------------------------------------------------------


def advance_section(state: ResearchState) -> dict[str, Any]:
    """Advance ``current_section_idx`` and reset per-section counters."""
    current_idx = state.get("current_section_idx", 0)
    outline = state.get("outline", [])
    new_idx = current_idx + 1
    section_title = outline[new_idx].title if new_idx < len(outline) else "End"
    logger.info(
        "Advancing to section %d — %s",
        new_idx + 1,
        section_title,
    )
    return {
        "current_section_idx": new_idx,
        "iteration_count": 0,
        "status": "searching",
    }


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------


def build_research_graph() -> Any:
    """Build and compile the Apollo research state machine.

    The compiled graph is a runnable that accepts a ``ResearchState`` dict and
    drives the full research workflow from planning through to final report
    assembly.

    Returns
    -------
    CompiledGraph
        The compiled LangGraph state machine, ready for ``.ainvoke()`` or
        ``.astream()``.
    """
    graph = StateGraph(ResearchState)

    # -- Nodes ---------------------------------------------------------------
    graph.add_node("plan", apollo_plan)
    graph.add_node("search", athena_search)
    graph.add_node("analyze", hermes_analyze)
    graph.add_node("synthesize", apollo_synthesize_section)
    graph.add_node("advance_section", advance_section)
    graph.add_node("assemble", apollo_assemble_report)
    graph.add_node("qa_validate", apollo_qa_validate)

    # -- Edges ---------------------------------------------------------------
    # Entry: decompose the query into an outline.
    graph.add_edge(START, "plan")

    # After planning, verify outline was generated; route to search or assemble.
    graph.add_conditional_edges(
        "plan",
        check_plan_result,
        {
            "search": "search",
            "assemble": "assemble",
        },
    )

    # After searching, analyse the gathered evidence.
    graph.add_edge("search", "analyze")

    # After analysis, decide: search again or synthesise?
    graph.add_conditional_edges(
        "analyze",
        should_continue_searching,
        {
            "search": "search",
            "synthesize": "synthesize",
        },
    )

    # After synthesis, decide: next section or assemble final report?
    graph.add_conditional_edges(
        "synthesize",
        should_continue_sections,
        {
            "next_section": "advance_section",
            "assemble": "assemble",
        },
    )

    # After advancing to the next section, begin searching it.
    graph.add_edge("advance_section", "search")

    # After assembly, run pre-publication quality assurance guardrail
    graph.add_edge("assemble", "qa_validate")

    # QA validation is the terminal node.
    graph.add_edge("qa_validate", END)

    # -- Compile & return ----------------------------------------------------
    compiled = graph.compile()
    logger.info("Apollo research graph compiled successfully.")
    return compiled
