"""Hermes analyzer sub-agent — evaluates evidence quality and detects gaps."""

from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI

from src.config import get_settings
from src.models.state import Evidence, GapReport, ResearchState, SectionOutline
from src.prompts.hermes_analyzer import HERMES_SYSTEM_PROMPT

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _build_primary_llm() -> ChatGoogleGenerativeAI:
    """Instantiate the primary Gemini LLM for analysis."""
    settings = get_settings()
    return ChatGoogleGenerativeAI(
        model=settings.PRIMARY_MODEL,
        google_api_key=settings.GEMINI_API_KEY,
        temperature=0.2,
    )


def _build_fallback_llm() -> Any:
    """Instantiate fallback LLM (OpenAI if key is configured, else secondary Gemini model)."""
    settings = get_settings()
    if settings.OPENAI_API_KEY:
        return ChatOpenAI(
            model=settings.FALLBACK_MODEL if "gpt" in settings.FALLBACK_MODEL else "gpt-4o",
            api_key=settings.OPENAI_API_KEY,
            temperature=0.2,
        )
    fallback_model = settings.FALLBACK_MODEL if "gemini" in settings.FALLBACK_MODEL else "gemini-3.1-flash-lite-preview"
    return ChatGoogleGenerativeAI(
        model=fallback_model,
        google_api_key=settings.GEMINI_API_KEY,
        temperature=0.2,
    )


def _extract_text_content(content: Any) -> str:
    """Normalize response content to a plain string."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        texts = []
        for part in content:
            if isinstance(part, str):
                texts.append(part)
            elif isinstance(part, dict) and "text" in part:
                texts.append(part["text"])
            elif hasattr(part, "text"):
                texts.append(part.text)
            else:
                texts.append(str(part))
        return "".join(texts)
    return str(content)


async def _invoke_with_fallback(
    messages: list[Any],
    *,
    timeout: float,
) -> str:
    """Try the primary LLM; on any failure, retry with the fallback.

    Returns the raw string content from the model response.
    """
    try:
        llm = _build_primary_llm()
        response = await asyncio.wait_for(
            llm.ainvoke(messages),
            timeout=timeout,
        )
        return _extract_text_content(response.content)
    except Exception as primary_err:
        logger.warning(
            "Primary LLM failed for Hermes (%s). Switching to fallback.",
            primary_err,
        )
        try:
            llm = _build_fallback_llm()
            response = await asyncio.wait_for(
                llm.ainvoke(messages),
                timeout=timeout,
            )
            return _extract_text_content(response.content)
        except Exception as fallback_err:
            logger.error("Fallback LLM also failed for Hermes: %s", fallback_err)
            raise


def _extract_json(text: Any) -> Any:
    """Extract and parse JSON from LLM output, tolerating markdown fences and wrappers."""
    cleaned = _extract_text_content(text).strip()
    match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", cleaned)
    if match:
        cleaned = match.group(1).strip()
    else:
        # Find first [ or { and last ] or }
        first_bracket = min([i for i in [cleaned.find("["), cleaned.find("{")] if i != -1], default=-1)
        last_bracket = max([cleaned.rfind("]"), cleaned.rfind("}")], default=-1)
        if first_bracket != -1 and last_bracket != -1 and last_bracket > first_bracket:
            cleaned = cleaned[first_bracket : last_bracket + 1]
    return json.loads(cleaned)


def _estimate_tokens(text: str) -> int:
    """Rough token estimate (≈ 4 characters per token)."""
    return max(1, len(text) // 4)


def _summarize_evidence(evidence_items: list[Evidence], max_items: int = 30) -> str:
    """Create a concise textual summary of evidence for the LLM prompt."""
    if not evidence_items:
        return "(No evidence collected yet.)"

    parts: list[str] = []
    for idx, ev in enumerate(evidence_items[:max_items], start=1):
        parts.append(
            f"[{idx}] Title: {ev.title}\n"
            f"    Source: {ev.source_url}\n"
            f"    Type: {ev.source_type}\n"
            f"    Relevance: {ev.relevance_score:.2f}\n"
            f"    Content preview: {ev.content[:500]}",
        )
    summary = "\n\n".join(parts)
    if len(evidence_items) > max_items:
        summary += f"\n\n... and {len(evidence_items) - max_items} more items."
    return summary


# ---------------------------------------------------------------------------
# Node function
# ---------------------------------------------------------------------------


async def hermes_analyze(state: ResearchState) -> dict[str, Any]:
    """Analysis node: evaluate evidence completeness and detect gaps.

    Produces a ``GapReport`` for the current section and increments
    ``iteration_count``.
    """
    settings = get_settings()
    outline: list[SectionOutline] = state.get("outline", [])
    current_idx: int = state.get("current_section_idx", 0)

    if not outline or current_idx >= len(outline):
        logger.warning("Hermes analyze called with empty outline or invalid section index.")
        return {
            "status": "error",
            "error_log": state.get("error_log", []) + ["No outline sections available to analyze"],
        }
    section = outline[current_idx]
    iteration: int = state.get("iteration_count", 0)

    logger.info(
        "Hermes analyzing section %d/%d — %s  (iteration %d)",
        current_idx + 1,
        len(outline),
        section.title,
        iteration + 1,
    )

    # Collect evidence (all gathered so far — Hermes evaluates holistically).
    all_evidence: list[Evidence] = state.get("evidence", [])
    evidence_summary = _summarize_evidence(all_evidence)

    # Build section info block.
    section_info = (
        f"Section ID: {section.section_id}\n"
        f"Title: {section.title}\n"
        f"Description: {section.description}\n"
        f"Key Questions:\n"
        + "\n".join(f"  - {q}" for q in section.key_questions)
    )

    user_prompt = (
        f"## Section Outline\n{section_info}\n\n"
        f"## Evidence Collected ({len(all_evidence)} items)\n{evidence_summary}\n\n"
        f"Current iteration: {iteration + 1}"
    )

    messages = [
        SystemMessage(content=HERMES_SYSTEM_PROMPT),
        HumanMessage(content=user_prompt),
    ]

    try:
        raw = await _invoke_with_fallback(
            messages,
            timeout=settings.AGENT_TIMEOUT_SECONDS,
        )
        parsed = _extract_json(raw)

        gap_report = GapReport(
            section_id=parsed.get("section_id", section.section_id),
            coverage_score=float(parsed.get("coverage_score", 0.0)),
            missing_topics=parsed.get("missing_topics", []),
            suggested_queries=parsed.get("suggested_queries", []),
            reasoning=parsed.get("reasoning", ""),
        )

        logger.info(
            "Hermes analysis for '%s': coverage=%.2f, gaps=%d, queries=%d",
            section.title,
            gap_report.coverage_score,
            len(gap_report.missing_topics),
            len(gap_report.suggested_queries),
        )
        logger.info("Hermes reasoning: %s", gap_report.reasoning)

        tokens_used = _estimate_tokens(raw)

        existing_gaps: list[GapReport] = list(state.get("gap_reports", []))
        existing_gaps.append(gap_report)

        coverage_status = (
            "sufficient" if gap_report.coverage_score >= settings.COVERAGE_THRESHOLD
            else "insufficient"
        )

        return {
            "gap_reports": existing_gaps,
            "iteration_count": iteration + 1,
            "token_usage": state.get("token_usage", 0) + tokens_used,
            "messages": [
                HumanMessage(
                    content=(
                        f"Hermes analysis for '{section.title}': "
                        f"coverage={gap_report.coverage_score:.2f} ({coverage_status}), "
                        f"{len(gap_report.missing_topics)} missing topics."
                    ),
                ),
            ],
        }

    except json.JSONDecodeError as exc:
        logger.error("Hermes JSON parse error: %s", exc)
        # Create a conservative gap report to force another search iteration.
        fallback_gap = GapReport(
            section_id=section.section_id,
            coverage_score=0.4,
            missing_topics=["Unable to parse analysis — assuming gaps remain"],
            suggested_queries=[q for q in section.key_questions[:3]],
            reasoning=f"Analysis parse error ({exc}); defaulting to low coverage.",
        )
        existing_gaps = list(state.get("gap_reports", []))
        existing_gaps.append(fallback_gap)

        return {
            "gap_reports": existing_gaps,
            "iteration_count": iteration + 1,
            "error_log": state.get("error_log", [])
            + [f"Hermes JSON parse error for '{section.section_id}': {exc}"],
            "messages": [
                HumanMessage(
                    content=(
                        f"Warning: Hermes analysis parse error for '{section.title}'. "
                        "Using conservative fallback gap report."
                    ),
                ),
            ],
        }

    except Exception as exc:
        logger.error("Hermes analysis failed: %s", exc)
        fallback_gap = GapReport(
            section_id=section.section_id,
            coverage_score=0.5,
            missing_topics=["Analysis failed — proceeding with available evidence"],
            suggested_queries=[],
            reasoning=f"Analysis error ({exc}); assigning moderate coverage score.",
        )
        existing_gaps = list(state.get("gap_reports", []))
        existing_gaps.append(fallback_gap)

        return {
            "gap_reports": existing_gaps,
            "iteration_count": iteration + 1,
            "error_log": state.get("error_log", [])
            + [f"Hermes analysis error for '{section.section_id}': {exc}"],
            "messages": [
                HumanMessage(
                    content=f"Hermes analysis error for '{section.title}': {exc}",
                ),
            ],
        }
