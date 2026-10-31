"""Apollo supervisor agent — planning, section synthesis, and report assembly nodes."""

from __future__ import annotations

import asyncio
import json
import logging
import re
from datetime import datetime, timezone
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI

from src.config import get_settings
from src.models.state import (
    Citation,
    ResearchState,
    SectionDraft,
    SectionOutline,
)
from src.prompts.apollo_planner import APOLLO_PLANNER_PROMPT
from src.prompts.apollo_synthesizer import (
    APOLLO_REPORT_ASSEMBLER_PROMPT,
    APOLLO_SECTION_SYNTHESIZER_PROMPT,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _build_primary_llm() -> ChatGoogleGenerativeAI:
    """Instantiate the primary Gemini LLM."""
    settings = get_settings()
    return ChatGoogleGenerativeAI(
        model=settings.PRIMARY_MODEL,
        google_api_key=settings.GEMINI_API_KEY,
        temperature=0.3,
    )


def _build_fallback_llm() -> Any:
    """Instantiate fallback LLM (OpenAI if key is configured, else secondary Gemini model)."""
    settings = get_settings()
    if settings.OPENAI_API_KEY:
        return ChatOpenAI(
            model=settings.FALLBACK_MODEL if "gpt" in settings.FALLBACK_MODEL else "gpt-4o",
            api_key=settings.OPENAI_API_KEY,
            temperature=0.3,
        )
    fallback_model = settings.FALLBACK_MODEL if "gemini" in settings.FALLBACK_MODEL else "gemini-3.1-flash-lite-preview"
    return ChatGoogleGenerativeAI(
        model=fallback_model,
        google_api_key=settings.GEMINI_API_KEY,
        temperature=0.3,
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
    """Invoke the primary LLM; fall back to secondary on failure.

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
            "Primary LLM failed (%s). Switching to fallback.", primary_err,
        )
        try:
            llm = _build_fallback_llm()
            response = await asyncio.wait_for(
                llm.ainvoke(messages),
                timeout=timeout,
            )
            return _extract_text_content(response.content)
        except Exception as fallback_err:
            logger.error("Fallback LLM also failed: %s", fallback_err)
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


# ---------------------------------------------------------------------------
# Node functions
# ---------------------------------------------------------------------------


async def apollo_plan(state: ResearchState) -> dict[str, Any]:
    """Planning node: decompose the user's query into a research outline.

    Produces a ``list[SectionOutline]`` stored under the ``outline`` key.
    """
    settings = get_settings()
    query: str = state["query"]

    logger.info("Apollo planning — decomposing query: %.120s", query)

    messages = [
        SystemMessage(content=APOLLO_PLANNER_PROMPT),
        HumanMessage(content=f"Research query:\n{query}"),
    ]

    try:
        raw = await _invoke_with_fallback(
            messages,
            timeout=settings.AGENT_TIMEOUT_SECONDS,
        )
        parsed = _extract_json(raw)

        outlines: list[SectionOutline] = [
            SectionOutline(**item) for item in parsed
        ]
        logger.info(
            "Apollo planning complete — %d sections generated.", len(outlines),
        )

        tokens_used = _estimate_tokens(raw)

        return {
            "outline": outlines,
            "status": "searching",
            "current_section_idx": 0,
            "iteration_count": 0,
            "token_usage": state.get("token_usage", 0) + tokens_used,
            "messages": [
                HumanMessage(
                    content=(
                        f"Research outline created with {len(outlines)} sections."
                    ),
                ),
            ],
        }

    except json.JSONDecodeError as exc:
        logger.error("Failed to parse planner output as JSON: %s", exc)
        return {
            "error_log": state.get("error_log", [])
            + [f"Planner JSON parse error: {exc}"],
            "status": "error",
            "messages": [
                HumanMessage(content="Error: could not parse research outline."),
            ],
        }
    except Exception as exc:
        logger.error("Apollo planning failed: %s", exc)
        return {
            "error_log": state.get("error_log", [])
            + [f"Planner error: {exc}"],
            "status": "error",
            "messages": [
                HumanMessage(content=f"Planning error: {exc}"),
            ],
        }


async def apollo_synthesize_section(state: ResearchState) -> dict[str, Any]:
    """Synthesis node: generate content for the current section from evidence.

    Appends a ``SectionDraft`` to the ``section_drafts`` list.
    """
    settings = get_settings()
    outline: list[SectionOutline] = state.get("outline", [])
    current_idx: int = state.get("current_section_idx", 0)

    if not outline or current_idx >= len(outline):
        logger.warning("Apollo synthesize called with empty outline or invalid index.")
        return {
            "status": "error",
            "error_log": state.get("error_log", []) + ["Invalid section index for synthesis"],
        }

    section = outline[current_idx]

    logger.info(
        "Apollo synthesising section %d/%d — %s",
        current_idx + 1,
        len(outline),
        section.title,
    )

    # Collect evidence relevant to this section by keyword matching.
    all_evidence = state.get("evidence", [])
    section_keywords = {
        section.title.lower(),
        section.section_id.lower(),
        *[q.lower() for q in section.key_questions],
    }

    relevant_evidence = []
    for ev in all_evidence:
        text_lower = (ev.content + " " + ev.title).lower()
        if any(kw in text_lower for kw in section_keywords):
            relevant_evidence.append(ev)

    # If keyword matching was too strict, use all collected evidence
    if not relevant_evidence and all_evidence:
        relevant_evidence = all_evidence[:10]

    # Fall back to all evidence if nothing matched (better than nothing).
    if not relevant_evidence:
        relevant_evidence = list(all_evidence)

    # Build the evidence block for the prompt with full author and publication data.
    evidence_text_parts: list[str] = []
    for idx, ev in enumerate(relevant_evidence, start=1):
        author_str = ev.author if ev.author and ev.author != "Unknown" else (
            ", ".join(ev.metadata.get("authors", [])[:3]) if ev.metadata.get("authors") else "Unknown"
        )
        year_str = ev.date or str(ev.metadata.get("year", ""))
        evidence_text_parts.append(
            f"[{idx}] Title: {ev.title}\n"
            f"    Author(s): {author_str}\n"
            f"    Year: {year_str}\n"
            f"    Source: {ev.source_url}\n"
            f"    Type: {ev.source_type}\n"
            f"    Content: {ev.content[:2000]}",
        )
    evidence_block = "\n\n".join(evidence_text_parts) or "(No evidence available.)"

    section_info = (
        f"Section ID: {section.section_id}\n"
        f"Title: {section.title}\n"
        f"Description: {section.description}\n"
        f"Key Questions:\n"
        + "\n".join(f"  - {q}" for q in section.key_questions)
    )

    user_prompt = (
        f"## Section Outline\n{section_info}\n\n"
        f"## Evidence\n{evidence_block}"
    )

    messages = [
        SystemMessage(content=APOLLO_SECTION_SYNTHESIZER_PROMPT),
        HumanMessage(content=user_prompt),
    ]

    try:
        raw = await _invoke_with_fallback(
            messages,
            timeout=settings.AGENT_TIMEOUT_SECONDS,
        )
        parsed = _extract_json(raw)

        parsed_citations = parsed.get("citations", [])
        citations = []
        for c in parsed_citations:
            try:
                citations.append(Citation(**c))
            except Exception:
                pass

        # Auto-link citations from relevant_evidence if model omitted citations array
        content_text = parsed.get("content", "")
        if not citations and relevant_evidence:
            for idx, ev in enumerate(relevant_evidence, start=1):
                if f"[{idx}]" in content_text:
                    citations.append(
                        Citation(
                            number=idx,
                            source_url=ev.source_url,
                            title=ev.title,
                            author=ev.author,
                            date=ev.date,
                            source_type=ev.source_type,
                        )
                    )

        draft = SectionDraft(
            section_id=parsed.get("section_id", section.section_id),
            title=parsed.get("title", section.title),
            content=content_text,
            citations=citations,
        )

        tokens_used = _estimate_tokens(raw)
        existing_drafts: list[SectionDraft] = list(
            state.get("section_drafts", []),
        )
        existing_drafts.append(draft)

        logger.info(
            "Section '%s' synthesised — %d citations.", section.title, len(citations),
        )

        return {
            "section_drafts": existing_drafts,
            "status": "synthesized",
            "token_usage": state.get("token_usage", 0) + tokens_used,
            "messages": [
                HumanMessage(
                    content=f"Section '{section.title}' synthesised successfully.",
                ),
            ],
        }

    except json.JSONDecodeError as exc:
        logger.error("Section synthesis JSON parse error: %s", exc)
        # Produce a best-effort draft from the raw LLM text.
        fallback_draft = SectionDraft(
            section_id=section.section_id,
            title=section.title,
            content=raw if "raw" in dir() else "Synthesis failed.",  # noqa: F821
            citations=[],
        )
        existing_drafts = list(state.get("section_drafts", []))
        existing_drafts.append(fallback_draft)
        return {
            "section_drafts": existing_drafts,
            "error_log": state.get("error_log", [])
            + [f"Section synthesis JSON error for '{section.section_id}': {exc}"],
            "status": "synthesized",
            "messages": [
                HumanMessage(
                    content=(
                        f"Warning: section '{section.title}' used fallback "
                        f"(JSON parse error)."
                    ),
                ),
            ],
        }
    except Exception as exc:
        logger.error("Section synthesis failed: %s", exc)
        fallback_draft = SectionDraft(
            section_id=section.section_id,
            title=section.title,
            content="Section synthesis encountered an error.",
            citations=[],
        )
        existing_drafts = list(state.get("section_drafts", []))
        existing_drafts.append(fallback_draft)
        return {
            "section_drafts": existing_drafts,
            "error_log": state.get("error_log", [])
            + [f"Section synthesis error for '{section.section_id}': {exc}"],
            "status": "synthesized",
            "messages": [
                HumanMessage(
                    content=f"Error synthesising section '{section.title}': {exc}",
                ),
            ],
        }


async def apollo_assemble_report(state: ResearchState) -> dict[str, Any]:
    """Assembly node: generate executive summary, methodology, and strategic synthesis.

    Sets ``status`` to ``"complete"`` on success.
    """
    settings = get_settings()
    drafts: list[SectionDraft] = [
        d for d in state.get("section_drafts", [])
        if d.section_id not in {"__final_report__", "__assembly_metadata__"}
    ]

    logger.info("Apollo assembling final report from %d sections.", len(drafts))

    drafts_text_parts: list[str] = []
    for draft in drafts:
        drafts_text_parts.append(
            f"### Section: {draft.title} (id: {draft.section_id})\n\n"
            f"{draft.content[:2500]}"
        )
    drafts_block = "\n\n---\n\n".join(drafts_text_parts)

    messages = [
        SystemMessage(content=APOLLO_REPORT_ASSEMBLER_PROMPT),
        HumanMessage(
            content=f"## Research Query\n{state.get('query', '')}\n\n## Section Drafts Overview\n\n{drafts_block}",
        ),
    ]

    try:
        raw = await _invoke_with_fallback(
            messages,
            timeout=settings.AGENT_TIMEOUT_SECONDS,
        )
        parsed = _extract_json(raw)

        report_title: str = parsed.get("title", "Research Report")
        tokens_used = _estimate_tokens(raw)

        logger.info("Report synthesis assembly complete — title: %s", report_title)

        # Store metadata in special draft for ReportGenerator
        meta_draft = SectionDraft(
            section_id="__assembly_metadata__",
            title=report_title,
            content=json.dumps(parsed),
            citations=[],
        )
        existing_drafts = list(drafts)
        existing_drafts.append(meta_draft)

        return {
            "section_drafts": existing_drafts,
            "status": "complete",
            "token_usage": state.get("token_usage", 0) + tokens_used,
            "messages": [
                HumanMessage(
                    content=f"Final report synthesis complete for '{report_title}'.",
                ),
            ],
        }

    except Exception as exc:
        logger.error("Report assembly synthesis encountered error: %s", exc)

        meta_draft = SectionDraft(
            section_id="__assembly_metadata__",
            title="Research Report",
            content=json.dumps({
                "title": "Research Report",
                "executive_summary": "Comprehensive analysis synthesized across empirical findings and research literature.",
                "methodology": "Multi-source research conducted across arXiv, academic databases, and enterprise case studies.",
                "key_findings": [],
            }),
            citations=[],
        )
        existing_drafts = list(drafts)
        existing_drafts.append(meta_draft)

        return {
            "section_drafts": existing_drafts,
            "error_log": state.get("error_log", []) + [f"Report assembly error: {exc}"],
            "status": "complete",
            "token_usage": state.get("token_usage", 0),
            "messages": [
                HumanMessage(
                    content=f"Report assembly completed with fallback synthesis: {exc}",
                ),
            ],
        }

        return {
            "section_drafts": existing_drafts,
            "error_log": state.get("error_log", [])
            + [f"Report assembly error: {exc}"],
            "status": "complete",
            "token_usage": state.get("token_usage", 0),
            "messages": [
                HumanMessage(
                    content=(
                        "Warning: report assembly used fallback concatenation "
                        f"due to error: {exc}"
                    ),
                ),
            ],
        }
