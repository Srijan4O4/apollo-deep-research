"""Athena search sub-agent — gathers evidence for the current research section."""

from __future__ import annotations

import asyncio
import json
import logging
import re
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import create_react_agent

from src.config import get_settings
from src.models.state import Evidence, ResearchState, SectionOutline
from src.prompts.athena_searcher import ATHENA_SYSTEM_PROMPT
from src.tools import (
    academic_search_tool,
    arxiv_search_tool,
    financial_search_tool,
    web_search_tool,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _build_search_queries(
    section: SectionOutline,
    suggested: list[str] | None = None,
) -> list[str]:
    """Derive concrete search queries from section metadata or gap suggestions."""
    if suggested:
        return list(suggested)
    # Derive from the section's key questions & title.
    queries: list[str] = [section.title]
    queries.extend(section.key_questions)
    return queries


def _parse_evidence_from_tool_messages(
    messages: list[Any],
    source_type_default: str = "web",
) -> list[Evidence]:
    """Extract ``Evidence`` objects from ReAct agent tool-call result messages.

    The function inspects every message for tool-call results and tries to
    parse them into ``Evidence`` items.  It is intentionally lenient — if a
    particular message cannot be parsed it is skipped with a warning.
    """
    evidence_items: list[Evidence] = []
    now = datetime.now(tz=timezone.utc)

    for msg in messages:
        # LangChain tool messages expose content as a string.
        content = getattr(msg, "content", None)
        if content is None or not isinstance(content, str):
            continue

        # Determine source type from tool name when available.
        tool_name: str = getattr(msg, "name", "") or ""
        if "academic" in tool_name:
            source_type = "academic"
        elif "financial" in tool_name:
            source_type = "financial"
        else:
            source_type = source_type_default

        # Try JSON parsing first (tools may return structured JSON).
        try:
            data = json.loads(content)
            if isinstance(data, list):
                for item in data:
                    evidence_items.append(
                        _dict_to_evidence(item, source_type, now),
                    )
                continue
            if isinstance(data, dict):
                # Might contain a "results" key.
                results = data.get("results", data.get("data", [data]))
                if isinstance(results, list):
                    for item in results:
                        evidence_items.append(
                            _dict_to_evidence(item, source_type, now),
                        )
                else:
                    evidence_items.append(
                        _dict_to_evidence(data, source_type, now),
                    )
                continue
        except (json.JSONDecodeError, TypeError):
            pass

        # Fallback: only create an evidence item if a valid URL can be extracted
        urls = re.findall(r"https?://[^\s\]\)\>]+", content)
        if urls and len(content.strip()) > 20:
            evidence_items.append(
                Evidence(
                    content=content.strip()[:5000],
                    source_url=urls[0],
                    source_type=source_type,  # type: ignore[arg-type]
                    title=f"Extracted Reference ({urls[0].split('/')[2]})",
                    relevance_score=0.6,
                    retrieved_at=now,
                    author="Web Source",
                    date="",
                    metadata={"raw": True},
                ),
            )

    return evidence_items


def _dict_to_evidence(
    item: Any,
    source_type: str,
    retrieved_at: datetime,
) -> Evidence:
    """Convert an arbitrary dict-like object to an ``Evidence`` instance."""
    if not isinstance(item, dict):
        return Evidence(
            content=str(item)[:5000],
            source_url="",
            source_type=source_type,  # type: ignore[arg-type]
            title="Parsed result",
            relevance_score=0.5,
            retrieved_at=retrieved_at,
            metadata={},
            author="Unknown",
            date="",
        )

    # Extract author and publication year
    author = item.get("author", "")
    if not author or author == "Unknown":
        authors = item.get("authors", [])
        if authors:
            author = ", ".join(authors[:3]) + (" et al." if len(authors) > 3 else "")
        elif source_type == "web":
            author = "Web Source"
        else:
            author = "Unknown"

    date = str(item.get("date") or item.get("year", "")).strip()

    source_url = str(item.get("source_url", item.get("url", item.get("link", "")))).strip()
    arxiv_id = str(item.get("arxiv_id", "")).strip()
    if not source_url and arxiv_id:
        source_url = f"https://arxiv.org/abs/{arxiv_id}"

    title = str(item.get("title", "Untitled"))[:500].strip()

    return Evidence(
        content=str(item.get("content", item.get("snippet", item.get("abstract", ""))))[:5000],
        source_url=source_url,
        source_type=item.get("source_type", source_type),
        title=title,
        relevance_score=float(item.get("relevance_score", 0.5)),
        retrieved_at=retrieved_at,
        author=author,
        date=date,
        metadata={k: v for k, v in item.items() if k not in {
            "content", "snippet", "abstract", "source_url", "url", "link",
            "source_type", "title", "relevance_score", "author", "date",
        }},
    )


def _estimate_tokens(text: str) -> int:
    """Rough token estimate (≈ 4 characters per token)."""
    return max(1, len(text) // 4)


# ---------------------------------------------------------------------------
# Node function
# ---------------------------------------------------------------------------


async def athena_search(state: ResearchState) -> dict[str, Any]:
    """Search node: gather evidence for the current section.

    Uses a LangGraph ReAct agent armed with three search tools to discover
    and retrieve evidence.  On the first iteration for a section the queries
    are derived from the section's key questions; on subsequent iterations
    they come from the latest ``GapReport.suggested_queries``.
    """
    settings = get_settings()
    outline: list[SectionOutline] = state.get("outline", [])
    current_idx: int = state.get("current_section_idx", 0)

    if not outline or current_idx >= len(outline):
        logger.warning("Athena search called with empty outline or invalid section index.")
        return {
            "status": "error",
            "error_log": state.get("error_log", []) + ["No outline sections available to search"],
        }
    section = outline[current_idx]

    logger.info(
        "Athena searching for section %d/%d — %s",
        current_idx + 1,
        len(outline),
        section.title,
    )

    # Determine queries --------------------------------------------------
    gap_reports = state.get("gap_reports", [])
    current_gaps = [
        g for g in gap_reports if g.section_id == section.section_id
    ]

    if current_gaps:
        # Subsequent iteration — use gap-report suggested queries.
        suggested = current_gaps[-1].suggested_queries
        queries = _build_search_queries(section, suggested=suggested)
        logger.info(
            "Using %d gap-report suggested queries.", len(queries),
        )
    else:
        # First iteration.
        queries = _build_search_queries(section)
        logger.info(
            "Using %d initial queries from section outline.", len(queries),
        )

    # Build the ReAct search agent ----------------------------------------
    try:
        llm = ChatGoogleGenerativeAI(
            model=settings.PRIMARY_MODEL,
            google_api_key=settings.GEMINI_API_KEY,
            temperature=0.2,
        )
    except Exception as exc:
        logger.warning("Primary LLM init failed (%s); using fallback.", exc)
        from langchain_openai import ChatOpenAI

        llm = ChatOpenAI(
            model=settings.FALLBACK_MODEL,
            api_key=settings.OPENAI_API_KEY,
            temperature=0.2,
        )

    tools = [arxiv_search_tool, academic_search_tool, web_search_tool, financial_search_tool]
    react_agent = create_react_agent(
        model=llm,
        tools=tools,
        prompt=ATHENA_SYSTEM_PROMPT,
    )

    # Compose the task message --------------------------------------------
    queries_text = "\n".join(f"  - {q}" for q in queries)
    task_message = (
        f"## Research Task\n"
        f"Section: {section.title}\n"
        f"Description: {section.description}\n\n"
        f"Key Questions:\n"
        + "\n".join(f"  - {q}" for q in section.key_questions)
        + f"\n\nSuggested queries to execute:\n{queries_text}"
    )

    # Invoke the agent with timeout ---------------------------------------
    new_evidence: list[Evidence] = []
    try:
        result = await asyncio.wait_for(
            react_agent.ainvoke(
                {"messages": [HumanMessage(content=task_message)]},
            ),
            timeout=settings.AGENT_TIMEOUT_SECONDS,
        )

        result_messages = result.get("messages", [])
        new_evidence = _parse_evidence_from_tool_messages(result_messages)

        # Fallback: if tool message parsing yielded no items, execute direct search
        if not new_evidence and queries:
            from src.tools.arxiv_search import ArxivSearchInput, arxiv_search
            from src.tools.web_search import WebSearchInput, web_search
            try:
                arxiv_ev = await arxiv_search(ArxivSearchInput(query=queries[0], max_results=5))
                new_evidence.extend(arxiv_ev)
            except Exception as arxiv_err:
                logger.warning("Direct arXiv fallback error: %s", arxiv_err)
            try:
                direct_ev = await web_search(WebSearchInput(query=queries[0], max_results=5))
                new_evidence.extend(direct_ev)
            except Exception as direct_err:
                logger.warning("Direct web search fallback error: %s", direct_err)

        tokens_text = " ".join(
            str(getattr(m, "content", "")) for m in result_messages
        )
        tokens_used = _estimate_tokens(tokens_text)

        logger.info(
            "Athena found %d evidence items for section '%s'.",
            len(new_evidence),
            section.title,
        )

    except asyncio.TimeoutError:
        logger.error(
            "Athena search timed out after %ds for section '%s'.",
            settings.AGENT_TIMEOUT_SECONDS,
            section.title,
        )
        tokens_used = 0
        fallback_ev: list[Evidence] = []
        if queries:
            from src.tools.web_search import WebSearchInput, web_search
            try:
                fallback_ev = await web_search(WebSearchInput(query=queries[0], max_results=3))
            except Exception:
                pass
        return {
            "evidence": list(state.get("evidence", [])) + fallback_ev,
            "error_log": state.get("error_log", [])
            + [
                f"Athena search timeout for section '{section.section_id}' "
                f"after {settings.AGENT_TIMEOUT_SECONDS}s",
            ],
            "status": "analyzing",
            "messages": [
                HumanMessage(
                    content=(
                        f"Athena search timed out for section '{section.title}'. "
                        "Proceeding with available evidence."
                    ),
                ),
            ],
        }
    except Exception as exc:
        logger.error("Athena search failed: %s", exc)
        tokens_used = 0
        fallback_ev = []
        if queries:
            from src.tools.web_search import WebSearchInput, web_search
            try:
                fallback_ev = await web_search(WebSearchInput(query=queries[0], max_results=3))
            except Exception:
                pass
        return {
            "evidence": list(state.get("evidence", [])) + fallback_ev,
            "error_log": state.get("error_log", [])
            + [f"Athena search error for '{section.section_id}': {exc}"],
            "status": "analyzing",
            "messages": [
                HumanMessage(
                    content=f"Athena search error for '{section.title}': {exc}",
                ),
            ],
        }

    # Merge new evidence with existing -----------------------------------
    existing_evidence: list[Evidence] = list(state.get("evidence", []))
    existing_evidence.extend(new_evidence)

    return {
        "evidence": existing_evidence,
        "status": "analyzing",
        "token_usage": state.get("token_usage", 0) + tokens_used,
        "messages": [
            HumanMessage(
                content=(
                    f"Athena gathered {len(new_evidence)} evidence items "
                    f"for section '{section.title}' "
                    f"(total evidence: {len(existing_evidence)})."
                ),
            ),
        ],
    }
