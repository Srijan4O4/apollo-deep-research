"""Streamlit frontend for the Apollo Deep Research Agent.

Provides an interactive web UI for submitting research queries,
monitoring real-time progress, and downloading generated reports.
"""

from __future__ import annotations

import json
import time
from typing import Any

import requests
import streamlit as st

BACKEND_URL = "http://localhost:8000"
API_BASE = f"{BACKEND_URL}/api/v1"
WS_BASE = "ws://localhost:8000/api/v1"


def configure_page() -> None:
    """Configure the Streamlit page settings."""
    st.set_page_config(
        page_title="Apollo Deep Research Agent",
        page_icon="🚀",
        layout="wide",
        initial_sidebar_state="expanded",
    )


def render_sidebar() -> dict[str, Any]:
    """Render the sidebar with configuration settings.

    Returns:
        Dictionary containing user-configured settings.
    """
    with st.sidebar:
        st.header("⚙️ Settings")
        st.divider()

        st.subheader("API Keys")
        gemini_default = getattr(st, "secrets", {}).get("GEMINI_API_KEY", "")
        tavily_default = getattr(st, "secrets", {}).get("TAVILY_API_KEY", "")
        openai_default = getattr(st, "secrets", {}).get("OPENAI_API_KEY", "")

        gemini_key = st.text_input(
            "Gemini API Key",
            value=gemini_default,
            type="password",
            help="Required for primary LLM access",
            key="gemini_key",
        )
        openai_key = st.text_input(
            "OpenAI API Key (Optional)",
            value=openai_default,
            type="password",
            help="Used as fallback LLM",
            key="openai_key",
        )
        tavily_key = st.text_input(
            "Tavily API Key",
            value=tavily_default,
            type="password",
            help="Required for web search",
            key="tavily_key",
        )

        st.divider()
        st.subheader("Research Parameters")

        max_sections = st.slider(
            "Max Sections",
            min_value=3,
            max_value=10,
            value=8,
            help="Maximum number of sections in the report",
            key="max_sections",
        )

        max_iterations = st.slider(
            "Max Iterations",
            min_value=1,
            max_value=10,
            value=5,
            help="Maximum research iterations per section",
            key="max_iterations",
        )

        st.divider()
        st.caption("🚀 Apollo Deep Research Agent v1.0.0")

        return {
            "gemini_key": gemini_key,
            "openai_key": openai_key,
            "tavily_key": tavily_key,
            "max_sections": max_sections,
            "max_iterations": max_iterations,
        }


def check_backend_health() -> bool:
    """Check if the backend API is healthy.

    Returns:
        True if backend is reachable and healthy, False otherwise.
    """
    try:
        response = requests.get(f"{BACKEND_URL}/health", timeout=5)
        return response.status_code == 200
    except requests.ConnectionError:
        return False
    except Exception:
        return False


def submit_research(query: str, settings: dict[str, Any]) -> str | None:
    """Submit a research request to the backend API.

    Args:
        query: The research query string.
        settings: Configuration settings from the sidebar.

    Returns:
        The task_id if successful, None otherwise.
    """
    try:
        payload = {
            "query": query,
            "max_sections": settings["max_sections"],
            "max_iterations": settings["max_iterations"],
        }

        response = requests.post(
            f"{API_BASE}/research",
            json=payload,
            timeout=30,
        )
        response.raise_for_status()

        data = response.json()
        return data.get("task_id")

    except requests.ConnectionError:
        st.error("❌ Cannot connect to backend. Make sure the API server is running on port 8000.")
        return None
    except requests.HTTPError as exc:
        st.error(f"❌ API error: {exc.response.status_code} — {exc.response.text}")
        return None
    except Exception as exc:
        st.error(f"❌ Unexpected error: {exc}")
        return None


def poll_progress(task_id: str) -> dict[str, Any] | None:
    """Poll the backend for task progress.

    Args:
        task_id: The research task identifier.

    Returns:
        Progress data dictionary, or None on error.
    """
    try:
        response = requests.get(f"{API_BASE}/research/{task_id}", timeout=10)
        if response.status_code == 200:
            return response.json()
        if response.status_code == 500:
            return {"status": "error", "detail": response.json().get("detail", "Unknown error")}
        return None
    except Exception:
        return None


def fetch_report(task_id: str) -> dict[str, Any] | None:
    """Fetch the completed research report.

    Args:
        task_id: The research task identifier.

    Returns:
        Report data dictionary, or None if not ready.
    """
    try:
        response = requests.get(f"{API_BASE}/research/{task_id}/report", timeout=30)
        if response.status_code == 200:
            return response.json()
        return None
    except Exception:
        return None


def monitor_research(task_id: str) -> dict[str, Any] | None:
    """Monitor research progress using HTTP polling with real-time UI updates.

    Uses st.status and progress bar for a responsive experience.
    Falls back to HTTP polling since Streamlit doesn't natively support
    WebSocket consumption.

    Args:
        task_id: The research task identifier.

    Returns:
        The final report data, or None if an error occurred.
    """
    progress_bar = st.progress(0, text="Initializing research...")
    status_container = st.status("🔬 Research in progress...", expanded=True)
    sections_container = st.container()

    completed_sections: list[str] = []
    prev_sections_completed = 0

    with status_container:
        status_placeholder = st.empty()
        while True:
            progress_data = poll_progress(task_id)

            if progress_data is None:
                status_placeholder.warning("⏳ Waiting for backend response...")
                time.sleep(1)
                continue

            status = progress_data.get("status", "unknown")

            if status == "error":
                detail = progress_data.get("detail", "Unknown error")
                status_placeholder.error(f"❌ Research failed: {detail}")
                progress_bar.empty()
                return None

            current_section = progress_data.get("current_section", "")
            sections_completed = progress_data.get("sections_completed", 0)
            total_sections = progress_data.get("total_sections", 1)
            iteration = progress_data.get("iteration", 0)
            evidence_count = progress_data.get("evidence_count", 0)

            status_labels = {
                "pending": "⏳ Initializing task...",
                "planning": "📋 Apollo: Planning structured research outline...",
                "searching": "🔍 Athena: Gathering multi-source evidence...",
                "analyzing": "🔬 Hermes: Evaluating coverage & identifying gaps...",
                "synthesizing": "✍️ Apollo: Synthesizing section drafts with citations...",
                "assembling": "📄 Apollo: Assembling final comprehensive report...",
                "complete": "✅ Research complete!",
            }

            status_text = status_labels.get(status, f"Status: {status}")

            status_placeholder.markdown(
                f"### {status_text}\n\n"
                f"- **Current Section:** `{current_section or 'Planning...'}`\n"
                f"- **Completed Sections:** `{sections_completed} / {total_sections}`\n"
                f"- **Evidence Items Gathered:** `{evidence_count}` sources\n"
                f"- **Current Loop Iteration:** `{iteration}`"
            )

            if total_sections > 0:
                pct = min(sections_completed / total_sections, 1.0)
                progress_bar.progress(
                    pct,
                    text=f"Sections: {sections_completed}/{total_sections}",
                )

            if sections_completed > prev_sections_completed and current_section:
                completed_sections.append(current_section)
                prev_sections_completed = sections_completed

            if status == "complete":
                progress_bar.progress(1.0, text="✅ Complete!")
                status_container.update(label="✅ Research Finished!", state="complete", expanded=False)
                break

            time.sleep(1.0)

    if completed_sections:
        with sections_container:
            for section_title in completed_sections:
                with st.expander(f"✅ {section_title}", expanded=False):
                    st.write("Section completed successfully.")

    status_container.update(label="✅ Research complete!", state="complete", expanded=False)

    report_data = fetch_report(task_id)
    return report_data


def render_report(report_data: dict[str, Any]) -> None:
    """Render the completed research report in the UI.

    Args:
        report_data: The report data dictionary from the API.
    """
    st.divider()
    st.header(report_data.get("title", "Research Report"))

    st.subheader("Executive Summary")
    st.markdown(report_data.get("executive_summary", ""))

    metadata = report_data.get("metadata", {})
    methodology = metadata.get("methodology", "")
    if methodology:
        with st.expander("🔬 Research Methodology & Evidence Standards", expanded=True):
            st.markdown(methodology)

    sections = report_data.get("sections", [])
    for section in sections:
        title = section.get("section_title", "Untitled Section")
        content = section.get("content", "")
        with st.expander(f"📄 {title}", expanded=True):
            st.markdown(content)

    bibliography = report_data.get("bibliography", [])
    if bibliography:
        with st.expander("📚 Bibliography", expanded=True):
            for cite in bibliography:
                number = cite.get("number", "")
                title = cite.get("title", "Untitled")
                url = cite.get("source_url", "")
                author = cite.get("author", "")
                date = cite.get("date", "")
                author_str = f"{author}. " if author and author != "Unknown" else ""
                date_str = f" ({date})." if date else "."
                st.markdown(f"[{number}] {author_str}\"{title}\"{date_str} [{url}]({url})")

    md_content = _format_report_markdown(report_data)
    st.download_button(
        label="📥 Download Report (.md)",
        data=md_content,
        file_name="apollo_research_report.md",
        mime="text/markdown",
    )

    st.divider()
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Sources", report_data.get("total_sources", 0))
    with col2:
        metadata = report_data.get("metadata", {})
        st.metric("Total Iterations", metadata.get("total_iterations", 0))
    with col3:
        st.metric("Sections", len(sections))


def _format_report_markdown(report_data: dict[str, Any]) -> str:
    """Format the report data as a downloadable markdown string.

    Args:
        report_data: The report data dictionary.

    Returns:
        Formatted markdown string.
    """
    parts: list[str] = []
    parts.append(f"# {report_data.get('title', 'Research Report')}")
    parts.append("")
    parts.append(f"*Generated: {report_data.get('generated_at', '')}*")
    parts.append("")

    parts.append("## Executive Summary")
    parts.append("")
    parts.append(report_data.get("executive_summary", ""))
    parts.append("")

    metadata = report_data.get("metadata", {})
    methodology = metadata.get("methodology", "")
    if methodology:
        parts.append("## Research Methodology & Evidence Standards")
        parts.append("")
        parts.append(methodology)
        parts.append("")

    for section in report_data.get("sections", []):
        parts.append(f"## {section.get('section_title', 'Untitled')}")
        parts.append("")
        parts.append(section.get("content", ""))
        parts.append("")

    bibliography = report_data.get("bibliography", [])
    if bibliography:
        parts.append("## Bibliography")
        parts.append("")
        for cite in bibliography:
            number = cite.get("number", "")
            title = cite.get("title", "Untitled")
            url = cite.get("source_url", "")
            author = cite.get("author", "")
            date = cite.get("date", "")
            author_str = f"{author}. " if author and author != "Unknown" else ""
            date_str = f" ({date})." if date else "."
            parts.append(f"[{number}] {author_str}\"{title}\"{date_str} Available at: {url}")
        parts.append("")

    return "\n".join(parts)


def run_in_process_research(query: str, settings: dict[str, Any]) -> dict[str, Any] | None:
    """Execute research graph directly in-process for standalone Streamlit Cloud deployments."""
    import asyncio
    import os
    import sys
    from pathlib import Path

    # Ensure project root is in sys.path
    root_dir = Path(__file__).resolve().parent.parent
    if str(root_dir) not in sys.path:
        sys.path.insert(0, str(root_dir))

    # Apply API keys from sidebar or st.secrets
    gemini_key = settings.get("gemini_key") or getattr(st, "secrets", {}).get("GEMINI_API_KEY", "")
    tavily_key = settings.get("tavily_key") or getattr(st, "secrets", {}).get("TAVILY_API_KEY", "")
    openai_key = settings.get("openai_key") or getattr(st, "secrets", {}).get("OPENAI_API_KEY", "")

    if gemini_key:
        os.environ["GEMINI_API_KEY"] = gemini_key
    if tavily_key:
        os.environ["TAVILY_API_KEY"] = tavily_key
    if openai_key:
        os.environ["OPENAI_API_KEY"] = openai_key

    if not os.environ.get("GEMINI_API_KEY"):
        st.error("❌ Gemini API Key is required. Please enter it in the sidebar or app secrets.")
        return None

    from src.agents.graph import build_research_graph
    from src.services.citation_manager import CitationManager
    from src.services.report_generator import ReportGenerator

    progress_bar = st.progress(0, text="Initializing research in standalone mode...")
    status_container = st.status("🔬 Autonomous Research in Progress...", expanded=True)
    status_placeholder = status_container.empty()

    async def _run():
        initial_state = {
            "query": query,
            "outline": [],
            "evidence": [],
            "gap_reports": [],
            "section_drafts": [],
            "current_section_idx": 0,
            "iteration_count": 0,
            "max_iterations": settings.get("max_iterations", 3),
            "status": "planning",
            "messages": [],
            "error_log": [],
            "token_usage": 0,
        }

        graph = build_research_graph()
        final_state = initial_state

        async for chunk in graph.astream(initial_state, stream_mode="values"):
            final_state = chunk
            status = chunk.get("status", "planning")
            outline = chunk.get("outline", [])
            idx = chunk.get("current_section_idx", 0)
            drafts = chunk.get("section_drafts", [])
            evidence = chunk.get("evidence", [])

            sec_title = outline[idx].title if outline and idx < len(outline) else "Planning Research Itinerary"
            status_labels = {
                "planning": "📋 Apollo: Planning structured research outline...",
                "searching": "🔍 Athena: Discovering academic & web sources...",
                "analyzing": "🔬 Hermes: Auditing coverage & empirical benchmarks...",
                "synthesized": "✍️ Apollo: Synthesizing section drafts with citations...",
                "complete": "🛡️ Apollo QA: Auditing pre-publication citations & tables...",
            }
            status_text = status_labels.get(status, f"Status: {status.upper()}")

            status_placeholder.markdown(
                f"### {status_text}\n\n"
                f"- **Current Section:** `{sec_title}`\n"
                f"- **Sections Synthesized:** `{len(drafts)} / {len(outline)}`\n"
                f"- **Verified Sources Gathered:** `{len(evidence)}`"
            )
            if outline:
                progress_bar.progress(min(len(drafts) / max(len(outline), 1), 1.0))

        status_placeholder.markdown("### 📄 Generating Final Consolidated Report...")
        citation_mgr = CitationManager()
        generator = ReportGenerator(citation_mgr)
        report = await generator.generate_report(final_state)
        return report.model_dump(mode="json")

    try:
        report_data = asyncio.run(_run())
        progress_bar.progress(1.0, text="✅ Research Complete!")
        status_container.update(label="✅ Research Complete!", state="complete", expanded=False)
        return report_data
    except Exception as exc:
        status_placeholder.error(f"❌ Standalone research execution failed: {exc}")
        return None


def main() -> None:
    """Main entry point for the Streamlit application."""
    configure_page()

    st.title("🚀 Apollo Deep Research Agent")
    st.caption("AI-powered multi-agent system for comprehensive, citation-backed research reports")

    settings = render_sidebar()

    backend_healthy = check_backend_health()
    if not backend_healthy:
        st.info("ℹ️ Running in Standalone In-Process Mode (Streamlit Cloud). Direct agent execution enabled.")

    st.subheader("📝 Research Query")
    query = st.text_area(
        "Enter your research question or topic",
        height=120,
        placeholder=(
            "e.g., What are the latest advances in quantum computing and their "
            "potential impact on cryptography and cybersecurity?"
        ),
        key="research_query",
    )

    col1, col2 = st.columns([1, 5])
    with col1:
        start_button = st.button("🔬 Start Research", type="primary", use_container_width=True)

    if start_button:
        if not query or not query.strip():
            st.error("❌ Please enter a research query.")
            return

        if not backend_healthy:
            report_data = run_in_process_research(query.strip(), settings)
            if report_data is not None:
                render_report(report_data)
            return

        task_id = submit_research(query.strip(), settings)

        if task_id is None:
            return

        st.info(f"🆔 Task ID: `{task_id}`")

        report_data = monitor_research(task_id)

        if report_data is not None:
            render_report(report_data)
        else:
            st.error("❌ Failed to retrieve the research report.")


if __name__ == "__main__":
    main()
