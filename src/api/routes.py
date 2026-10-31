"""FastAPI router defining all research API endpoints and WebSocket streaming."""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
import uuid
from typing import Any

from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel

from src.agents.graph import build_research_graph
from src.models.requests import ResearchRequest
from src.models.responses import (
    ResearchProgress,
    ResearchReport,
    ResearchTaskResponse,
    TaskStatus,
)
from src.models.state import ResearchState
from src.services.citation_manager import CitationManager
from src.services.report_generator import ReportGenerator

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1")

tasks: dict[str, dict[str, Any]] = {}


async def _run_research(task_id: str, request: ResearchRequest) -> None:
    """Execute the research graph in the background and update task state.

    Initializes a ResearchState from the request, invokes the compiled
    LangGraph research graph, then generates the final report.

    Args:
        task_id: Unique identifier for this research task.
        request: The validated ResearchRequest from the user.
    """
    try:
        tasks[task_id]["status"] = TaskStatus.PLANNING
        logger.info("Task %s: Starting research for query: %s", task_id, request.query)

        initial_state: dict[str, Any] = {
            "query": request.query,
            "outline": [],
            "evidence": [],
            "gap_reports": [],
            "section_drafts": [],
            "current_section_idx": 0,
            "iteration_count": 0,
            "max_iterations": request.max_iterations,
            "status": "planning",
            "messages": [],
            "error_log": [],
            "token_usage": 0,
        }

        tasks[task_id]["state"] = initial_state
        tasks[task_id]["status"] = TaskStatus.PLANNING

        graph = build_research_graph()

        final_state = initial_state
        async for chunk in graph.astream(initial_state, stream_mode="values"):
            final_state = chunk
            tasks[task_id]["state"] = chunk
            chunk_status = chunk.get("status")
            if chunk_status:
                try:
                    tasks[task_id]["status"] = TaskStatus(chunk_status.lower())
                except (ValueError, KeyError):
                    pass

        tasks[task_id]["state"] = final_state
        tasks[task_id]["status"] = TaskStatus.ASSEMBLING

        logger.info("Task %s: Research complete, generating report", task_id)

        citation_manager = CitationManager()
        report_generator = ReportGenerator(citation_manager)
        report = await report_generator.generate_report(final_state)

        report = ResearchReport(
            task_id=task_id,
            query=report.query,
            title=report.title,
            executive_summary=report.executive_summary,
            sections=report.sections,
            bibliography=report.bibliography,
            generated_at=report.generated_at,
            total_sources=report.total_sources,
            metadata=report.metadata,
        )

        tasks[task_id]["result"] = report
        tasks[task_id]["status"] = TaskStatus.COMPLETE
        logger.info("Task %s: Report generation complete", task_id)

        try:
            reports_dir = Path("reports")
            reports_dir.mkdir(exist_ok=True)
            md_path = reports_dir / f"{task_id}.md"
            md_content = report_generator.format_as_markdown(report)
            md_path.write_text(md_content, encoding="utf-8")
            logger.info("Task %s: Saved report markdown to %s", task_id, md_path)
        except Exception as save_err:
            logger.warning("Task %s: Could not persist report to disk: %s", task_id, save_err)

    except Exception as exc:
        logger.exception("Task %s: Research failed with error: %s", task_id, exc)
        tasks[task_id]["status"] = TaskStatus.ERROR
        tasks[task_id]["error"] = str(exc)


def _build_progress(task_id: str) -> ResearchProgress:
    """Build a ResearchProgress object from the current task state.

    Args:
        task_id: The task identifier.

    Returns:
        A ResearchProgress representing the current state.
    """
    task = tasks[task_id]
    state: dict[str, Any] = task.get("state", {})

    outline = state.get("outline", [])
    section_drafts = state.get("section_drafts", [])
    current_idx = state.get("current_section_idx", 0)

    current_section: str | None = None
    if outline and 0 <= current_idx < len(outline):
        section = outline[current_idx]
        current_section = section.title if hasattr(section, "title") else str(section)

    return ResearchProgress(
        task_id=task_id,
        status=task["status"],
        current_section=current_section,
        sections_completed=len(section_drafts),
        total_sections=len(outline),
        iteration=state.get("iteration_count", 0),
        evidence_count=len(state.get("evidence", [])),
    )


@router.post("/research", response_model=ResearchTaskResponse)
async def start_research(request: ResearchRequest) -> ResearchTaskResponse:
    """Launch a new research task in the background.

    Accepts a ResearchRequest, generates a unique task ID, and starts
    the research graph execution as an asyncio background task.

    Args:
        request: The research request containing query and parameters.

    Returns:
        A ResearchTaskResponse with the task_id and initial status.
    """
    task_id = str(uuid.uuid4())

    tasks[task_id] = {
        "status": TaskStatus.PENDING,
        "state": {},
        "result": None,
        "error": None,
    }

    logger.info("Creating research task %s for query: %s", task_id, request.query)

    asyncio.create_task(_run_research(task_id, request))

    return ResearchTaskResponse(
        task_id=task_id,
        status=TaskStatus.PENDING,
        message=f"Research task created. Track progress at /api/v1/research/{task_id}",
    )


@router.get("/research/{task_id}", response_model=ResearchProgress)
async def get_research_progress(task_id: str) -> ResearchProgress:
    """Retrieve the current progress of a research task.

    Args:
        task_id: The unique identifier of the research task.

    Returns:
        A ResearchProgress object with current status details.

    Raises:
        HTTPException: If the task_id is not found (404).
    """
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found")

    task = tasks[task_id]

    if task["status"] == TaskStatus.ERROR:
        raise HTTPException(
            status_code=500,
            detail=f"Task {task_id} failed: {task.get('error', 'Unknown error')}",
        )

    return _build_progress(task_id)


@router.get("/research/{task_id}/report", response_model=ResearchReport)
async def get_research_report(task_id: str) -> ResearchReport:
    """Retrieve the completed research report.

    Args:
        task_id: The unique identifier of the research task.

    Returns:
        The completed ResearchReport.

    Raises:
        HTTPException: If task not found (404), not yet complete (202), or errored (500).
    """
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found")

    task = tasks[task_id]

    if task["status"] == TaskStatus.ERROR:
        raise HTTPException(
            status_code=500,
            detail=f"Task {task_id} failed: {task.get('error', 'Unknown error')}",
        )

    if task["status"] != TaskStatus.COMPLETE:
        raise HTTPException(
            status_code=202,
            detail=f"Task {task_id} is still in progress (status: {task['status'].value})",
        )

    if task["result"] is None:
        raise HTTPException(
            status_code=500,
            detail=f"Task {task_id} completed but no report was generated",
        )

    return task["result"]


@router.websocket("/research/{task_id}/stream")
async def stream_research_progress(websocket: WebSocket, task_id: str) -> None:
    """Stream real-time research progress updates via WebSocket.

    Sends JSON events to the client at ~0.5s intervals until the task
    completes or an error occurs. Event types:
      - progress: Current task status and section info
      - section_complete: Notification when a section finishes
      - complete: The final research report
      - error: Error information if the task fails

    Args:
        websocket: The WebSocket connection.
        task_id: The unique identifier of the research task.
    """
    await websocket.accept()

    if task_id not in tasks:
        await websocket.send_json({"event": "error", "data": {"message": f"Task {task_id} not found"}})
        await websocket.close(code=4004)
        return

    previous_sections_completed = 0
    previous_status: TaskStatus | None = None

    try:
        while True:
            task = tasks.get(task_id)
            if task is None:
                await websocket.send_json({"event": "error", "data": {"message": "Task disappeared"}})
                break

            current_status = task["status"]

            progress = _build_progress(task_id)
            progress_data = progress.model_dump(mode="json")

            if current_status != previous_status:
                await websocket.send_json({"event": "progress", "data": progress_data})
                previous_status = current_status

            if progress.sections_completed > previous_sections_completed:
                state = task.get("state", {})
                section_drafts = state.get("section_drafts", [])
                for i in range(previous_sections_completed, progress.sections_completed):
                    if i < len(section_drafts):
                        draft = section_drafts[i]
                        section_title = draft.section_title if hasattr(draft, "section_title") else str(draft)
                        await websocket.send_json({
                            "event": "section_complete",
                            "data": {"section": section_title},
                        })
                previous_sections_completed = progress.sections_completed

            if current_status == TaskStatus.COMPLETE:
                result = task.get("result")
                if result is not None:
                    report_data = result.model_dump(mode="json")
                    await websocket.send_json({"event": "complete", "data": report_data})
                break

            if current_status == TaskStatus.ERROR:
                await websocket.send_json({
                    "event": "error",
                    "data": {"message": task.get("error", "Unknown error")},
                })
                break

            await asyncio.sleep(0.5)

    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected for task %s", task_id)
    except Exception as exc:
        logger.exception("WebSocket error for task %s: %s", task_id, exc)
        try:
            await websocket.send_json({"event": "error", "data": {"message": str(exc)}})
        except Exception:
            pass
    finally:
        try:
            await websocket.close()
        except Exception:
            pass
