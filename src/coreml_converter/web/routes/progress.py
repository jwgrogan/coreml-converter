from __future__ import annotations

from fastapi import APIRouter, Request
from sse_starlette.sse import EventSourceResponse

from coreml_converter.web.dependencies import render, get_build_store

router = APIRouter()


@router.get("/build/{job_id}")
async def progress_page(request: Request, job_id: str):
    build_store = get_build_store(request)
    record = build_store.get(job_id)
    return render(request, "progress.html", {"job_id": job_id, "record": record})


@router.get("/build/{job_id}/events")
async def progress_events(request: Request, job_id: str):
    job_manager = request.app.state.job_manager
    return EventSourceResponse(job_manager.progress_stream(job_id))
