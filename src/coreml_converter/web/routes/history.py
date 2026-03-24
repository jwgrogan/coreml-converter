from datetime import datetime, timezone

from fastapi import APIRouter, Request
from coreml_converter.web.dependencies import render, get_build_store

router = APIRouter()

_EPOCH = datetime.min.replace(tzinfo=timezone.utc)


@router.get("/history")
async def history_page(request: Request):
    build_store = get_build_store(request)
    builds = build_store.list_all()
    builds.sort(key=lambda b: b.started_at or b.completed_at or _EPOCH, reverse=True)
    return render(request, "history.html", {"builds": builds})
