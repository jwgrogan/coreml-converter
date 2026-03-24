from fastapi import APIRouter, Request
from coreml_converter.web.dependencies import templates, get_build_store

router = APIRouter()


@router.get("/history")
async def history_page(request: Request):
    build_store = get_build_store(request)
    builds = build_store.list_all()
    from datetime import datetime, timezone
    _epoch = datetime.min.replace(tzinfo=timezone.utc)
    builds.sort(key=lambda b: b.started_at or b.completed_at or _epoch, reverse=True)
    return templates.TemplateResponse("history.html", {
        "request": request,
        "builds": builds,
    })
