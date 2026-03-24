from fastapi import APIRouter, Request
from coreml_converter.web.dependencies import templates

router = APIRouter()

@router.get("/build/{job_id}")
async def progress_page(request: Request, job_id: str):
    return templates.TemplateResponse("progress.html", {"request": request, "job_id": job_id})
