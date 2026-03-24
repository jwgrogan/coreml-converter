from fastapi import APIRouter, Request
from coreml_converter.web.dependencies import templates

router = APIRouter()

@router.get("/build")
async def builder_page(request: Request):
    return templates.TemplateResponse("builder.html", {"request": request})
