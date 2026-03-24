from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor

from fastapi import APIRouter, Request, Query

from coreml_converter.core.models import BaseArchitecture, ModelSource, ModelType
from coreml_converter.web.dependencies import render, get_registry

router = APIRouter()
_executor = ThreadPoolExecutor(max_workers=2)

_SOURCE_MAP = {"hf": ModelSource.HUGGINGFACE, "civitai": ModelSource.CIVITAI}
_TYPE_MAP = {"checkpoint": ModelType.CHECKPOINT, "lora": ModelType.LORA}
_ARCH_MAP = {"sd1.5": BaseArchitecture.SD15, "sd2.0": BaseArchitecture.SD20}


@router.get("/")
async def home(request: Request):
    return render(request, "search.html", {"results": None})


@router.get("/search")
async def search(
    request: Request,
    q: str = Query(default=""),
    source: str = Query(default="all"),
    type: str = Query(default=""),
    arch: str = Query(default=""),
):
    registry = get_registry(request)
    if not q.strip():
        return render(request, "partials/search_results.html", {"results": []})

    loop = asyncio.get_event_loop()
    results = await loop.run_in_executor(
        _executor,
        lambda: registry.search(
            query=q,
            source=_SOURCE_MAP.get(source) if source != "all" else None,
            model_type=_TYPE_MAP.get(type) if type else None,
            base_arch=_ARCH_MAP.get(arch) if arch else None,
        ),
    )

    return render(request, "partials/search_results.html", {"results": results})
