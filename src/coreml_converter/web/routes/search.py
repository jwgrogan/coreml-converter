from __future__ import annotations

import asyncio
import json
from concurrent.futures import ThreadPoolExecutor

from fastapi import APIRouter, Request, Query, Form
from starlette.responses import HTMLResponse

from coreml_converter.core.models import BaseArchitecture, ModelInfo, ModelSource, ModelType
from coreml_converter.web.dependencies import render, get_registry

router = APIRouter()
_executor = ThreadPoolExecutor(max_workers=2)

_SOURCE_MAP = {"hf": ModelSource.HUGGINGFACE, "civitai": ModelSource.CIVITAI}
_TYPE_MAP = {"checkpoint": ModelType.CHECKPOINT, "lora": ModelType.LORA}
_ARCH_MAP = {"sd1.5": BaseArchitecture.SD15, "sd2.0": BaseArchitecture.SD20}


def _get_favorites(request: Request):
    return request.app.state.favorites


@router.get("/")
async def home(request: Request):
    return render(request, "search.html", {"results": None, "favorites": None, "show_favorites": False})


@router.get("/search")
async def search(
    request: Request,
    q: str = Query(default=""),
    source: str = Query(default="all"),
    type: str = Query(default=""),
    arch: str = Query(default=""),
):
    registry = get_registry(request)
    favorites = _get_favorites(request)
    fav_keys = favorites.favorite_keys()

    if not q.strip():
        return render(request, "partials/search_results.html", {"results": [], "fav_keys": fav_keys})

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

    return render(request, "partials/search_results.html", {"results": results, "fav_keys": fav_keys})


@router.get("/favorites")
async def favorites_page(request: Request):
    favorites = _get_favorites(request)
    models = favorites.list_all()
    fav_keys = favorites.favorite_keys()
    return render(request, "search.html", {
        "results": models,
        "favorites": models,
        "show_favorites": True,
        "fav_keys": fav_keys,
    })


@router.post("/favorites/add")
async def add_favorite(request: Request, model_json: str = Form(...)):
    favorites = _get_favorites(request)
    model = ModelInfo(**json.loads(model_json))
    favorites.add(model)
    fav_keys = favorites.favorite_keys()
    return render(request, "partials/fav_button.html", {
        "model": model,
        "is_fav": True,
        "fav_keys": fav_keys,
    })


@router.post("/favorites/remove")
async def remove_favorite(request: Request, source: str = Form(...), model_id: str = Form(...)):
    favorites = _get_favorites(request)
    favorites.remove(source, model_id)
    return HTMLResponse('<span class="badge unfaved">Removed</span>')


@router.get("/import/civitai")
async def import_civitai_page(request: Request):
    """Show CivitAI collections available for import."""
    registry = get_registry(request)
    civitai_client = registry._clients.get(ModelSource.CIVITAI)

    collections = []
    error = None
    if civitai_client and hasattr(civitai_client, "get_collections"):
        try:
            loop = asyncio.get_event_loop()
            collections = await loop.run_in_executor(
                _executor, civitai_client.get_collections
            )
        except Exception as e:
            error = str(e)
    else:
        error = "CivitAI API key not configured. Run: coreml-converter config set civitai-key YOUR_KEY"

    return render(request, "import_civitai.html", {
        "collections": collections,
        "error": error,
    })


@router.post("/import/civitai/{collection_id}")
async def import_collection(request: Request, collection_id: int):
    """Import all models from a CivitAI collection into favorites."""
    registry = get_registry(request)
    favorites = _get_favorites(request)
    civitai_client = registry._clients.get(ModelSource.CIVITAI)

    if not civitai_client or not hasattr(civitai_client, "get_collection_items"):
        return HTMLResponse("<p class='error'>CivitAI not configured</p>", status_code=400)

    loop = asyncio.get_event_loop()
    models = await loop.run_in_executor(
        _executor,
        lambda: civitai_client.get_collection_items(collection_id),
    )

    imported = 0
    for model in models:
        favorites.add(model)
        imported += 1

    return HTMLResponse(
        f'<p style="color:var(--pico-ins-color);">Imported {imported} model(s) to favorites.</p>'
    )
