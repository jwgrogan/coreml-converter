from __future__ import annotations

import asyncio
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from fastapi import APIRouter, Request, Form, Query

from coreml_converter.core.analyzer import check_compatibility, detect_tag_conflicts, get_recommended_weight
from coreml_converter.core.models import (
    BaseArchitecture, BuildRecord, ConversionConfig, LoRAEntry,
    ModelInfo, ModelSource, ModelType, Recipe,
)
from coreml_converter.web.dependencies import templates, get_registry, get_build_store

router = APIRouter()
_executor = ThreadPoolExecutor(max_workers=2)


@router.get("/build")
async def builder_page(request: Request, base: str = Query(default="")):
    base_model = None
    if base:
        registry = get_registry(request)
        if ":" in base:
            source_str, model_id = base.split(":", 1)
            source_map = {"hf": ModelSource.HUGGINGFACE, "civitai": ModelSource.CIVITAI}
            source = source_map.get(source_str)
            loop = asyncio.get_event_loop()
            results = await loop.run_in_executor(
                _executor,
                lambda: registry.search(model_id, source=source, model_type=ModelType.CHECKPOINT, limit=1),
            )
            if results:
                base_model = results[0]

    return templates.TemplateResponse("builder.html", {
        "request": request,
        "base_model": base_model,
    })


@router.get("/build/search-loras")
async def search_loras(request: Request, q: str = Query(default=""), arch: str = Query(default="")):
    registry = get_registry(request)
    arch_map = {"SD1.5": BaseArchitecture.SD15, "SD2.0": BaseArchitecture.SD20}

    loop = asyncio.get_event_loop()
    results = await loop.run_in_executor(
        _executor,
        lambda: registry.search(
            query=q,
            model_type=ModelType.LORA,
            base_arch=arch_map.get(arch),
        ),
    )

    # Add weight recommendations
    lora_entries = []
    for model in results:
        weight, source = get_recommended_weight(model)
        lora_entries.append({
            "model": model,
            "recommended_weight": weight,
            "weight_source": source,
        })

    return templates.TemplateResponse("partials/search_results.html", {
        "request": request,
        "results": results,
        "lora_data": lora_entries,
    })


@router.post("/build/check-compatibility")
async def check_compat(request: Request):
    form = await request.form()
    # Parse recipe from form data
    base_json = form.get("base_model")
    loras_json = form.get("loras")

    if not base_json:
        return templates.TemplateResponse("partials/compatibility_report.html", {
            "request": request, "report": None,
        })

    base_model = ModelInfo(**json.loads(base_json))
    lora_entries = [LoRAEntry(**l) for l in json.loads(loras_json or "[]")]

    report = check_compatibility(base_model, lora_entries)
    tag_conflicts = detect_tag_conflicts(lora_entries)
    report.conflicts.extend(tag_conflicts)

    return templates.TemplateResponse("partials/compatibility_report.html", {
        "request": request, "report": report,
    })


@router.post("/build/start")
async def start_build(request: Request):
    form = await request.form()
    base_model = ModelInfo(**json.loads(form.get("base_model", "{}")))
    loras_raw = json.loads(form.get("loras", "[]"))
    lora_entries = [LoRAEntry(**l) for l in loras_raw]
    model_name = form.get("name", "custom-model")

    config = ConversionConfig(
        output_dir=Path(form.get("output_dir", "./output")),
        model_name=model_name,
        compute_units=form.get("compute_units", "all"),
        attention=form.get("attention", "split_einsum"),
    )

    recipe = Recipe(name=model_name, base_model=base_model, loras=lora_entries, conversion_config=config)
    record = BuildRecord(recipe=recipe)

    build_store = get_build_store(request)
    build_store.save(record)

    job_manager = request.app.state.job_manager
    await job_manager.submit(record)

    from starlette.responses import RedirectResponse
    return RedirectResponse(url=f"/build/{record.id}", status_code=303)
