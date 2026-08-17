from __future__ import annotations

import asyncio
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from fastapi import APIRouter, Request, Form, Query, UploadFile, File
from starlette.responses import HTMLResponse, RedirectResponse

from coreml_converter.core.analyzer import check_compatibility, detect_tag_conflicts, get_recommended_weight
from coreml_converter.core.models import (
    BaseArchitecture, BuildRecord, ConversionConfig, LoRAEntry,
    ModelInfo, ModelSource, ModelType, Recipe,
)
from coreml_converter.web import uploads
from coreml_converter.web.dependencies import render, get_registry, get_build_store

router = APIRouter()
_executor = ThreadPoolExecutor(max_workers=2)


@router.get("/build")
async def builder_page(request: Request, base: str = Query(default="")):
    base_model = None
    if base:
        if ":" in base:
            source_str, model_id = base.split(":", 1)
            local = uploads.get(model_id)
            if local is not None:
                base_model = local
            else:
                registry = get_registry(request)
                source_map = {"hf": ModelSource.HUGGINGFACE, "civitai": ModelSource.CIVITAI}
                source = source_map.get(source_str)
                if source:
                    loop = asyncio.get_event_loop()
                    results = await loop.run_in_executor(
                        _executor,
                        lambda: registry.search(model_id, source=source, model_type=ModelType.CHECKPOINT, limit=1),
                    )
                    if results:
                        base_model = results[0]

    return render(request, "builder.html", {"base_model": base_model})


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

    lora_entries = []
    for model in results:
        weight, source = get_recommended_weight(model)
        lora_entries.append({
            "model": model,
            "recommended_weight": weight,
            "weight_source": source,
        })

    return render(request, "partials/search_results.html", {
        "results": results,
        "lora_data": lora_entries,
    })


@router.post("/build/check-compatibility")
async def check_compat(request: Request):
    form = await request.form()
    base_json = form.get("base_model")
    loras_json = form.get("loras")

    if not base_json:
        return render(request, "partials/compatibility_report.html", {"report": None})

    base_model = ModelInfo(**json.loads(base_json))
    lora_entries = [LoRAEntry(**l) for l in json.loads(loras_json or "[]")]

    report = check_compatibility(base_model, lora_entries)
    tag_conflicts = detect_tag_conflicts(lora_entries)
    report.conflicts.extend(tag_conflicts)

    return render(request, "partials/compatibility_report.html", {"report": report})


@router.post("/build/upload")
async def upload_model(
    request: Request,
    file: UploadFile = File(...),
    model_type: str = Form("checkpoint"),
    arch: str = Form("SD1.5"),
):
    """Upload a local .safetensors/.ckpt file as a base model or LoRA."""
    from coreml_converter.core.config import get_app_dir

    if not file.filename:
        return HTMLResponse("<p class='error'>No file selected</p>", status_code=400)

    try:
        model = uploads.store_upload(
            file.file,
            file.filename,
            cache_dir=get_app_dir() / "cache" / "uploads",
            model_type=model_type,
            arch=arch,
        )
    except ValueError as e:
        return HTMLResponse(f"<p class='error'>{e}</p>", status_code=400)

    return render(request, "partials/model_card.html", {
        "model": model,
        "uploaded": True,
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
    # Pass API key so the build thread can download models
    from coreml_converter.core.config import get_app_dir, load_config
    config = load_config(get_app_dir() / "config.json")
    await job_manager.submit(record, civitai_api_key=config.civitai_api_key)

    return RedirectResponse(url=f"/build/{record.id}", status_code=303)
