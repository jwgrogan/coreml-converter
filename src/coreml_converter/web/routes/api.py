"""JSON API for programmatic clients.

The routes under `/build` and `/history` are an HTMX UI: they take form
encoding, answer with HTML fragments, and redirect. Fanny (the Swift app)
drives the converter as a child process and needs machine-readable
equivalents, so this router mirrors that surface under `/api` without
replacing it. Both share the same JobManager, BuildStore and upload registry,
so a build started from either surface is visible to the other.
"""
from __future__ import annotations

import asyncio
import json
import logging
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, ValidationError

import coreml_converter
from coreml_converter.core.ml_check import check_ml_deps
from coreml_converter.core.models import (
    BuildRecord,
    BuildStatus,
    ConversionConfig,
    LoRAEntry,
    ModelInfo,
    ModelSource,
    ModelType,
    Recipe,
)
from coreml_converter.web import uploads
from coreml_converter.web.dependencies import get_build_store, get_registry
from coreml_converter.web.jobs import get_progress

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api")

# Registry lookups are blocking HTTP calls; keep them off the event loop.
_executor = ThreadPoolExecutor(max_workers=2)

_EPOCH = datetime.min.replace(tzinfo=timezone.utc)

_SOURCE_MAP = {
    "hf": ModelSource.HUGGINGFACE,
    "huggingface": ModelSource.HUGGINGFACE,
    "civitai": ModelSource.CIVITAI,
}


def _error(message: str, status_code: int, code: str) -> JSONResponse:
    return JSONResponse({"error": {"code": code, "message": message}}, status_code)


# --- Health ---------------------------------------------------------------


@router.get("/health")
async def health() -> dict:
    """Liveness plus whether this environment can actually build.

    Fanny polls this to decide when a converter it spawned is ready, and to
    tell "running but can't convert" apart from "not running".
    """
    ok, missing = check_ml_deps()
    return {
        "status": "ok",
        "version": coreml_converter.__version__,
        "ml_deps_ok": ok,
        "missing_deps": missing,
    }


# --- Local model registration --------------------------------------------


class RegisterPathRequest(BaseModel):
    path: str
    model_type: str = "checkpoint"
    arch: str = "SD1.5"


@router.post("/upload")
async def upload(request: Request):
    """Register a local checkpoint, by path or by multipart upload.

    JSON `{"path": "/abs/model.safetensors"}` registers the file where it
    already sits — the path Fanny uses, since checkpoints are 2-7GB and both
    processes share a filesystem. A multipart body with a `file` field copies
    the upload into the converter's cache instead.
    """
    content_type = request.headers.get("content-type", "")

    if content_type.startswith("application/json"):
        try:
            payload = RegisterPathRequest(**(await request.json()))
        except (ValidationError, json.JSONDecodeError, TypeError) as e:
            return _error(f"Invalid request body: {e}", 400, "invalid_body")

        try:
            model = uploads.register_path(
                payload.path, payload.model_type, payload.arch
            )
        except ValueError as e:
            return _error(str(e), 400, "unsupported_format")
        except FileNotFoundError as e:
            return _error(str(e), 400, "not_found")
    else:
        form = await request.form()
        file = form.get("file")
        if file is None or not getattr(file, "filename", ""):
            return _error("No file provided", 400, "no_file")

        from coreml_converter.core.config import get_app_dir

        try:
            model = uploads.store_upload(
                file.file,
                file.filename,
                cache_dir=get_app_dir() / "cache" / "uploads",
                model_type=str(form.get("model_type") or "checkpoint"),
                arch=str(form.get("arch") or "SD1.5"),
            )
        except ValueError as e:
            return _error(str(e), 400, "unsupported_format")

    local_path = Path(model.metadata["local_path"])
    return {
        "model_ref": model.id,
        "name": model.name,
        "size_bytes": local_path.stat().st_size if local_path.is_file() else 0,
    }


# --- Builds ---------------------------------------------------------------


class ModelRef(BaseModel):
    """Either a local `ref` from /api/upload, or a `source` + `id` to fetch."""

    ref: str | None = None
    source: str | None = None
    id: str | None = None


class LoRARef(ModelRef):
    weight: float = 0.7


class BuildStartRequest(BaseModel):
    name: str
    base: ModelRef
    output_dir: str
    loras: list[LoRARef] = Field(default_factory=list)
    attention: str = "split_einsum"
    compute_units: str = "all"
    # Fanny forwards its own configured key so NSFW/gated CivitAI checkpoints
    # can be downloaded without configuring the converter separately.
    civitai_api_key: str | None = None


async def _resolve_model(registry, spec: ModelRef, model_type: ModelType) -> ModelInfo:
    """Turn a ref or source+id into a ModelInfo, or raise LookupError/ValueError."""
    if spec.ref:
        model = uploads.get(spec.ref)
        if model is None:
            raise LookupError(
                f"Unknown model ref '{spec.ref}'. Register it with /api/upload first."
            )
        return model

    if spec.source and spec.id:
        source = _SOURCE_MAP.get(spec.source.lower())
        if source is None:
            raise ValueError(f"Unknown source '{spec.source}'. Use 'civitai' or 'hf'.")
        loop = asyncio.get_event_loop()
        model = await loop.run_in_executor(
            _executor, lambda: registry.get_by_id(source, spec.id)
        )
        if model is None:
            raise LookupError(f"No {spec.source} model found with id '{spec.id}'")
        return model

    raise ValueError("Each model needs either 'ref' or both 'source' and 'id'")


@router.post("/build/start")
async def build_start(request: Request):
    """Submit a build and return immediately with its id.

    Conversion takes 10-20+ minutes, so this never blocks on the result —
    poll /api/build/{id}/status.
    """
    try:
        payload = BuildStartRequest(**(await request.json()))
    except (ValidationError, json.JSONDecodeError, TypeError) as e:
        return _error(f"Invalid request body: {e}", 400, "invalid_body")

    registry = get_registry(request)
    try:
        base_model = await _resolve_model(registry, payload.base, ModelType.CHECKPOINT)
        lora_entries = [
            LoRAEntry(
                model=await _resolve_model(registry, spec, ModelType.LORA),
                weight=spec.weight,
            )
            for spec in payload.loras
        ]
    except LookupError as e:
        return _error(str(e), 404, "model_not_found")
    except (ValueError, ValidationError) as e:
        return _error(str(e), 400, "invalid_model_ref")

    config = ConversionConfig(
        output_dir=Path(payload.output_dir).expanduser(),
        model_name=payload.name,
        compute_units=payload.compute_units,
        attention=payload.attention,
    )
    recipe = Recipe(
        name=payload.name,
        base_model=base_model,
        loras=lora_entries,
        conversion_config=config,
    )
    record = BuildRecord(recipe=recipe)

    get_build_store(request).save(record)

    api_key = payload.civitai_api_key
    if not api_key:
        from coreml_converter.core.config import get_app_dir, load_config

        api_key = load_config(get_app_dir() / "config.json").civitai_api_key

    await request.app.state.job_manager.submit(record, civitai_api_key=api_key)

    return {"build_id": record.id, "status": record.status.value}


def _summarize(record: BuildRecord) -> dict:
    """Shared shape for status and history entries."""
    progress = get_progress(record.id)
    terminal = record.status in (BuildStatus.COMPLETED, BuildStatus.FAILED)

    if progress:
        step = progress.get("step", record.status.value)
        message = progress.get("message", "")
        percent = progress.get("percent", 0)
    else:
        # Progress is dropped once the SSE stream closes; fall back to the
        # durable record so a late poll still gets a sane terminal answer.
        step = record.status.value
        message = record.error or ""
        percent = 100 if record.status == BuildStatus.COMPLETED else 0

    return {
        "build_id": record.id,
        "name": record.recipe.name,
        "status": record.status.value,
        "step": step,
        "message": message,
        "percent": percent,
        "error": record.error,
        "started_at": record.started_at.isoformat() if record.started_at else None,
        "completed_at": (
            record.completed_at.isoformat() if record.completed_at else None
        ),
        "output_path": (
            str(record.result.mlmodelc_path)
            if (terminal and record.result)
            else None
        ),
        "lora_count": len(record.recipe.loras),
    }


@router.get("/build/{build_id}/status")
async def build_status(request: Request, build_id: str):
    record = get_build_store(request).get(build_id)
    if record is None:
        return _error(f"No build with id '{build_id}'", 404, "build_not_found")
    return _summarize(record)


@router.get("/builds")
async def builds(request: Request):
    records = get_build_store(request).list_all()
    records.sort(
        key=lambda r: r.started_at or r.completed_at or _EPOCH, reverse=True
    )
    return [_summarize(r) for r in records]
