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
import shutil
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
from coreml_converter.web.dependencies import (
    get_build_store, get_registry, get_train_manager, get_train_store,
)
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
    # Display name, when the on-disk filename is not what the user should see.
    name: str | None = None


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
                payload.path, payload.model_type, payload.arch, name=payload.name
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


@router.delete("/builds/{build_id}")
async def delete_build(request: Request, build_id: str):
    """Forget one build.

    Only the history record is removed — a model this build produced was moved
    into the user's models directory and is managed there, so deleting the
    record must not touch it.
    """
    store = get_build_store(request)
    record = store.get(build_id)
    if record is None:
        return _error(f"No build with id '{build_id}'", 404, "build_not_found")
    if record.status == BuildStatus.RUNNING:
        return _error(
            "That build is still running. Wait for it to finish first.",
            409,
            "build_running",
        )
    store.delete(build_id)
    return {"deleted": build_id}


@router.delete("/builds")
async def delete_finished_builds(request: Request):
    """Clear every build that is not still running."""
    removed = get_build_store(request).delete_finished()
    return {"deleted": removed}


# ----------------------------------------------------------------- training ---

def _summarize_training(record) -> dict:
    """Status payload for Fanny's poller.

    `percent`/`steps_done` come from the live progress store while a run is in
    flight, and from the persisted result once it has finished.
    """
    from coreml_converter.web.train_jobs import get_progress as train_progress

    progress = train_progress(record.id) or {}
    payload = {
        "train_id": record.id,
        "name": record.request.name,
        "mode": record.request.mode.value,
        "trigger": record.request.trigger,
        "status": record.status.value,
        "step": progress.get("step"),
        "message": progress.get("message"),
        "percent": progress.get("percent", 100 if record.status.value == "completed" else 0),
        "steps_done": progress.get("steps_done", 0),
        "steps_total": progress.get("steps_total", record.request.params.steps),
        "started_at": record.started_at,
        "completed_at": record.completed_at,
        "error": record.error,
    }
    if record.result:
        payload["result"] = {
            "lora_path": str(record.result.lora_path),
            "steps_completed": record.result.steps_completed,
            "training_time": record.result.training_time,
            "seconds_per_step": record.result.seconds_per_step,
            "file_size_mb": record.result.file_size_mb,
            "images_used": record.result.images_used,
            "loss_first": record.result.loss_first,
            "loss_last": record.result.loss_last,
        }
    return payload


@router.get("/train/bases")
async def train_bases() -> dict:
    """What to train against, per target family.

    Surfaced so the UI can pre-select a sensible base instead of making the
    user reason about weight-space distance between checkpoint families.
    """
    from coreml_converter.core.models import RECOMMENDED_BASES
    return {"families": RECOMMENDED_BASES}


@router.post("/dataset/inspect")
async def dataset_inspect(request: Request):
    """Screen a dataset without training on it.

    Lets the Train tab tell the user what looks questionable *before* they
    commit to an hour of compute. Flags are advisory — the detectors are not
    reliable enough to discard photos unattended.
    """
    import tempfile
    from pathlib import Path as _Path

    from coreml_converter.core.trainer.dataset import DatasetPrep

    body = await request.json()
    image_paths = body.get("image_paths") or []
    if not image_paths:
        return _error("image_paths is required.", 400, "images_required")

    resolution = int(body.get("resolution") or 512)
    scratch = _Path(tempfile.mkdtemp(prefix="fanny-inspect-"))
    try:
        prepared = DatasetPrep(resolution).prepare(image_paths, scratch / "out")
    except ValueError as exc:
        return _error(str(exc), 400, "dataset_unusable")
    finally:
        shutil.rmtree(scratch, ignore_errors=True)

    items = [{
        "name": item.source.name,
        "flagged": item.flagged,
        "reason": item.reason(),
        "low_detail": item.low_detail,
        "upscale_factor": round(item.upscale_factor, 2),
        "face_detected": item.face_detected,
        "greyscale": item.is_greyscale,
    } for item in prepared]
    flagged = [i for i in items if i["flagged"]]
    return {
        "total": len(items),
        "flagged": len(flagged),
        "low_detail": sum(1 for i in items if i["low_detail"]),
        "images": items,
    }


@router.post("/train/start")
async def train_start(request: Request):
    from coreml_converter.core.models import (
        StyleFamily, TrainingMode, TrainingParams, TrainRecord, TrainRequest,
        overtraining_warning,
    )
    from coreml_converter.core.trainer.dataset import DatasetPrep

    body = await request.json()
    manager = get_train_manager(request)

    # One GPU: a second concurrent run would just make both slower.
    if manager.active_count():
        return _error("A training run is already in progress.", 409, "train_busy")

    name = (body.get("name") or "").strip()
    trigger = (body.get("trigger") or "").strip()
    if not name:
        return _error("A name is required.", 400, "name_required")
    if not trigger:
        return _error("A trigger word is required.", 400, "trigger_required")

    raw_family = body.get("style_family") or StyleFamily.PHOTOREAL.value
    try:
        family = StyleFamily(raw_family)
    except ValueError:
        return _error(f"Unknown style_family '{raw_family}'.", 400, "bad_style_family")

    raw_mode = body.get("mode") or TrainingMode.CHARACTER.value
    try:
        mode = TrainingMode(raw_mode)
    except ValueError:
        return _error(f"Unknown mode '{raw_mode}'.", 400, "bad_mode")

    image_paths = body.get("image_paths") or []
    if not image_paths:
        return _error("image_paths is required.", 400, "images_required")
    found = DatasetPrep.collect(image_paths)
    if len(found) < 10:
        # Matches the design spec's 412 precondition for too few references.
        return _error(
            f"Need at least 10 usable images, found {len(found)}.",
            412, "not_enough_images")

    base_path = body.get("base_path")
    base = Path(base_path) if base_path else None
    # Either a single-file checkpoint or a diffusers directory.
    if base is None or not (base.is_file() or (base.is_dir() and (base / "model_index.json").is_file())):
        return _error(
            "base_path must be a checkpoint file or a diffusers model directory.",
            400, "base_not_found")

    output_dir = body.get("output_dir")
    if not output_dir:
        return _error("output_dir is required.", 400, "output_dir_required")

    overrides = body.get("params") or {}
    allowed = set(TrainingParams.model_fields)
    unknown = set(overrides) - allowed
    if unknown:
        return _error(f"Unknown params: {', '.join(sorted(unknown))}.",
                      400, "bad_params")
    params = TrainingParams.for_mode(mode.value, **overrides)
    # Advisory, not a rejection: a user who deliberately wants a long
    # text-encoder run should be able to have one.
    warning = overtraining_warning(params.steps, params.train_text_encoder)

    record = TrainRecord(request=TrainRequest(
        name=name, trigger=trigger, mode=mode, style_family=family,
        class_token=(body.get("class_token") or "woman").strip(),
        caption_suffix=body.get("caption_suffix") or "",
        image_paths=[str(p) for p in image_paths],
        output_dir=Path(output_dir), base_path=Path(base_path),
        params=params,
    ))
    await manager.submit(record)
    return {
        "train_id": record.id,
        "status": record.status.value,
        "images_found": len(found),
        "steps": params.steps,
        "mode": mode.value,
        "style_family": family.value,
        "warning": warning,
    }


@router.get("/train/{train_id}/status")
async def train_status(request: Request, train_id: str):
    record = get_train_store(request).get(train_id)
    if record is None:
        return _error(f"No training run with id '{train_id}'", 404, "train_not_found")
    return _summarize_training(record)


@router.post("/train/{train_id}/cancel")
async def train_cancel(request: Request, train_id: str):
    from coreml_converter.core.models import TrainStatus
    from coreml_converter.web.train_jobs import request_cancel

    store = get_train_store(request)
    record = store.get(train_id)
    if record is None:
        return _error(f"No training run with id '{train_id}'", 404, "train_not_found")
    if record.status not in (TrainStatus.PENDING, TrainStatus.RUNNING):
        return _error("That run has already finished.", 409, "train_not_running")
    request_cancel(train_id)
    return {"cancelling": train_id}


@router.get("/trains")
async def trains(request: Request):
    records = get_train_store(request).list_all()
    records.sort(key=lambda r: r.started_at or r.completed_at or _EPOCH, reverse=True)
    return [_summarize_training(r) for r in records]


@router.delete("/trains/{train_id}")
async def delete_train(request: Request, train_id: str):
    """Forget one run. The produced .safetensors lives in the user's LoRA
    directory and is managed there, so this must not touch it."""
    from coreml_converter.core.models import TrainStatus

    store = get_train_store(request)
    record = store.get(train_id)
    if record is None:
        return _error(f"No training run with id '{train_id}'", 404, "train_not_found")
    if record.status in (TrainStatus.PENDING, TrainStatus.RUNNING):
        return _error("That run is still going. Cancel it first.", 409, "train_running")
    store.delete(train_id)
    return {"deleted": train_id}
