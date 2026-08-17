from __future__ import annotations

import asyncio
import json
import logging
import shutil
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import AsyncGenerator

from coreml_converter.core.models import BuildRecord, BuildStatus

logger = logging.getLogger(__name__)

# Thread-safe progress store: job_id -> {step, message, percent, steps_done, steps_total}
_progress: dict[str, dict] = {}
_progress_lock = threading.Lock()


def _set_progress(job_id: str, step: str, message: str, percent: int, steps_done: int = 0, steps_total: int = 0):
    with _progress_lock:
        _progress[job_id] = {
            "step": step,
            "message": message,
            "percent": percent,
            "steps_done": steps_done,
            "steps_total": steps_total,
        }


def _get_progress(job_id: str) -> dict:
    with _progress_lock:
        return _progress.get(job_id, {})


def get_progress(job_id: str) -> dict:
    """Latest progress snapshot for a job, or {} once it has been consumed.

    Public counterpart to the SSE stream, for pollers like the JSON API.
    """
    return _get_progress(job_id)


def _run_build(job_id: str, record_dict: dict, cache_dir: str, output_dir: str, civitai_api_key: str | None = None) -> dict:
    """Runs in a thread. Updates _progress dict for SSE consumption."""
    from coreml_converter.core.models import BuildRecord, BuildStatus
    from coreml_converter.core.merger.merger import Merger
    from coreml_converter.core.converter.converter import Converter
    from coreml_converter.core.registry import Registry
    from coreml_converter.core.registry.huggingface import HuggingFaceClient
    from coreml_converter.core.registry.civitai import CivitAIClient

    record = BuildRecord(**record_dict)
    recipe = record.recipe

    # Count total steps: download base + download loras + merge + convert
    total_loras = len(recipe.loras)
    needs_download_base = not recipe.base_model.metadata.get("local_path")
    needs_download_loras = [e for e in recipe.loras if not e.model.metadata.get("local_path")]
    total_steps = (1 if needs_download_base else 0) + len(needs_download_loras) + 1 + 1  # merge + convert
    current_step = 0

    try:
        registry = Registry(
            hf_client=HuggingFaceClient(),
            civitai_client=CivitAIClient(api_key=civitai_api_key),
        )
        cache = Path(cache_dir)

        # Download base
        if needs_download_base:
            current_step += 1
            _set_progress(job_id, "downloading", f"Downloading base model: {recipe.base_model.name}...",
                          int(current_step / total_steps * 100), current_step, total_steps)
            base_path = registry.download(recipe.base_model, cache)
            recipe.base_model.metadata["local_path"] = str(base_path)
        else:
            _set_progress(job_id, "downloading", "Using local base model", 5, 0, total_steps)

        # Download LoRAs
        for i, entry in enumerate(recipe.loras):
            if not entry.model.metadata.get("local_path"):
                current_step += 1
                _set_progress(job_id, "downloading",
                              f"Downloading LoRA {i+1}/{total_loras}: {entry.model.name}...",
                              int(current_step / total_steps * 100), current_step, total_steps)
                lora_path = registry.download(entry.model, cache)
                entry.model.metadata["local_path"] = str(lora_path)

        # Merge + convert. The merged diffusers pipeline is a multi-GB
        # intermediate, so it goes in a scratch dir that is always cleaned up
        # rather than into output_dir — which is Fanny's models folder, where
        # a stray `merged_pipeline/` would sit next to the user's models. The
        # scratch path must also be space-free: Apple's converter embeds this
        # path in the filenames it later compiles with an unquoted shell
        # command (see scratch_root).
        from coreml_converter.core.converter.converter import (
            make_build_scratch, sweep_stale_scratch_dirs,
        )

        out_root = Path(output_dir)
        out_root.mkdir(parents=True, exist_ok=True)
        sweep_stale_scratch_dirs(out_root)
        scratch = make_build_scratch(out_root)
        try:
            current_step += 1
            _set_progress(job_id, "merging",
                          f"Merging {total_loras} LoRA(s) into base model..." if total_loras else "Preparing model...",
                          int(current_step / total_steps * 100), current_step, total_steps)
            merger = Merger()
            merged_path = merger.merge(recipe, cache, scratch)

            # Convert
            current_step += 1
            _set_progress(job_id, "converting", "Converting to CoreML (this may take several minutes)...",
                          int(current_step / total_steps * 90), current_step, total_steps)
            converter = Converter()
            result = converter.convert(merged_path, recipe)
        finally:
            shutil.rmtree(scratch, ignore_errors=True)

        _set_progress(job_id, "completed", "Build complete!", 100, total_steps, total_steps)

        record.status = BuildStatus.COMPLETED
        record.result = result
        record.completed_at = datetime.now(timezone.utc)
    except Exception as e:
        _set_progress(job_id, "failed", f"Error: {e}", 0, 0, 0)
        record.status = BuildStatus.FAILED
        record.error = str(e)
        record.completed_at = datetime.now(timezone.utc)

    return json.loads(record.model_dump_json())


class JobManager:
    def __init__(self, cache_dir: Path, build_store) -> None:
        self._cache_dir = cache_dir
        self._store = build_store
        self._executor = ThreadPoolExecutor(max_workers=1)

    async def submit(self, record: BuildRecord, civitai_api_key: str | None = None) -> str:
        record.status = BuildStatus.RUNNING
        record.started_at = datetime.now(timezone.utc)
        self._store.save(record)

        _set_progress(record.id, "starting", "Starting build...", 0, 0, 0)

        record_dict = json.loads(record.model_dump_json())
        output_dir = str(record.recipe.conversion_config.output_dir)

        loop = asyncio.get_event_loop()
        future = loop.run_in_executor(
            self._executor,
            _run_build,
            record.id,
            record_dict,
            str(self._cache_dir),
            output_dir,
            civitai_api_key,
        )

        async def _on_complete(fut):
            try:
                result_dict = await fut
                updated = BuildRecord(**result_dict)
                self._store.save(updated)
            except Exception as e:
                record.status = BuildStatus.FAILED
                record.error = str(e)
                self._store.save(record)

        asyncio.ensure_future(_on_complete(future))
        return record.id

    async def progress_stream(self, job_id: str) -> AsyncGenerator[str, None]:
        """SSE stream with step-by-step progress."""
        while True:
            progress = _get_progress(job_id)
            record = self._store.get(job_id)

            if not record:
                yield f"data: {json.dumps({'step': 'error', 'message': 'Job not found', 'percent': 0})}\n\n"
                return

            if progress:
                yield f"data: {json.dumps(progress)}\n\n"
            else:
                yield f"data: {json.dumps({'step': record.status.value, 'message': 'Waiting...', 'percent': 0})}\n\n"

            if record.status in (BuildStatus.COMPLETED, BuildStatus.FAILED):
                # Send final state
                if record.status == BuildStatus.COMPLETED and record.result:
                    final = {
                        "step": "completed",
                        "message": "Build complete!",
                        "percent": 100,
                        "result": {
                            "mlpackage_path": str(record.result.mlpackage_path),
                            "mlmodelc_path": str(record.result.mlmodelc_path),
                            "manifest_path": str(record.result.manifest_path),
                            "model_size_mb": record.result.model_size_mb,
                            "conversion_time": record.result.conversion_time,
                        },
                    }
                elif record.status == BuildStatus.FAILED:
                    final = {"step": "failed", "message": record.error or "Unknown error", "percent": 0}
                else:
                    final = {"step": record.status.value, "message": "", "percent": 0}
                yield f"data: {json.dumps(final)}\n\n"
                # Clean up progress
                with _progress_lock:
                    _progress.pop(job_id, None)
                return

            await asyncio.sleep(1)
