"""Training job manager — mirrors web/jobs.py, kept separate because a
training run is long, cancellable, and has its own progress semantics.
"""
from __future__ import annotations

import asyncio
import contextlib
import logging
import shutil
import subprocess
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

from coreml_converter.core.models import TrainRecord, TrainStatus

logger = logging.getLogger(__name__)

@contextlib.contextmanager
def _keep_awake():
    """Hold an idle-sleep assertion for the duration of a run.

    Phase 0 measured runs stalling for ~17 minutes at a stretch because macOS
    schedules heavy background maintenance (Photos analysis, Spotlight
    indexing) whenever it decides the machine is unattended — which is exactly
    when someone leaves a 40-90 minute training job going. Worse, without an
    assertion the Mac is free to idle-sleep mid-run and silently lose the work.
    caffeinate does not suppress the background work, but it does stop the
    machine sleeping underneath us.
    """
    proc = None
    try:
        proc = subprocess.Popen(
            ["caffeinate", "-i", "-m"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except (FileNotFoundError, OSError):
        logger.warning("caffeinate unavailable; the machine may sleep mid-run")
    try:
        yield
    finally:
        if proc is not None:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except Exception:
                proc.kill()


_progress: dict[str, dict] = {}
_cancelled: set[str] = set()
_lock = threading.Lock()


def set_progress(job_id: str, step: str, message: str, percent: int,
                 steps_done: int = 0, steps_total: int = 0) -> None:
    with _lock:
        _progress[job_id] = {
            "step": step, "message": message, "percent": percent,
            "steps_done": steps_done, "steps_total": steps_total,
        }


def get_progress(job_id: str) -> dict:
    with _lock:
        return _progress.get(job_id, {})


def request_cancel(job_id: str) -> None:
    with _lock:
        _cancelled.add(job_id)


def is_cancelled(job_id: str) -> bool:
    with _lock:
        return job_id in _cancelled


def _clear(job_id: str) -> None:
    with _lock:
        _cancelled.discard(job_id)


def _run_training(job_id: str, record_dict: dict) -> dict:
    """Runs in a worker thread; updates the progress store as it goes."""
    from coreml_converter.core.trainer.dataset import DatasetPrep
    from coreml_converter.core.trainer.export import training_metadata
    from coreml_converter.core.trainer.trainer import LoRATrainer, TrainingCancelled
    from coreml_converter.core.models import caption_for

    record = TrainRecord(**record_dict)
    req = record.request
    params = req.params
    scratch = Path(tempfile.mkdtemp(prefix="fanny-train-"))

    try:
        with _keep_awake():
            set_progress(job_id, "prepare", "Preparing images", 1)
            prepared = DatasetPrep(params.resolution).prepare(
                req.image_paths, scratch / "images")
            warnings = [w for w in (p.warning() for p in prepared) if w]
            for w in warnings:
                logger.warning("dataset: %s", w)

            base = Path(req.base_path) if req.base_path else None
            if base is None or not (base.is_file() or base.is_dir()):
                raise ValueError("base checkpoint not found — base_path is required in v1")

            caption = caption_for(req.mode.value, req.trigger,
                                  req.class_token, req.caption_suffix)

            output_dir = Path(req.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            staged = scratch / f"{req.name}.safetensors"

            def metadata_fn(steps_done: int) -> dict:
                return training_metadata(
                    rank=params.rank, steps=steps_done,
                    learning_rate=params.learning_rate, resolution=params.resolution,
                    image_count=len(prepared), trigger=req.trigger,
                    class_token=req.class_token, caption=caption,
                    base_checkpoint=base.name,
                    style_family=req.style_family.value,
                    mode=req.mode.value)

            trainer = LoRATrainer(
                params,
                progress=lambda s, m, p_, d, t: set_progress(job_id, s, m, p_, d, t),
                should_cancel=lambda: is_cancelled(job_id),
                mode=req.mode.value)

            result = trainer.train(
                image_dir=scratch / "images", base_checkpoint=base,
                caption=caption, output_path=staged, metadata_fn=metadata_fn,
                # Intermediates go straight to the user's LoRA directory so a
                # cancelled or over-trained run still leaves something usable.
                checkpoint_dir=output_dir)

            # Atomic-ish handoff: only a finished run lands in the output dir.
            final = output_dir / f"{req.name}.safetensors"
            shutil.move(str(staged), str(final))
            result.lora_path = final

            return {"status": TrainStatus.COMPLETED.value,
                    "result": result.model_dump(mode="json"),
                    "warnings": warnings}

    except TrainingCancelled:
        return {"status": TrainStatus.CANCELLED.value, "error": "Cancelled"}
    except Exception as exc:                       # noqa: BLE001 - reported to the client
        logger.exception("training job %s failed", job_id)
        return {"status": TrainStatus.FAILED.value, "error": str(exc)}
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
        _clear(job_id)


class TrainJobManager:
    def __init__(self, store) -> None:
        self._store = store
        # One at a time: concurrent runs would contend for the same GPU.
        self._pool = ThreadPoolExecutor(max_workers=1)

    @property
    def store(self):
        return self._store

    def active_count(self) -> int:
        return sum(1 for r in self._store.list_all()
                   if r.status in (TrainStatus.PENDING, TrainStatus.RUNNING))

    async def submit(self, record: TrainRecord) -> str:
        record.status = TrainStatus.RUNNING
        record.started_at = datetime.now(timezone.utc)
        self._store.save(record)
        set_progress(record.id, "queued", "Queued", 0)

        loop = asyncio.get_running_loop()
        future = loop.run_in_executor(
            self._pool, _run_training, record.id,
            __import__("json").loads(record.model_dump_json()))

        def _finish(fut) -> None:
            from coreml_converter.core.models import TrainResult
            try:
                outcome = fut.result()
            except Exception as exc:               # noqa: BLE001
                outcome = {"status": TrainStatus.FAILED.value, "error": str(exc)}
            saved = self._store.get(record.id) or record
            saved.status = TrainStatus(outcome["status"])
            saved.completed_at = datetime.now(timezone.utc)
            saved.error = outcome.get("error")
            if outcome.get("result"):
                saved.result = TrainResult(**outcome["result"])
            self._store.save(saved)

        future.add_done_callback(_finish)
        return record.id
