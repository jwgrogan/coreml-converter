from __future__ import annotations

import asyncio
import logging
import queue
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import AsyncGenerator

from coreml_converter.core.models import BuildRecord, BuildStatus

logger = logging.getLogger(__name__)

# Progress events per job
_progress_queues: dict[str, queue.Queue] = {}


def get_progress_queue(job_id: str) -> queue.Queue:
    if job_id not in _progress_queues:
        _progress_queues[job_id] = queue.Queue()
    return _progress_queues[job_id]


def _run_build(record_dict: dict, cache_dir: str, output_dir: str) -> dict:
    """Runs in a separate process."""
    from coreml_converter.core.models import BuildRecord, BuildStatus
    from coreml_converter.core.merger.merger import Merger
    from coreml_converter.core.converter.converter import Converter

    record = BuildRecord(**record_dict)
    recipe = record.recipe

    try:
        merger = Merger()
        merged_path = merger.merge(recipe, Path(cache_dir), Path(output_dir))

        converter = Converter()
        result = converter.convert(merged_path, recipe)

        record.status = BuildStatus.COMPLETED
        record.result = result
        record.completed_at = datetime.now(timezone.utc)
    except Exception as e:
        record.status = BuildStatus.FAILED
        record.error = str(e)
        record.completed_at = datetime.now(timezone.utc)

    import json
    return json.loads(record.model_dump_json())


class JobManager:
    def __init__(self, cache_dir: Path, build_store) -> None:
        self._cache_dir = cache_dir
        self._store = build_store
        self._executor = ProcessPoolExecutor(max_workers=1)

    async def submit(self, record: BuildRecord) -> str:
        record.status = BuildStatus.RUNNING
        record.started_at = datetime.now(timezone.utc)
        self._store.save(record)

        import json
        record_dict = json.loads(record.model_dump_json())
        output_dir = str(record.recipe.conversion_config.output_dir)

        loop = asyncio.get_event_loop()
        future = loop.run_in_executor(
            self._executor,
            _run_build,
            record_dict,
            str(self._cache_dir),
            output_dir,
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
        """SSE stream for job progress."""
        while True:
            record = self._store.get(job_id)
            if record is None:
                yield f"data: {{\"error\": \"Job not found\"}}\n\n"
                return

            status = record.status.value
            yield f"data: {{\"status\": \"{status}\"}}\n\n"

            if record.status in (BuildStatus.COMPLETED, BuildStatus.FAILED):
                return

            await asyncio.sleep(2)
