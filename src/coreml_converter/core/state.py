from __future__ import annotations
import fcntl
import json
from pathlib import Path
from coreml_converter.core.models import BuildRecord


class BuildStore:
    def __init__(self, path: Path) -> None:
        self._path = path

    def _read_all(self) -> dict[str, dict]:
        if not self._path.exists():
            return {}
        data = json.loads(self._path.read_text())
        return {r["id"]: r for r in data.get("builds", [])}

    def _write_all(self, records: dict[str, dict]) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        data = {"schema_version": 1, "builds": list(records.values())}
        tmp = self._path.with_suffix(".tmp")
        with open(tmp, "w") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            json.dump(data, f, indent=2, default=str)
            fcntl.flock(f, fcntl.LOCK_UN)
        tmp.rename(self._path)

    def save(self, record: BuildRecord) -> None:
        records = self._read_all()
        records[record.id] = json.loads(record.model_dump_json())
        self._write_all(records)

    def get(self, build_id: str) -> BuildRecord | None:
        records = self._read_all()
        data = records.get(build_id)
        if data is None:
            return None
        return BuildRecord(**data)

    def list_all(self) -> list[BuildRecord]:
        records = self._read_all()
        return [BuildRecord(**r) for r in records.values()]
