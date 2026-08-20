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

    def delete(self, build_id: str) -> bool:
        """Remove one build from history. Returns whether it existed."""
        records = self._read_all()
        if build_id not in records:
            return False
        del records[build_id]
        self._write_all(records)
        return True

    def delete_finished(self) -> int:
        """Remove every build that is no longer running.

        A build still in progress is kept: dropping its record would orphan the
        running job and leave the UI polling an id that no longer exists.
        """
        from coreml_converter.core.models import BuildStatus

        records = self._read_all()
        keep = {
            build_id: data
            for build_id, data in records.items()
            if data.get("status") == BuildStatus.RUNNING.value
        }
        removed = len(records) - len(keep)
        if removed:
            self._write_all(keep)
        return removed

    def fail_interrupted(self, reason: str = "Interrupted — the converter stopped mid-build") -> int:
        """Mark builds still recorded as running as failed.

        Builds run in-process and do not survive a restart, so a record still
        marked RUNNING at startup belongs to a process that is gone. Without
        this it stays "running" in the user's build history forever.

        Returns how many records were reconciled.
        """
        from datetime import datetime, timezone
        from coreml_converter.core.models import BuildStatus

        records = self._read_all()
        changed = 0
        for data in records.values():
            if data.get("status") != BuildStatus.RUNNING.value:
                continue
            data["status"] = BuildStatus.FAILED.value
            data["error"] = reason
            data["completed_at"] = datetime.now(timezone.utc).isoformat()
            changed += 1

        if changed:
            self._write_all(records)
        return changed


class TrainStore:
    """Training-run history, mirroring BuildStore's on-disk conventions."""

    def __init__(self, path: Path) -> None:
        self._path = path

    def _read_all(self) -> dict[str, dict]:
        if not self._path.exists():
            return {}
        data = json.loads(self._path.read_text())
        return {r["id"]: r for r in data.get("trainings", [])}

    def _write_all(self, records: dict[str, dict]) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        data = {"schema_version": 1, "trainings": list(records.values())}
        tmp = self._path.with_suffix(".tmp")
        with open(tmp, "w") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            json.dump(data, f, indent=2, default=str)
            fcntl.flock(f, fcntl.LOCK_UN)
        tmp.rename(self._path)

    def save(self, record: "TrainRecord") -> None:
        records = self._read_all()
        records[record.id] = json.loads(record.model_dump_json())
        self._write_all(records)

    def get(self, train_id: str) -> "TrainRecord | None":
        from coreml_converter.core.models import TrainRecord
        data = self._read_all().get(train_id)
        return TrainRecord(**data) if data else None

    def list_all(self) -> list["TrainRecord"]:
        from coreml_converter.core.models import TrainRecord
        return [TrainRecord(**r) for r in self._read_all().values()]

    def delete(self, train_id: str) -> bool:
        records = self._read_all()
        if train_id not in records:
            return False
        del records[train_id]
        self._write_all(records)
        return True

    def fail_interrupted(self, reason: str = "Interrupted — the converter stopped mid-run") -> int:
        """Mark runs that were mid-flight when the process died.

        Without this a killed converter leaves rows stuck in "running"
        forever, and the UI shows a training job that is not happening.
        """
        from coreml_converter.core.models import TrainStatus
        records = self._read_all()
        changed = 0
        for rec in records.values():
            if rec.get("status") in (TrainStatus.PENDING.value, TrainStatus.RUNNING.value):
                rec["status"] = TrainStatus.FAILED.value
                rec["error"] = reason
                changed += 1
        if changed:
            self._write_all(records)
        return changed
