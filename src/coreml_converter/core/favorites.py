from __future__ import annotations

import json
from pathlib import Path

from coreml_converter.core.models import ModelInfo


class FavoritesStore:
    def __init__(self, path: Path) -> None:
        self._path = path

    def _read(self) -> dict[str, dict]:
        if not self._path.exists():
            return {}
        data = json.loads(self._path.read_text())
        return {f["id"]: f for f in data.get("favorites", [])}

    def _write(self, favs: dict[str, dict]) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        data = {"schema_version": 1, "favorites": list(favs.values())}
        self._path.write_text(json.dumps(data, indent=2))

    def add(self, model: ModelInfo) -> None:
        favs = self._read()
        key = f"{model.source.value}:{model.id}"
        favs[key] = json.loads(model.model_dump_json())
        self._write(favs)

    def remove(self, source: str, model_id: str) -> None:
        favs = self._read()
        key = f"{source}:{model_id}"
        favs.pop(key, None)
        self._write(favs)

    def is_favorite(self, source: str, model_id: str) -> bool:
        favs = self._read()
        return f"{source}:{model_id}" in favs

    def list_all(self) -> list[ModelInfo]:
        favs = self._read()
        return [ModelInfo(**f) for f in favs.values()]

    def favorite_keys(self) -> set[str]:
        return set(self._read().keys())
