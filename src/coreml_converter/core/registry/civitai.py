from __future__ import annotations
import logging
import time
from pathlib import Path
import httpx
from coreml_converter.core.models import (
    BaseArchitecture, ModelInfo, ModelSource, ModelType,
)
from coreml_converter.core.registry.base import RegistryClient
from coreml_converter.core.registry.rate_limiter import TokenBucketRateLimiter

logger = logging.getLogger(__name__)

CIVITAI_API = "https://civitai.com/api/v1"

_BASE_MODEL_MAP = {
    "SD 1.4": BaseArchitecture.SD15,
    "SD 1.5": BaseArchitecture.SD15,
    "SD 1.5 LCM": BaseArchitecture.SD15,
    "SD 1.5 Hyper": BaseArchitecture.SD15,
    "SD 2.0": BaseArchitecture.SD20,
    "SD 2.1": BaseArchitecture.SD20,
}

_TYPE_MAP = {
    "Checkpoint": ModelType.CHECKPOINT,
    "LORA": ModelType.LORA,
}


class CivitAIClient(RegistryClient):
    def __init__(self, api_key: str | None = None) -> None:
        self._api_key = api_key
        self._client = httpx.Client(timeout=30.0)
        self._rate_limiter = TokenBucketRateLimiter(rate=2.0, capacity=3)

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        return headers

    def _parse_base_model(self, base_model_str: str) -> BaseArchitecture | None:
        return _BASE_MODEL_MAP.get(base_model_str)

    def _parse_civitai_type(self, type_str: str) -> ModelType | None:
        return _TYPE_MAP.get(type_str)

    def search(self, query: str, model_type: ModelType | None = None,
               base_arch: BaseArchitecture | None = None, limit: int = 20) -> list[ModelInfo]:
        self._rate_limiter.acquire()
        # CivitAI API has a bug: combining query + types returns 0 results.
        # When query is provided, we omit the types param and filter client-side.
        params: dict = {"limit": limit * 3, "sort": "Most Downloaded"}
        if query:
            params["query"] = query
        elif model_type:
            # Only use server-side type filter when there's no query
            civitai_type = {ModelType.CHECKPOINT: "Checkpoint", ModelType.LORA: "LORA"}
            params["types"] = civitai_type.get(model_type, "Checkpoint")
        resp = self._client.get(f"{CIVITAI_API}/models", params=params, headers=self._headers())
        resp.raise_for_status()
        data = resp.json()
        results: list[ModelInfo] = []
        for item in data.get("items", []):
            mt = self._parse_civitai_type(item.get("type", ""))
            if mt is None:
                continue
            if model_type and mt != model_type:
                continue
            versions = item.get("modelVersions", [])
            if not versions:
                continue
            # Find the first version with a compatible base model
            version = None
            arch = None
            files = []
            for v in versions:
                a = self._parse_base_model(v.get("baseModel", ""))
                if a is None:
                    continue
                if base_arch and a != base_arch:
                    continue
                f = [f for f in v.get("files", []) if f.get("type") == "Model"]
                if f:
                    version = v
                    arch = a
                    files = f
                    break
            if version is None or arch is None:
                continue
            if len(results) >= limit:
                break
            results.append(ModelInfo(
                source=ModelSource.CIVITAI, id=str(item["id"]), name=item["name"],
                base_architecture=arch, model_type=mt, tags=item.get("tags", []),
                download_url=files[0]["downloadUrl"],
                metadata={
                    "version_id": version["id"],
                    "version_name": version.get("name", ""),
                    "download_count": item.get("stats", {}).get("downloadCount", 0),
                    "images": [img.get("url") for img in version.get("images", [])[:3]],
                    "description": item.get("description", ""),
                },
            ))
        return results

    def get_by_id(self, model_id: str) -> ModelInfo | None:
        """Fetch a single model by its CivitAI ID."""
        self._rate_limiter.acquire()
        resp = self._client.get(f"{CIVITAI_API}/models/{model_id}", headers=self._headers())
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
        item = resp.json()
        mt = self._parse_civitai_type(item.get("type", ""))
        if mt is None:
            return None
        versions = item.get("modelVersions", [])
        version = None
        arch = None
        files = []
        for v in versions:
            a = self._parse_base_model(v.get("baseModel", ""))
            if a is None:
                continue
            f = [fi for fi in v.get("files", []) if fi.get("type") == "Model"]
            if f:
                version = v
                arch = a
                files = f
                break
        if version is None or arch is None:
            return None
        return ModelInfo(
            source=ModelSource.CIVITAI, id=str(item["id"]), name=item["name"],
            base_architecture=arch, model_type=mt, tags=item.get("tags", []),
            download_url=files[0]["downloadUrl"],
            metadata={
                "version_id": version["id"],
                "version_name": version.get("name", ""),
                "download_count": item.get("stats", {}).get("downloadCount", 0),
                "images": [img.get("url") for img in version.get("images", [])[:3]],
                "description": item.get("description", ""),
            },
        )

    def get_compatible_loras(self, base_model: ModelInfo, limit: int = 20) -> list[ModelInfo]:
        return self.search(query="", model_type=ModelType.LORA,
                          base_arch=base_model.base_architecture, limit=limit)

    def download(self, model: ModelInfo, dest: Path, retries: int = 3) -> Path:
        dest.mkdir(parents=True, exist_ok=True)
        filename = f"{model.source.value}_{model.id}.safetensors"
        file_path = dest / filename
        partial_path = file_path.with_suffix(".partial")
        for attempt in range(retries):
            try:
                self._rate_limiter.acquire()
                headers = self._headers()
                if partial_path.exists():
                    existing_size = partial_path.stat().st_size
                    headers["Range"] = f"bytes={existing_size}-"
                    mode = "ab"
                else:
                    mode = "wb"
                with self._client.stream("GET", model.download_url, headers=headers) as resp:
                    resp.raise_for_status()
                    with open(partial_path, mode) as f:
                        for chunk in resp.iter_bytes(chunk_size=8192):
                            f.write(chunk)
                partial_path.rename(file_path)
                return file_path
            except (httpx.HTTPError, OSError) as e:
                logger.warning(f"Download attempt {attempt + 1} failed: {e}")
                if attempt == retries - 1:
                    if partial_path.exists():
                        partial_path.unlink()
                    raise
                time.sleep(2 ** attempt)
        raise RuntimeError("Download failed after all retries")
