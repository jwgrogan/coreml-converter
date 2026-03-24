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
        params: dict = {"query": query, "limit": limit, "sort": "Most Downloaded"}
        if model_type:
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
            versions = item.get("modelVersions", [])
            if not versions:
                continue
            version = versions[0]
            arch = self._parse_base_model(version.get("baseModel", ""))
            if arch is None:
                continue
            if base_arch and arch != base_arch:
                continue
            files = [f for f in version.get("files", []) if f.get("type") == "Model"]
            if not files:
                continue
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
