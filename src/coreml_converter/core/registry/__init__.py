from __future__ import annotations
import logging
from pathlib import Path
from coreml_converter.core.models import (
    BaseArchitecture, ModelInfo, ModelSource, ModelType,
)
from coreml_converter.core.registry.base import RegistryClient

logger = logging.getLogger(__name__)


class Registry:
    def __init__(self, hf_client: RegistryClient | None = None,
                 civitai_client: RegistryClient | None = None) -> None:
        self._clients: dict[ModelSource, RegistryClient] = {}
        if hf_client:
            self._clients[ModelSource.HUGGINGFACE] = hf_client
        if civitai_client:
            self._clients[ModelSource.CIVITAI] = civitai_client

    def search(self, query: str, source: ModelSource | None = None,
               model_type: ModelType | None = None,
               base_arch: BaseArchitecture | None = None,
               limit: int = 20) -> list[ModelInfo]:
        clients = (
            {source: self._clients[source]}
            if source and source in self._clients
            else self._clients
        )
        results: list[ModelInfo] = []
        for src, client in clients.items():
            try:
                results.extend(client.search(query, model_type=model_type, base_arch=base_arch, limit=limit))
            except Exception:
                logger.exception(f"Search failed for {src.value}")
        return results

    def get_compatible_loras(self, base_model: ModelInfo, limit: int = 20) -> list[ModelInfo]:
        results: list[ModelInfo] = []
        for src, client in self._clients.items():
            try:
                results.extend(client.get_compatible_loras(base_model, limit=limit))
            except Exception:
                logger.exception(f"LoRA search failed for {src.value}")
        return results

    def download(self, model: ModelInfo, dest: Path) -> Path:
        client = self._clients.get(model.source)
        if client is None:
            raise ValueError(f"No client registered for {model.source.value}")
        return client.download(model, dest)
