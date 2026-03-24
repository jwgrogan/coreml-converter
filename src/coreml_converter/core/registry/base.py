from __future__ import annotations
from abc import ABC, abstractmethod
from pathlib import Path
from coreml_converter.core.models import BaseArchitecture, ModelInfo, ModelType


class RegistryClient(ABC):
    @abstractmethod
    def search(self, query: str, model_type: ModelType | None = None,
               base_arch: BaseArchitecture | None = None, limit: int = 20) -> list[ModelInfo]: ...

    @abstractmethod
    def get_compatible_loras(self, base_model: ModelInfo, limit: int = 20) -> list[ModelInfo]: ...

    @abstractmethod
    def download(self, model: ModelInfo, dest: Path) -> Path: ...
