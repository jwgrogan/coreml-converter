from __future__ import annotations
from pathlib import Path
from huggingface_hub import HfApi, snapshot_download
from coreml_converter.core.models import (
    BaseArchitecture, ModelInfo, ModelSource, ModelType,
)
from coreml_converter.core.registry.base import RegistryClient

_ARCH_TAG_MAP = {
    "sd-1.5": BaseArchitecture.SD15,
    "sd-1.4": BaseArchitecture.SD15,
    "sd-2.0": BaseArchitecture.SD20,
    "sd-2.1": BaseArchitecture.SD20,
}

# Infer arch from diffusers pipeline tags
_PIPELINE_ARCH_MAP = {
    "diffusers:StableDiffusionPipeline": BaseArchitecture.SD15,
    "diffusers:StableDiffusionImg2ImgPipeline": BaseArchitecture.SD15,
    "diffusers:StableDiffusionInpaintPipeline": BaseArchitecture.SD15,
}


class HuggingFaceClient(RegistryClient):
    def __init__(self) -> None:
        self._api = HfApi()

    def _infer_architecture(self, tags: list[str]) -> BaseArchitecture | None:
        # Check explicit architecture tags first
        for tag in tags:
            tag_lower = tag.lower()
            if tag_lower in _ARCH_TAG_MAP:
                return _ARCH_TAG_MAP[tag_lower]
        # Fall back to diffusers pipeline tags
        for tag in tags:
            if tag in _PIPELINE_ARCH_MAP:
                return _PIPELINE_ARCH_MAP[tag]
        return None

    def _infer_model_type(self, tags: list[str]) -> ModelType:
        for tag in tags:
            if "lora" in tag.lower():
                return ModelType.LORA
        return ModelType.CHECKPOINT

    def search(self, query: str, model_type: ModelType | None = None,
               base_arch: BaseArchitecture | None = None, limit: int = 20) -> list[ModelInfo]:
        models = self._api.list_models(
            search=query, pipeline_tag="text-to-image",
            sort="downloads", limit=limit * 3,
        )
        results: list[ModelInfo] = []
        for m in models:
            tags = list(m.tags or [])
            arch = self._infer_architecture(tags)
            if arch is None:
                continue
            if base_arch and arch != base_arch:
                continue
            mt = self._infer_model_type(tags)
            if model_type and mt != model_type:
                continue
            results.append(ModelInfo(
                source=ModelSource.HUGGINGFACE, id=m.id,
                name=m.id.split("/")[-1] if "/" in m.id else m.id,
                base_architecture=arch, model_type=mt, tags=tags,
                download_url=f"https://huggingface.co/{m.id}",
                metadata={"downloads": getattr(m, "downloads", 0)},
            ))
            if len(results) >= limit:
                break
        return results

    def get_compatible_loras(self, base_model: ModelInfo, limit: int = 20) -> list[ModelInfo]:
        return self.search(query="lora", model_type=ModelType.LORA,
                          base_arch=base_model.base_architecture, limit=limit)

    def download(self, model: ModelInfo, dest: Path) -> Path:
        return Path(snapshot_download(repo_id=model.id,
                                      local_dir=str(dest / model.id.replace("/", "_"))))
