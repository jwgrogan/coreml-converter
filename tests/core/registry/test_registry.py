import pytest
from unittest.mock import MagicMock
from coreml_converter.core.models import (
    ModelSource, BaseArchitecture, ModelType, ModelInfo,
)
from coreml_converter.core.registry import Registry


def _make_model(source: ModelSource, name: str) -> ModelInfo:
    return ModelInfo(
        source=source, id="1", name=name,
        base_architecture=BaseArchitecture.SD15,
        model_type=ModelType.CHECKPOINT,
        tags=[], download_url="", metadata={},
    )


class TestRegistry:
    def test_search_all_sources(self):
        hf_client = MagicMock()
        civitai_client = MagicMock()
        hf_client.search.return_value = [_make_model(ModelSource.HUGGINGFACE, "HF Model")]
        civitai_client.search.return_value = [_make_model(ModelSource.CIVITAI, "Civitai Model")]
        registry = Registry(hf_client=hf_client, civitai_client=civitai_client)
        results = registry.search("test")
        assert len(results) == 2

    def test_search_single_source(self):
        hf_client = MagicMock()
        civitai_client = MagicMock()
        hf_client.search.return_value = [_make_model(ModelSource.HUGGINGFACE, "HF")]
        registry = Registry(hf_client=hf_client, civitai_client=civitai_client)
        results = registry.search("test", source=ModelSource.HUGGINGFACE)
        assert len(results) == 1
        civitai_client.search.assert_not_called()

    def test_search_handles_source_error_gracefully(self):
        hf_client = MagicMock()
        civitai_client = MagicMock()
        hf_client.search.side_effect = Exception("HF down")
        civitai_client.search.return_value = [_make_model(ModelSource.CIVITAI, "Civitai")]
        registry = Registry(hf_client=hf_client, civitai_client=civitai_client)
        results = registry.search("test")
        assert len(results) == 1
