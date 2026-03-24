import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock, call
from coreml_converter.core.models import (
    ModelInfo, ModelSource, BaseArchitecture, ModelType,
    LoRAEntry, ConversionConfig, Recipe,
)
from coreml_converter.core.merger.merger import Merger

def _make_recipe(loras=None):
    base = ModelInfo(source=ModelSource.CIVITAI, id="1", name="Base Model",
        base_architecture=BaseArchitecture.SD15, model_type=ModelType.CHECKPOINT,
        tags=[], download_url="", metadata={"local_path": "/tmp/base"})
    config = ConversionConfig(output_dir=Path("/tmp/output"), model_name="test")
    return Recipe(name="test", base_model=base, loras=loras or [], conversion_config=config)

def _make_lora(name, weight=0.7):
    model = ModelInfo(source=ModelSource.CIVITAI, id=name, name=name,
        base_architecture=BaseArchitecture.SD15, model_type=ModelType.LORA,
        tags=[], download_url="", metadata={"local_path": f"/tmp/{name}.safetensors"})
    return LoRAEntry(model=model, weight=weight)

class TestMerger:
    @patch("coreml_converter.core.merger.merger.StableDiffusionPipeline")
    def test_merge_no_loras(self, mock_pipeline_cls, tmp_path):
        mock_pipe = MagicMock()
        mock_pipeline_cls.from_pretrained.return_value = mock_pipe
        merger = Merger()
        result = merger.merge(_make_recipe(), cache_dir=Path("/tmp/cache"), output_dir=tmp_path)
        mock_pipe.save_pretrained.assert_called_once()

    @patch("coreml_converter.core.merger.merger.StableDiffusionPipeline")
    def test_merge_applies_loras_in_order(self, mock_pipeline_cls, tmp_path):
        mock_pipe = MagicMock()
        mock_pipeline_cls.from_pretrained.return_value = mock_pipe
        merger = Merger()
        merger.merge(_make_recipe([_make_lora("A", 0.8), _make_lora("B", 0.5)]),
                     cache_dir=Path("/tmp/cache"), output_dir=tmp_path)
        assert mock_pipe.load_lora_weights.call_count == 2
        assert mock_pipe.fuse_lora.call_count == 2

    @patch("coreml_converter.core.merger.merger.StableDiffusionPipeline")
    def test_merge_uses_correct_weights(self, mock_pipeline_cls, tmp_path):
        mock_pipe = MagicMock()
        mock_pipeline_cls.from_pretrained.return_value = mock_pipe
        merger = Merger()
        merger.merge(_make_recipe([_make_lora("A", 0.6)]),
                     cache_dir=Path("/tmp/cache"), output_dir=tmp_path)
        mock_pipe.fuse_lora.assert_called_once_with(lora_scale=0.6)
