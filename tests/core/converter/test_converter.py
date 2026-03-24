import json
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
from coreml_converter.core.models import (
    ConversionConfig, ConversionResult, ModelInfo, ModelSource,
    BaseArchitecture, ModelType, Recipe,
)
from coreml_converter.core.converter.converter import Converter, check_disk_space

def _make_config(tmp_path):
    return ConversionConfig(output_dir=tmp_path / "output", model_name="test-model",
        compute_units="all", attention="split_einsum", precision="float16")

def _make_recipe(tmp_path):
    base = ModelInfo(source=ModelSource.CIVITAI, id="1", name="Base",
        base_architecture=BaseArchitecture.SD15, model_type=ModelType.CHECKPOINT,
        tags=[], download_url="", metadata={})
    return Recipe(name="test-model", base_model=base, loras=[], conversion_config=_make_config(tmp_path))

class TestCheckDiskSpace:
    def test_sufficient_space(self, tmp_path):
        check_disk_space(tmp_path, required_gb=0.001)

    def test_insufficient_space_raises(self, tmp_path):
        with pytest.raises(RuntimeError, match="Insufficient disk space"):
            check_disk_space(tmp_path, required_gb=999999)

class TestConverter:
    def test_generate_manifest(self, tmp_path):
        converter = Converter()
        recipe = _make_recipe(tmp_path)
        manifest_path = tmp_path / "manifest.json"
        converter._write_manifest(recipe, manifest_path, ["TextEncoder", "Unet", "VAEDecoder"])
        manifest = json.loads(manifest_path.read_text())
        assert manifest["name"] == "test-model"
        assert manifest["schema_version"] == 1
        assert manifest["base_model"]["source"] == "civitai"
        assert manifest["conversion"]["compute_units"] == "all"
        assert manifest["components"] == ["TextEncoder", "Unet", "VAEDecoder"]
