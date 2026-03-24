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
    @patch("coreml_converter.core.converter.converter.check_disk_space")
    @patch("coreml_converter.core.converter.converter.subprocess")
    def test_convert_calls_apple_script(self, mock_subprocess, mock_check_disk_space, tmp_path):
        mock_subprocess.run.return_value = MagicMock(returncode=0)
        converter = Converter()
        config = _make_config(tmp_path)
        merged_path = tmp_path / "merged"
        merged_path.mkdir()
        output_dir = config.output_dir / config.model_name
        output_dir.mkdir(parents=True)
        (output_dir / f"{config.model_name}.mlpackage").mkdir()
        (output_dir / f"{config.model_name}.mlmodelc").mkdir()
        result = converter.convert(merged_model_path=merged_path, recipe=_make_recipe(tmp_path))
        mock_subprocess.run.assert_called()

    def test_generate_manifest(self, tmp_path):
        converter = Converter()
        recipe = _make_recipe(tmp_path)
        manifest_path = tmp_path / "manifest.json"
        converter._write_manifest(recipe, manifest_path)
        manifest = json.loads(manifest_path.read_text())
        assert manifest["name"] == "test-model"
        assert manifest["schema_version"] == 1
        assert manifest["base_model"]["source"] == "civitai"
        assert manifest["conversion"]["compute_units"] == "all"
