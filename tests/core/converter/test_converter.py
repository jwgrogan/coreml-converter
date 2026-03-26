import json
import zipfile
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
from coreml_converter.core.models import (
    ConversionConfig, ConversionResult, ModelInfo, ModelSource,
    BaseArchitecture, ModelType, Recipe,
)
from coreml_converter.core.converter.converter import Converter, check_disk_space

def _make_config(tmp_path, **kwargs):
    defaults = dict(output_dir=tmp_path / "output", model_name="test-model",
        compute_units="all", attention="split_einsum", precision="float16")
    defaults.update(kwargs)
    return ConversionConfig(**defaults)

def _make_recipe(tmp_path, **config_kwargs):
    base = ModelInfo(source=ModelSource.CIVITAI, id="1", name="Base",
        base_architecture=BaseArchitecture.SD15, model_type=ModelType.CHECKPOINT,
        tags=[], download_url="", metadata={})
    return Recipe(name="test-model", base_model=base, loras=[],
                  conversion_config=_make_config(tmp_path, **config_kwargs))

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


class TestBundleTokenizer:
    def test_copies_tokenizer_files_to_both_locations(self, tmp_path):
        converter = Converter()
        merged_dir = tmp_path / "merged_pipeline"
        tokenizer_dir = merged_dir / "tokenizer"
        tokenizer_dir.mkdir(parents=True)
        (tokenizer_dir / "merges.txt").write_text("#version: 0.2\na b")
        (tokenizer_dir / "vocab.json").write_text('{"hello": 0}')

        output_dir = tmp_path / "output"
        compiled_dir = output_dir / "compiled"
        compiled_dir.mkdir(parents=True)

        converter._bundle_tokenizer(merged_dir, output_dir)

        # Root level
        assert (output_dir / "merges.txt").read_text() == "#version: 0.2\na b"
        assert (output_dir / "vocab.json").read_text() == '{"hello": 0}'
        # compiled/ level (for Studio import)
        assert (compiled_dir / "merges.txt").read_text() == "#version: 0.2\na b"
        assert (compiled_dir / "vocab.json").read_text() == '{"hello": 0}'

    def test_missing_tokenizer_files_warns(self, tmp_path, caplog):
        import logging
        converter = Converter()
        merged_dir = tmp_path / "merged_pipeline"
        (merged_dir / "tokenizer").mkdir(parents=True)
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        with caplog.at_level(logging.WARNING):
            converter._bundle_tokenizer(merged_dir, output_dir)

        assert not (output_dir / "merges.txt").exists()
        assert not (output_dir / "vocab.json").exists()


class TestStudioZip:
    def test_creates_zip_with_correct_structure(self, tmp_path):
        converter = Converter()
        output_dir = tmp_path / "output" / "my-model"
        compiled_dir = output_dir / "compiled"
        compiled_dir.mkdir(parents=True)
        (compiled_dir / "TextEncoder.mlmodelc").mkdir()
        (compiled_dir / "TextEncoder.mlmodelc" / "model.mil").write_text("data")
        (compiled_dir / "merges.txt").write_text("merges")
        (compiled_dir / "vocab.json").write_text("{}")
        (output_dir / "manifest.json").write_text("{}")
        (output_dir / "merges.txt").write_text("merges")
        (output_dir / "vocab.json").write_text("{}")

        zip_path = converter._create_studio_zip(output_dir, "my-model")

        assert zip_path.name == "my-model.studio.zip"
        assert zip_path.exists()
        with zipfile.ZipFile(zip_path) as zf:
            names = zf.namelist()
            assert "my-model/manifest.json" in names
            assert "my-model/merges.txt" in names
            assert "my-model/vocab.json" in names
            assert "my-model/compiled/merges.txt" in names
            assert "my-model/compiled/vocab.json" in names
            assert "my-model/compiled/TextEncoder.mlmodelc/model.mil" in names

    def test_studio_zip_deletes_output_dir(self, tmp_path):
        """Simulate the converter's studio flow: zip then delete output dir."""
        import shutil
        converter = Converter()
        output_dir = tmp_path / "output" / "my-model"
        compiled_dir = output_dir / "compiled"
        compiled_dir.mkdir(parents=True)
        (compiled_dir / "Unet.mlmodelc").mkdir()
        (compiled_dir / "Unet.mlmodelc" / "model.mil").write_text("data")
        (output_dir / "manifest.json").write_text("{}")

        zip_path = converter._create_studio_zip(output_dir, "my-model")
        shutil.rmtree(output_dir)

        assert zip_path.exists()
        assert not output_dir.exists()


class TestStudioConfig:
    def test_studio_defaults_false(self, tmp_path):
        config = _make_config(tmp_path)
        assert config.studio is False

    def test_studio_flag(self, tmp_path):
        config = _make_config(tmp_path, studio=True)
        assert config.studio is True
