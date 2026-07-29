import json
import zipfile
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
from coreml_converter.core.models import (
    ConversionConfig, ConversionResult, ModelInfo, ModelSource,
    BaseArchitecture, ModelType, Recipe,
)
from coreml_converter.core.converter.converter import (
    Converter, check_disk_space, _apple_attention, _apple_compute_unit,
)

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

def _make_resources(resources: Path, *, components=("TextEncoder", "Unet", "VAEDecoder", "VAEEncoder")):
    """Build a fake Apple `Resources/` bundle: .mlmodelc dirs + tokenizer files."""
    resources.mkdir(parents=True, exist_ok=True)
    for comp in components:
        mlmodelc = resources / f"{comp}.mlmodelc"
        mlmodelc.mkdir()
        (mlmodelc / "model.mil").write_text("data")
    (resources / "vocab.json").write_text('{"hello": 0}')
    (resources / "merges.txt").write_text("#version: 0.2\na b")
    return resources


class TestCheckDiskSpace:
    def test_sufficient_space(self, tmp_path):
        check_disk_space(tmp_path, required_gb=0.001)

    def test_insufficient_space_raises(self, tmp_path):
        with pytest.raises(RuntimeError, match="Insufficient disk space"):
            check_disk_space(tmp_path, required_gb=999999)


class TestOptionMapping:
    def test_attention_maps_to_apple(self):
        assert _apple_attention("split_einsum") == "SPLIT_EINSUM"
        assert _apple_attention("split_einsum_v2") == "SPLIT_EINSUM_V2"
        assert _apple_attention("original") == "ORIGINAL"
        assert _apple_attention("nonsense") == "SPLIT_EINSUM"  # safe default

    def test_compute_unit_maps_to_apple(self):
        assert _apple_compute_unit("all") == "ALL"
        assert _apple_compute_unit("cpuAndGPU") == "CPU_AND_GPU"
        assert _apple_compute_unit("cpu_and_ne") == "CPU_AND_NE"
        assert _apple_compute_unit("cpu") == "CPU_ONLY"
        assert _apple_compute_unit("nonsense") == "ALL"  # safe default


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


class TestRequireAppleConverter:
    def test_missing_raises_with_install_hint(self):
        with patch("coreml_converter.core.converter.converter.subprocess.run") as run:
            run.return_value = MagicMock(returncode=1, stderr="No module")
            with pytest.raises(RuntimeError, match="ml-stable-diffusion"):
                Converter._require_apple_converter()

    def test_present_does_not_raise(self):
        with patch("coreml_converter.core.converter.converter.subprocess.run") as run:
            run.return_value = MagicMock(returncode=0, stderr="")
            Converter._require_apple_converter()  # no raise


class TestInstallResourcesFlat:
    def test_copies_mlmodelc_and_tokenizer_flat(self, tmp_path):
        resources = _make_resources(tmp_path / "work" / "Resources")
        output_dir = tmp_path / "output" / "test-model"
        output_dir.mkdir(parents=True)

        installed = Converter._install_resources_flat(resources, output_dir)

        # .mlmodelc directories land flat at the top level (not under compiled/)
        for comp in ("TextEncoder", "Unet", "VAEDecoder", "VAEEncoder"):
            assert (output_dir / f"{comp}.mlmodelc" / "model.mil").read_text() == "data"
        # tokenizer files flat at top level
        assert (output_dir / "vocab.json").read_text() == '{"hello": 0}'
        assert (output_dir / "merges.txt").read_text() == "#version: 0.2\na b"
        assert set(installed) == {"TextEncoder", "Unet", "VAEDecoder", "VAEEncoder"}

    def test_chunked_unet_reported(self, tmp_path):
        resources = _make_resources(
            tmp_path / "Resources",
            components=("TextEncoder", "UnetChunk1", "UnetChunk2", "VAEDecoder"),
        )
        output_dir = tmp_path / "out"
        output_dir.mkdir()
        installed = Converter._install_resources_flat(resources, output_dir)
        assert "UnetChunk1" in installed and "UnetChunk2" in installed
        assert (output_dir / "UnetChunk1.mlmodelc").is_dir()


class TestConvertDelegatesToApple:
    def test_convert_installs_flat_and_writes_manifest(self, tmp_path):
        recipe = _make_recipe(tmp_path)

        def fake_run(self, *, merged_model_path, work_dir, latent_h, latent_w,
                     attention, compute_unit, report):
            # Apple would populate <work_dir>/Resources — simulate that.
            return _make_resources(Path(work_dir) / "Resources")

        with patch("coreml_converter.core.converter.converter.check_disk_space"), \
             patch.object(Converter, "_require_apple_converter"), \
             patch.object(Converter, "_run_apple_converter", new=fake_run):
            result = Converter().convert(tmp_path / "merged", recipe)

        model_dir = recipe.conversion_config.output_dir / "test-model"
        # Flat, Fanny-compatible layout
        assert (model_dir / "TextEncoder.mlmodelc").is_dir()
        assert (model_dir / "Unet.mlmodelc").is_dir()
        assert (model_dir / "VAEEncoder.mlmodelc").is_dir()  # img2img support
        assert (model_dir / "vocab.json").is_file()
        assert (model_dir / "merges.txt").is_file()
        assert (model_dir / "manifest.json").is_file()
        # No stale scratch dirs left behind
        assert not any(p.name.startswith("fanny-convert-")
                       for p in recipe.conversion_config.output_dir.iterdir())
        assert isinstance(result, ConversionResult)
        assert result.mlmodelc_path == model_dir

    def test_convert_studio_zips_flat_layout(self, tmp_path):
        recipe = _make_recipe(tmp_path, studio=True)

        def fake_run(self, *, merged_model_path, work_dir, latent_h, latent_w,
                     attention, compute_unit, report):
            return _make_resources(Path(work_dir) / "Resources")

        with patch("coreml_converter.core.converter.converter.check_disk_space"), \
             patch.object(Converter, "_require_apple_converter"), \
             patch.object(Converter, "_run_apple_converter", new=fake_run):
            result = Converter().convert(tmp_path / "merged", recipe)

        assert result.mlmodelc_path.name == "test-model.studio.zip"
        with zipfile.ZipFile(result.mlmodelc_path) as zf:
            names = zf.namelist()
            assert "test-model/TextEncoder.mlmodelc/model.mil" in names
            assert "test-model/vocab.json" in names
            assert "test-model/manifest.json" in names
            # flat — nothing under a compiled/ subdir
            assert not any("/compiled/" in n for n in names)


class TestStudioZip:
    def test_creates_zip_with_flat_structure(self, tmp_path):
        converter = Converter()
        output_dir = tmp_path / "output" / "my-model"
        output_dir.mkdir(parents=True)
        (output_dir / "TextEncoder.mlmodelc").mkdir()
        (output_dir / "TextEncoder.mlmodelc" / "model.mil").write_text("data")
        (output_dir / "merges.txt").write_text("merges")
        (output_dir / "vocab.json").write_text("{}")
        (output_dir / "manifest.json").write_text("{}")

        zip_path = converter._create_studio_zip(output_dir, "my-model")

        assert zip_path.name == "my-model.studio.zip"
        with zipfile.ZipFile(zip_path) as zf:
            names = zf.namelist()
            assert "my-model/manifest.json" in names
            assert "my-model/merges.txt" in names
            assert "my-model/vocab.json" in names
            assert "my-model/TextEncoder.mlmodelc/model.mil" in names


class TestStudioConfig:
    def test_studio_defaults_false(self, tmp_path):
        config = _make_config(tmp_path)
        assert config.studio is False

    def test_studio_flag(self, tmp_path):
        config = _make_config(tmp_path, studio=True)
        assert config.studio is True
