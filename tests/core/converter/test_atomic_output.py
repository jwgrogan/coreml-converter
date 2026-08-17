# tests/core/converter/test_atomic_output.py
"""Output hygiene: a build publishes all-or-nothing and never strands scratch.

The failure these cover is what left the previous machine's models folder
full of 0-byte `coreml-*` dirs and abandoned `fanny-convert-*` scratch: the
converter used to create its output directory up front and write into it as
it went, so any failure published a partial model that Fanny's scanner would
then discover.
"""
import os
import time
from pathlib import Path
from unittest.mock import patch

import pytest

from coreml_converter.core.converter.converter import (
    SCRATCH_PREFIX,
    Converter,
    sweep_stale_scratch_dirs,
)
from coreml_converter.core.models import (
    BaseArchitecture,
    ConversionConfig,
    ModelInfo,
    ModelSource,
    ModelType,
    Recipe,
)


def make_recipe(output_dir: Path, name="built-model") -> Recipe:
    return Recipe(
        name=name,
        base_model=ModelInfo(
            source=ModelSource.CIVITAI,
            id="1",
            name="Base",
            base_architecture=BaseArchitecture.SD15,
            model_type=ModelType.CHECKPOINT,
            tags=[],
            download_url="",
            metadata={},
        ),
        loras=[],
        conversion_config=ConversionConfig(output_dir=output_dir, model_name=name),
    )


def fake_resources(work_dir: Path) -> Path:
    """Stand in for what Apple's converter emits under work_dir."""
    resources = work_dir / "Resources"
    for component in ("TextEncoder", "Unet", "VAEDecoder", "VAEEncoder"):
        compiled = resources / f"{component}.mlmodelc"
        compiled.mkdir(parents=True, exist_ok=True)
        (compiled / "coremldata.bin").write_bytes(b"weights")
    (resources / "vocab.json").write_text("{}")
    (resources / "merges.txt").write_text("")
    return resources


class TestScratchSweep:
    def test_removes_dirs_older_than_cutoff(self, tmp_path):
        stale = tmp_path / f"{SCRATCH_PREFIX}abandoned"
        stale.mkdir()
        (stale / "huge.bin").write_bytes(b"0" * 128)
        old = time.time() - (48 * 3600)
        os.utime(stale, (old, old))

        removed = sweep_stale_scratch_dirs(tmp_path)

        assert removed == [stale]
        assert not stale.exists()

    def test_keeps_recent_scratch_from_a_live_build(self, tmp_path):
        active = tmp_path / f"{SCRATCH_PREFIX}running"
        active.mkdir()

        assert sweep_stale_scratch_dirs(tmp_path) == []
        assert active.exists()

    def test_leaves_real_models_alone(self, tmp_path):
        model = tmp_path / "sd15-split-einsum"
        model.mkdir()
        old = time.time() - (48 * 3600)
        os.utime(model, (old, old))

        assert sweep_stale_scratch_dirs(tmp_path) == []
        assert model.exists()

    def test_missing_directory_is_not_an_error(self, tmp_path):
        assert sweep_stale_scratch_dirs(tmp_path / "nope") == []


class TestAtomicPublish:
    def test_successful_build_publishes_flat_layout(self, tmp_path):
        out = tmp_path / "models"
        recipe = make_recipe(out)

        with patch.object(Converter, "_require_apple_converter"), patch.object(
            Converter, "_run_apple_converter", side_effect=lambda **kw: fake_resources(kw["work_dir"])
        ):
            result = Converter().convert(tmp_path / "merged", recipe)

        model_dir = out / "built-model"
        assert model_dir.is_dir()
        assert (model_dir / "Unet.mlmodelc").is_dir()
        assert (model_dir / "TextEncoder.mlmodelc").is_dir()
        assert (model_dir / "VAEDecoder.mlmodelc").is_dir()
        assert (model_dir / "VAEEncoder.mlmodelc").is_dir()
        assert (model_dir / "vocab.json").is_file()
        assert (model_dir / "merges.txt").is_file()
        assert (model_dir / "manifest.json").is_file()
        assert result.manifest_path == model_dir / "manifest.json"

    def test_no_scratch_survives_a_successful_build(self, tmp_path):
        out = tmp_path / "models"
        recipe = make_recipe(out)

        with patch.object(Converter, "_require_apple_converter"), patch.object(
            Converter, "_run_apple_converter", side_effect=lambda **kw: fake_resources(kw["work_dir"])
        ):
            Converter().convert(tmp_path / "merged", recipe)

        assert [p.name for p in out.iterdir() if p.name.startswith(SCRATCH_PREFIX)] == []

    def test_failed_build_publishes_nothing(self, tmp_path):
        out = tmp_path / "models"
        recipe = make_recipe(out)

        def boom(**kwargs):
            raise RuntimeError("Apple converter exited with code 1")

        with patch.object(Converter, "_require_apple_converter"), patch.object(
            Converter, "_run_apple_converter", side_effect=boom
        ):
            with pytest.raises(RuntimeError):
                Converter().convert(tmp_path / "merged", recipe)

        # No partial model folder for Fanny's scanner to discover...
        assert not (out / "built-model").exists()
        # ...and no scratch left behind.
        assert [p.name for p in out.iterdir() if p.name.startswith(SCRATCH_PREFIX)] == []

    def test_build_producing_no_components_fails_without_publishing(self, tmp_path):
        out = tmp_path / "models"
        recipe = make_recipe(out)

        def empty_resources(**kwargs):
            resources = kwargs["work_dir"] / "Resources"
            resources.mkdir(parents=True, exist_ok=True)
            return resources

        with patch.object(Converter, "_require_apple_converter"), patch.object(
            Converter, "_run_apple_converter", side_effect=empty_resources
        ):
            with pytest.raises(RuntimeError, match="no usable .mlmodelc"):
                Converter().convert(tmp_path / "merged", recipe)

        assert not (out / "built-model").exists()

    def test_rebuild_replaces_previous_model(self, tmp_path):
        out = tmp_path / "models"
        recipe = make_recipe(out)

        stale_model = out / "built-model"
        stale_model.mkdir(parents=True)
        (stale_model / "leftover-from-old-build.txt").write_text("x")

        with patch.object(Converter, "_require_apple_converter"), patch.object(
            Converter, "_run_apple_converter", side_effect=lambda **kw: fake_resources(kw["work_dir"])
        ):
            Converter().convert(tmp_path / "merged", recipe)

        assert (stale_model / "Unet.mlmodelc").is_dir()
        # The replaced build is gone, not merged with the new one.
        assert not (stale_model / "leftover-from-old-build.txt").exists()
        assert [p.name for p in out.iterdir() if "superseded" in p.name] == []

    def test_failed_rebuild_leaves_previous_model_intact(self, tmp_path):
        out = tmp_path / "models"
        recipe = make_recipe(out)

        existing = out / "built-model"
        existing.mkdir(parents=True)
        (existing / "Unet.mlmodelc").mkdir()
        (existing / "marker.txt").write_text("still here")

        with patch.object(Converter, "_require_apple_converter"), patch.object(
            Converter, "_run_apple_converter", side_effect=RuntimeError("died")
        ):
            with pytest.raises(RuntimeError):
                Converter().convert(tmp_path / "merged", recipe)

        assert (existing / "marker.txt").read_text() == "still here"
