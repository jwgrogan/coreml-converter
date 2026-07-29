from __future__ import annotations

import json
import logging
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import coreml_converter
from coreml_converter.core.models import ConversionResult, Recipe

logger = logging.getLogger(__name__)

# Files Apple's `--bundle-resources-for-swift-cli` emits, in the flat layout
# apple/ml-stable-diffusion's Swift `StableDiffusionPipeline(resourcesAt:)`
# (and Fanny's ModelPackageValidator) expect at the top of the model folder.
_RESOURCE_MLMODELC = [
    "TextEncoder.mlmodelc",
    "Unet.mlmodelc",
    "UnetChunk1.mlmodelc",
    "UnetChunk2.mlmodelc",
    "VAEDecoder.mlmodelc",
    "VAEEncoder.mlmodelc",
]
_RESOURCE_TOKENIZER = ["vocab.json", "merges.txt"]


def check_disk_space(path: Path, required_gb: float = 20.0) -> None:
    stat = shutil.disk_usage(path)
    available_gb = stat.free / (1024 ** 3)
    if available_gb < required_gb:
        raise RuntimeError(
            f"Insufficient disk space: {available_gb:.1f} GB available, "
            f"{required_gb:.1f} GB required"
        )


def _apple_attention(value: str) -> str:
    """Map our config's attention string to Apple's --attention-implementation."""
    return {
        "split_einsum": "SPLIT_EINSUM",
        "split_einsum_v2": "SPLIT_EINSUM_V2",
        "original": "ORIGINAL",
    }.get((value or "").lower(), "SPLIT_EINSUM")


def _apple_compute_unit(value: str) -> str:
    """Map our config's compute-units string to Apple's --compute-unit choice.

    This only sets the conversion target; Fanny's runtime picks the actual
    compute unit at load (and falls back if the Neural Engine rejects a model).
    """
    return {
        "all": "ALL",
        "cpuandgpu": "CPU_AND_GPU",
        "cpu_and_gpu": "CPU_AND_GPU",
        "cpuandne": "CPU_AND_NE",
        "cpu_and_ne": "CPU_AND_NE",
        "cpu": "CPU_ONLY",
        "cpu_only": "CPU_ONLY",
    }.get((value or "").lower(), "ALL")


class Converter:
    """Convert a merged diffusers model to Apple-format CoreML.

    The actual PyTorch->CoreML conversion is delegated to Apple's official
    `python_coreml_stable_diffusion.torch2coreml`, which produces the tensor
    layout (batch-2 sample, rank-4 `encoder_hidden_states`), SPLIT_EINSUM
    attention, UNet chunking, and `.mlmodelc` bundling that apple/ml-stable-
    diffusion's Swift runtime requires. Our own value-add — search, download,
    single-file handling, and LoRA merging — happens upstream and hands this
    method a ready diffusers directory in `merged_model_path`.
    """

    def convert(
        self,
        merged_model_path: Path,
        recipe: Recipe,
        progress_callback: Callable[[str, float], None] | None = None,
    ) -> ConversionResult:
        config = recipe.conversion_config
        output_dir = config.output_dir / config.model_name
        output_dir.mkdir(parents=True, exist_ok=True)

        def _report(msg, pct):
            if progress_callback:
                progress_callback(msg, pct)

        check_disk_space(config.output_dir)
        start_time = time.monotonic()

        # SD1.5 = 64x64 latent (512px), SD2.0 = 96x96 (768px).
        arch = recipe.base_model.base_architecture.value
        latent_h, latent_w = (96, 96) if arch == "SD2.0" else (64, 64)
        attention = _apple_attention(config.attention)
        compute_unit = _apple_compute_unit(config.compute_units)

        _report("Preparing Apple ml-stable-diffusion converter", 0.05)
        self._require_apple_converter()

        # Convert into a scratch dir on the same volume (so the disk check
        # applies), then keep only the compiled Resources/ bundle — the raw
        # .mlpackage intermediates are ~2x the size and unused at runtime.
        work_dir = Path(tempfile.mkdtemp(prefix="fanny-convert-", dir=str(config.output_dir)))
        try:
            resources = self._run_apple_converter(
                merged_model_path=merged_model_path,
                work_dir=work_dir,
                latent_h=latent_h,
                latent_w=latent_w,
                attention=attention,
                compute_unit=compute_unit,
                report=_report,
            )
            components_converted = self._install_resources_flat(resources, output_dir)
        finally:
            shutil.rmtree(work_dir, ignore_errors=True)

        if not components_converted:
            raise RuntimeError(
                "Apple converter produced no usable .mlmodelc components. Check logs above."
            )

        elapsed = time.monotonic() - start_time

        # Write manifest alongside the model.
        _report("Writing manifest", 0.95)
        manifest_path = output_dir / "manifest.json"
        self._write_manifest(recipe, manifest_path, components_converted)

        # Studio mode: zip the (flat) model folder for import into Fanny.
        if config.studio:
            _report("Creating Studio zip archive", 0.97)
            zip_path = self._create_studio_zip(output_dir, config.model_name)
            shutil.rmtree(output_dir)
            zip_size_mb = zip_path.stat().st_size / (1024 ** 2)
            _report("Conversion complete", 1.0)
            logger.info(
                f"Conversion done ({attention}): {len(components_converted)} components, "
                f"{zip_size_mb:.1f} MB, {elapsed:.1f}s"
            )
            return ConversionResult(
                mlpackage_path=zip_path,
                mlmodelc_path=zip_path,
                manifest_path=zip_path,
                conversion_time=elapsed,
                model_size_mb=zip_size_mb,
            )

        model_size_mb = sum(
            f.stat().st_size for f in output_dir.rglob("*") if f.is_file()
        ) / (1024 ** 2)

        _report("Conversion complete", 1.0)
        logger.info(
            f"Conversion done ({attention}): {len(components_converted)} components, "
            f"{model_size_mb:.1f} MB, {elapsed:.1f}s"
        )
        return ConversionResult(
            mlpackage_path=output_dir,
            mlmodelc_path=output_dir,
            manifest_path=manifest_path,
            conversion_time=elapsed,
            model_size_mb=model_size_mb,
        )

    # --- Apple converter delegation ---

    @staticmethod
    def _require_apple_converter() -> None:
        """Fail early, with an actionable message, if Apple's tool is missing."""
        probe = subprocess.run(
            [sys.executable, "-c", "import python_coreml_stable_diffusion.torch2coreml"],
            capture_output=True,
            text=True,
        )
        if probe.returncode != 0:
            raise RuntimeError(
                "Apple's Core ML converter (python_coreml_stable_diffusion) is not "
                "installed in this environment. Install it with:\n"
                "    pip install git+https://github.com/apple/ml-stable-diffusion.git\n"
                "It is a large dependency (pulls in torch + coremltools) and is only "
                "needed for local conversion."
            )

    def _run_apple_converter(
        self,
        merged_model_path: Path,
        work_dir: Path,
        latent_h: int,
        latent_w: int,
        attention: str,
        compute_unit: str,
        report: Callable[[str, float], None],
    ) -> Path:
        """Run torch2coreml and return the resulting Resources/ directory."""
        cmd = [
            sys.executable, "-m", "python_coreml_stable_diffusion.torch2coreml",
            "--model-version", str(merged_model_path),
            "--convert-text-encoder",
            "--convert-unet",
            "--convert-vae-decoder",
            "--convert-vae-encoder",
            "--bundle-resources-for-swift-cli",
            "--attention-implementation", attention,
            "--compute-unit", compute_unit,
            "--latent-h", str(latent_h),
            "--latent-w", str(latent_w),
            "-o", str(work_dir),
        ]
        report(f"Converting with Apple ({attention}) — this can take several minutes", 0.15)
        logger.info("Running: %s", " ".join(cmd))

        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            stripped = line.rstrip()
            if stripped:
                logger.info("torch2coreml: %s", stripped)
            low = stripped.lower()
            # Coarse phase mapping from Apple's own log lines.
            if "text_encoder" in low and "converting" in low:
                report("Converting text encoder", 0.25)
            elif "unet" in low and "converting" in low and "chunk" not in low:
                report("Converting UNet (largest component)", 0.45)
            elif "vae_decoder" in low and "converting" in low:
                report("Converting VAE decoder", 0.70)
            elif "vae_encoder" in low and "converting" in low:
                report("Converting VAE encoder", 0.78)
            elif "bundle" in low or "resources" in low or "compiled" in low:
                report("Bundling .mlmodelc resources", 0.85)
        proc.wait()

        if proc.returncode != 0:
            raise RuntimeError(
                f"Apple converter exited with code {proc.returncode}. See logs above."
            )

        resources = work_dir / "Resources"
        if not resources.is_dir():
            raise RuntimeError(
                "Apple converter did not produce a Resources/ bundle "
                "(expected with --bundle-resources-for-swift-cli)."
            )
        return resources

    @staticmethod
    def _install_resources_flat(resources: Path, output_dir: Path) -> list[str]:
        """Copy the compiled .mlmodelc + tokenizer files flat into output_dir.

        Returns the list of components installed. Fanny (and Apple's Swift
        pipeline) expect these at the top level of the model folder.
        """
        installed: list[str] = []

        for name in _RESOURCE_MLMODELC:
            src = resources / name
            if src.is_dir():
                dst = output_dir / name
                if dst.exists():
                    shutil.rmtree(dst)
                shutil.copytree(src, dst)
                installed.append(name.removesuffix(".mlmodelc"))

        for name in _RESOURCE_TOKENIZER:
            src = resources / name
            if src.is_file():
                shutil.copy2(src, output_dir / name)

        return installed

    # --- Packaging / manifest (unchanged behavior) ---

    def _create_studio_zip(self, output_dir: Path, model_name: str) -> Path:
        import zipfile
        zip_path = output_dir.parent / f"{model_name}.studio.zip"
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for file_path in sorted(output_dir.rglob("*")):
                if file_path.is_file():
                    arcname = f"{model_name}/{file_path.relative_to(output_dir)}"
                    zf.write(file_path, arcname)
        return zip_path

    def _write_manifest(self, recipe: Recipe, path: Path,
                        components: list[str] | None = None) -> None:
        manifest = {
            "schema_version": 1,
            "name": recipe.name,
            "created": datetime.now(timezone.utc).isoformat(),
            "base_model": {
                "source": recipe.base_model.source.value,
                "id": recipe.base_model.id,
                "name": recipe.base_model.name,
                "architecture": recipe.base_model.base_architecture.value,
            },
            "loras": [
                {
                    "source": e.model.source.value,
                    "id": e.model.id,
                    "name": e.model.name,
                    "weight": e.weight,
                }
                for e in recipe.loras
            ],
            "conversion": {
                "compute_units": recipe.conversion_config.compute_units,
                "attention": recipe.conversion_config.attention,
                "precision": recipe.conversion_config.precision,
                "include_safety_checker": recipe.conversion_config.include_safety_checker,
            },
            "components": components or [],
            "tool_version": coreml_converter.__version__,
        }
        path.write_text(json.dumps(manifest, indent=2))
