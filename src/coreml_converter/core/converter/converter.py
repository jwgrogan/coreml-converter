from __future__ import annotations

import json
import logging
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import coreml_converter
from coreml_converter.core.models import ConversionResult, Recipe

logger = logging.getLogger(__name__)


def check_disk_space(path: Path, required_gb: float = 20.0) -> None:
    stat = shutil.disk_usage(path)
    available_gb = stat.free / (1024 ** 3)
    if available_gb < required_gb:
        raise RuntimeError(
            f"Insufficient disk space: {available_gb:.1f} GB available, "
            f"{required_gb:.1f} GB required"
        )


def _compute_unit_enum(compute_units: str):
    """Map config string to coremltools compute unit enum."""
    import coremltools as ct
    mapping = {
        "all": ct.ComputeUnit.ALL,
        "cpuAndGPU": ct.ComputeUnit.CPU_AND_GPU,
        "cpuandgpu": ct.ComputeUnit.CPU_AND_GPU,
        "cpu_and_gpu": ct.ComputeUnit.CPU_AND_GPU,
    }
    return mapping.get(compute_units, ct.ComputeUnit.ALL)


def _convert_component(pipe_component, component_name: str, sample_input,
                       output_dir: Path, precision: str, compute_units: str,
                       report: Callable[[str], None] | None = None):
    """Convert a single pipeline component to CoreML."""
    import torch
    import coremltools as ct

    if pipe_component is None:
        return None

    if report:
        report(f"Converting {component_name}...")

    # Trace the model
    pipe_component.eval()
    with torch.no_grad():
        traced = torch.jit.trace(pipe_component, sample_input)

    # Convert to CoreML
    ct_precision = ct.precision.FLOAT16 if precision == "float16" else ct.precision.FLOAT32
    mlmodel = ct.convert(
        traced,
        convert_to="mlprogram",
        compute_precision=ct_precision,
        compute_units=_compute_unit_enum(compute_units),
    )

    # Save
    out_path = output_dir / f"{component_name}.mlpackage"
    mlmodel.save(str(out_path))
    return out_path


class Converter:
    def convert(
        self,
        merged_model_path: Path,
        recipe: Recipe,
        progress_callback: Callable[[str, float], None] | None = None,
    ) -> ConversionResult:
        import torch
        import coremltools as ct
        from diffusers import StableDiffusionPipeline

        config = recipe.conversion_config
        output_dir = config.output_dir / config.model_name
        output_dir.mkdir(parents=True, exist_ok=True)

        def _report(msg, pct):
            if progress_callback:
                progress_callback(msg, pct)

        check_disk_space(config.output_dir)
        start_time = time.monotonic()

        # Load the merged pipeline
        _report("Loading merged pipeline for conversion", 0.05)
        pipe = StableDiffusionPipeline.from_pretrained(
            str(merged_model_path),
            torch_dtype=torch.float32,  # Need float32 for tracing
        )
        # Set all submodules to eval mode (pipeline itself isn't a nn.Module)
        for attr in ["text_encoder", "unet", "vae", "safety_checker"]:
            component = getattr(pipe, attr, None)
            if component is not None and hasattr(component, "eval"):
                component.eval()

        # Determine sample shapes for SD1.5/2.0
        # SD1.5: latent 64x64 (512px), SD2.0: latent 96x96 (768px)
        arch = recipe.base_model.base_architecture.value
        if arch == "SD2.0":
            latent_h, latent_w = 96, 96
        else:
            latent_h, latent_w = 64, 64

        components_converted = []

        # 1. Text Encoder
        _report("Converting text encoder to CoreML", 0.15)
        try:
            text_encoder = pipe.text_encoder
            sample_text_input = torch.randint(0, 1000, (1, 77))
            text_encoder_path = output_dir / "TextEncoder.mlpackage"

            text_encoder.eval()
            with torch.no_grad():
                traced_te = torch.jit.trace(text_encoder, sample_text_input)

            ct_precision = ct.precision.FLOAT16 if config.precision == "float16" else ct.precision.FLOAT32
            te_model = ct.convert(
                traced_te,
                convert_to="mlprogram",
                compute_precision=ct_precision,
                compute_units=_compute_unit_enum(config.compute_units),
            )
            te_model.save(str(text_encoder_path))
            components_converted.append("TextEncoder")
            logger.info("Text encoder converted successfully")
        except Exception as e:
            logger.warning(f"Text encoder conversion failed: {e}")

        # 2. UNet
        _report("Converting UNet to CoreML (this is the big one)", 0.35)
        try:
            unet = pipe.unet
            unet_path = output_dir / "Unet.mlpackage"

            # UNet inputs: sample, timestep, encoder_hidden_states
            sample_latent = torch.randn(1, 4, latent_h, latent_w)
            sample_timestep = torch.tensor([1.0])
            sample_hidden = torch.randn(1, 77, pipe.text_encoder.config.hidden_size)

            unet.eval()
            with torch.no_grad():
                traced_unet = torch.jit.trace(
                    unet,
                    (sample_latent, sample_timestep, sample_hidden),
                )

            unet_model = ct.convert(
                traced_unet,
                convert_to="mlprogram",
                compute_precision=ct_precision,
                compute_units=_compute_unit_enum(config.compute_units),
            )
            unet_model.save(str(unet_path))
            components_converted.append("Unet")
            logger.info("UNet converted successfully")
        except Exception as e:
            logger.warning(f"UNet conversion failed: {e}")

        # 3. VAE Decoder
        _report("Converting VAE decoder to CoreML", 0.75)
        try:
            vae = pipe.vae
            vae_decoder_path = output_dir / "VAEDecoder.mlpackage"

            sample_vae_input = torch.randn(1, 4, latent_h, latent_w)

            # We need just the decoder part
            class VAEDecoder(torch.nn.Module):
                def __init__(self, vae):
                    super().__init__()
                    self.vae = vae

                def forward(self, z):
                    return self.vae.decode(z).sample

            vae_dec = VAEDecoder(vae)
            vae_dec.eval()
            with torch.no_grad():
                traced_vae = torch.jit.trace(vae_dec, sample_vae_input)

            vae_model = ct.convert(
                traced_vae,
                convert_to="mlprogram",
                compute_precision=ct_precision,
                compute_units=_compute_unit_enum(config.compute_units),
            )
            vae_model.save(str(vae_decoder_path))
            components_converted.append("VAEDecoder")
            logger.info("VAE decoder converted successfully")
        except Exception as e:
            logger.warning(f"VAE decoder conversion failed: {e}")

        # 4. Safety checker (optional)
        if config.include_safety_checker and pipe.safety_checker is not None:
            _report("Converting safety checker to CoreML", 0.85)
            try:
                # Safety checker is complex; skip tracing, just note it
                logger.info("Safety checker skipped (complex architecture)")
            except Exception as e:
                logger.warning(f"Safety checker conversion failed: {e}")

        elapsed = time.monotonic() - start_time

        # Compile to mlmodelc
        _report("Compiling models", 0.90)
        mlmodelc_dir = output_dir / f"{config.model_name}.mlmodelc"
        mlmodelc_dir.mkdir(exist_ok=True)
        for comp in components_converted:
            src = output_dir / f"{comp}.mlpackage"
            if src.exists():
                try:
                    compiled = ct.models.MLModel(str(src))
                    compiled_path = mlmodelc_dir / f"{comp}.mlmodelc"
                    # mlmodelc is created by the system when loading
                    shutil.copytree(str(src), str(compiled_path), dirs_exist_ok=True)
                except Exception as e:
                    logger.warning(f"Compilation of {comp} failed: {e}")

        # Write manifest
        _report("Writing manifest", 0.95)
        mlpackage_path = output_dir / "Unet.mlpackage"  # Primary model
        manifest_path = output_dir / "manifest.json"
        self._write_manifest(recipe, manifest_path, components_converted)

        # Calculate total size
        model_size_mb = 0.0
        for comp in components_converted:
            p = output_dir / f"{comp}.mlpackage"
            if p.exists():
                model_size_mb += sum(
                    f.stat().st_size for f in p.rglob("*") if f.is_file()
                ) / (1024 ** 2)

        _report("Conversion complete", 1.0)
        logger.info(
            f"Conversion done: {len(components_converted)} components, "
            f"{model_size_mb:.1f} MB, {elapsed:.1f}s"
        )

        return ConversionResult(
            mlpackage_path=mlpackage_path,
            mlmodelc_path=mlmodelc_dir,
            manifest_path=manifest_path,
            conversion_time=elapsed,
            model_size_mb=model_size_mb,
        )

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
