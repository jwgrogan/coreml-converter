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
    import coremltools as ct
    mapping = {
        "all": ct.ComputeUnit.ALL,
        "cpuAndGPU": ct.ComputeUnit.CPU_AND_GPU,
        "cpuandgpu": ct.ComputeUnit.CPU_AND_GPU,
        "cpu_and_gpu": ct.ComputeUnit.CPU_AND_GPU,
    }
    return mapping.get(compute_units, ct.ComputeUnit.ALL)


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

        ct_precision = ct.precision.FLOAT16 if config.precision == "float16" else ct.precision.FLOAT32
        ct_compute = _compute_unit_enum(config.compute_units)

        def _report(msg, pct):
            if progress_callback:
                progress_callback(msg, pct)

        check_disk_space(config.output_dir)
        start_time = time.monotonic()

        # Load the merged pipeline
        _report("Loading merged pipeline for conversion", 0.05)
        pipe = StableDiffusionPipeline.from_pretrained(
            str(merged_model_path),
            torch_dtype=torch.float32,
        )

        # Determine sample shapes: SD1.5 = 64x64 latent, SD2.0 = 96x96
        arch = recipe.base_model.base_architecture.value
        latent_h, latent_w = (96, 96) if arch == "SD2.0" else (64, 64)
        hidden_size = pipe.text_encoder.config.hidden_size  # 768 for SD1.5, 1024 for SD2.0

        components_converted = []

        # --- 1. Text Encoder ---
        _report("Converting text encoder to CoreML", 0.10)
        try:
            text_encoder = pipe.text_encoder
            text_encoder.eval()

            # Wrapper to return only last_hidden_state tensor (not a dict)
            class TextEncoderWrapper(torch.nn.Module):
                def __init__(self, encoder):
                    super().__init__()
                    self.encoder = encoder

                def forward(self, input_ids):
                    return self.encoder(input_ids)[0]  # last_hidden_state

            wrapper = TextEncoderWrapper(text_encoder)
            wrapper.eval()

            sample_input = torch.randint(0, 1000, (1, 77))
            with torch.no_grad():
                traced = torch.jit.trace(wrapper, sample_input, strict=False)

            te_model = ct.convert(
                traced,
                inputs=[ct.TensorType(name="input_ids", shape=(1, 77), dtype=int)],
                convert_to="mlprogram",
                compute_precision=ct_precision,
                compute_units=ct_compute,
            )

            te_path = output_dir / "TextEncoder.mlpackage"
            te_model.save(str(te_path))
            components_converted.append("TextEncoder")
            logger.info("Text encoder converted successfully")
        except Exception as e:
            logger.error(f"Text encoder conversion failed: {e}")
            _report(f"Text encoder failed: {e}", 0.15)

        # --- 2. UNet ---
        _report("Converting UNet to CoreML (largest component, may take minutes)", 0.25)
        try:
            unet = pipe.unet
            unet.eval()

            # Wrapper to call with return_dict=False and return just the sample
            class UNetWrapper(torch.nn.Module):
                def __init__(self, unet):
                    super().__init__()
                    self.unet = unet

                def forward(self, sample, timestep, encoder_hidden_states):
                    return self.unet(sample, timestep, encoder_hidden_states, return_dict=False)[0]

            unet_wrapper = UNetWrapper(unet)
            unet_wrapper.eval()

            sample_latent = torch.randn(1, 4, latent_h, latent_w)
            sample_timestep = torch.tensor([1.0])
            sample_hidden = torch.randn(1, 77, hidden_size)

            with torch.no_grad():
                traced = torch.jit.trace(
                    unet_wrapper,
                    (sample_latent, sample_timestep, sample_hidden),
                    strict=False,
                )

            unet_model = ct.convert(
                traced,
                inputs=[
                    ct.TensorType(name="sample", shape=(1, 4, latent_h, latent_w)),
                    ct.TensorType(name="timestep", shape=(1,)),
                    ct.TensorType(name="encoder_hidden_states", shape=(1, 77, hidden_size)),
                ],
                convert_to="mlprogram",
                compute_precision=ct_precision,
                compute_units=ct_compute,
            )

            unet_path = output_dir / "Unet.mlpackage"
            unet_model.save(str(unet_path))
            components_converted.append("Unet")
            logger.info("UNet converted successfully")
        except Exception as e:
            logger.error(f"UNet conversion failed: {e}")
            _report(f"UNet failed: {e}", 0.55)

        # --- 3. VAE Decoder ---
        _report("Converting VAE decoder to CoreML", 0.70)
        try:
            vae = pipe.vae
            vae.eval()

            class VAEDecoderWrapper(torch.nn.Module):
                def __init__(self, vae):
                    super().__init__()
                    self.vae = vae

                def forward(self, z):
                    return self.vae.decode(z, return_dict=False)[0]

            vae_wrapper = VAEDecoderWrapper(vae)
            vae_wrapper.eval()

            sample_z = torch.randn(1, 4, latent_h, latent_w)
            with torch.no_grad():
                traced = torch.jit.trace(vae_wrapper, sample_z, strict=False)

            vae_model = ct.convert(
                traced,
                inputs=[
                    ct.TensorType(name="z", shape=(1, 4, latent_h, latent_w)),
                ],
                convert_to="mlprogram",
                compute_precision=ct_precision,
                compute_units=ct_compute,
            )

            vae_path = output_dir / "VAEDecoder.mlpackage"
            vae_model.save(str(vae_path))
            components_converted.append("VAEDecoder")
            logger.info("VAE decoder converted successfully")
        except Exception as e:
            logger.error(f"VAE decoder conversion failed: {e}")
            _report(f"VAE decoder failed: {e}", 0.80)

        elapsed = time.monotonic() - start_time

        if not components_converted:
            raise RuntimeError("All component conversions failed. Check logs for details.")

        # Write manifest
        _report("Writing manifest", 0.95)
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
            f"Conversion done: {len(components_converted)}/3 components, "
            f"{model_size_mb:.1f} MB, {elapsed:.1f}s"
        )

        return ConversionResult(
            mlpackage_path=output_dir,
            mlmodelc_path=output_dir,
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
