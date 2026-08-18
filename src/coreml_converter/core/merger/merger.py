from __future__ import annotations
import logging
import shutil
from pathlib import Path
from typing import Callable

logger = logging.getLogger(__name__)

try:
    import torch
    from diffusers import StableDiffusionPipeline
except ImportError:
    torch = None
    StableDiffusionPipeline = None

from coreml_converter.core.models import Recipe


class Merger:
    def merge(self, recipe: Recipe, cache_dir: Path, output_dir: Path,
              progress_callback: Callable[[str, float], None] | None = None) -> Path:
        if StableDiffusionPipeline is None:
            raise RuntimeError("diffusers is not installed. Install with: pip install coreml-converter[ml]")

        def _report(msg, pct):
            if progress_callback:
                progress_callback(msg, pct)

        _report("Loading base model", 0.0)
        local_path = recipe.base_model.metadata.get("local_path")
        if local_path and Path(local_path).exists():
            model_path = Path(local_path)
        else:
            model_path = cache_dir / f"{recipe.base_model.source.value}_{recipe.base_model.id}"

        # float32 explicitly: "auto" is not a dtype diffusers accepts here (it
        # gets parsed as a device string and fails), and Apple's torch2coreml
        # loads this pipeline expecting full precision — it does its own fp16
        # conversion downstream.
        if model_path.is_file():
            if model_path.suffix == ".ckpt":
                logger.warning("WARNING: .ckpt files can contain arbitrary code. .safetensors format is recommended.")
                if progress_callback:
                    progress_callback("ckpt_security_warning", 0.0)
            pipe = StableDiffusionPipeline.from_single_file(
                str(model_path), torch_dtype=torch.float32, safety_checker=None
            )
        else:
            pipe = StableDiffusionPipeline.from_pretrained(
                str(model_path), torch_dtype=torch.float32, safety_checker=None
            )

        total_loras = len(recipe.loras)
        for i, entry in enumerate(recipe.loras):
            lora_path = entry.model.metadata.get("local_path")
            if lora_path and Path(lora_path).exists():
                lora_file = Path(lora_path)
            else:
                lora_file = cache_dir / f"{entry.model.source.value}_{entry.model.id}.safetensors"

            _report(f"Applying LoRA {i+1}/{total_loras}: {entry.model.name}", (i + 1) / (total_loras + 1))

            describe_lora_incompatibility(lora_file, entry.model.name)

            if lora_file.is_file():
                pipe.load_lora_weights(str(lora_file.parent), weight_name=lora_file.name)
            else:
                pipe.load_lora_weights(str(lora_file))

            pipe.fuse_lora(lora_scale=entry.weight)
            pipe.unload_lora_weights()

        merged_dir = output_dir / "merged_pipeline"
        if merged_dir.exists():
            shutil.rmtree(merged_dir)
        merged_dir.mkdir(parents=True)

        _report("Saving merged pipeline", 0.9)
        pipe.save_pretrained(str(merged_dir))
        _report("Merge complete", 1.0)
        return merged_dir


def describe_lora_incompatibility(lora_file: Path, name: str) -> None:
    """Reject LoRA layouts diffusers cannot map, with a usable explanation.

    diffusers converts Kohya LoRAs by rewriting keys onto its own UNet module
    names (`down_blocks`, `mid_block`, `up_blocks`). A LoRA trained against the
    original Stable Diffusion UNet instead uses `input_blocks` / `middle_block`
    / `output_blocks`, and the conversion produces module paths that do not
    exist — surfacing as a 400-character PEFT dump of "target modules not
    found", which says nothing about what is actually wrong or what to do.
    """
    if not lora_file.is_file():
        return
    try:
        from safetensors.torch import load_file
    except ImportError:
        return
    if lora_file.suffix.lower() != ".safetensors":
        return

    try:
        keys = list(load_file(str(lora_file)).keys())
    except Exception:
        # A load failure is the real loader's problem to report, not ours.
        return

    ldm_style = any(
        k.startswith(("lora_unet_input_blocks", "lora_unet_middle_block",
                      "lora_unet_output_blocks"))
        for k in keys
    )
    diffusers_style = any(
        k.startswith(("lora_unet_down_blocks", "lora_unet_mid_block",
                      "lora_unet_up_blocks"))
        for k in keys
    )
    if ldm_style and not diffusers_style:
        raise RuntimeError(
            f"LoRA '{name}' uses the original Stable Diffusion layout "
            f"(input_blocks/middle_block), which this converter cannot map onto "
            f"the diffusers UNet it merges into. LoRAs exported in diffusers "
            f"layout (down_blocks/up_blocks) work — most CivitAI SD 1.5 LoRAs "
            f"are. Re-export this one in diffusers format, or use a different "
            f"LoRA."
        )

    if any("hada_" in k for k in keys):
        raise RuntimeError(
            f"LoRA '{name}' is a LyCORIS/LoHa model, which diffusers cannot "
            f"fuse. Use a standard LoRA."
        )
    if any("lokr_" in k for k in keys):
        raise RuntimeError(
            f"LoRA '{name}' is a LyCORIS/LoKr model, which diffusers cannot "
            f"fuse. Use a standard LoRA."
        )
