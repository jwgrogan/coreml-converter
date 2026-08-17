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
