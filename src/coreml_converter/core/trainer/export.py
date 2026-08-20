"""Export a peft LoRA adapter as a kohya-format safetensors file.

Two non-obvious requirements, both verified the hard way — get either wrong and
the file loads in nothing while looking superficially fine:

1. Keys must be re-prefixed with "unet." before conversion. diffusers only
   emits the required `lora_unet_` header when it sees "unet" in the key, but
   `get_peft_model_state_dict()` returns bare keys.
2. `lora_alpha` must equal `r`. The diffusers converter writes
   alpha = len(lora_down) for every module regardless of the configured alpha,
   so any other value silently produces a file whose alpha lies.
"""
from __future__ import annotations

from pathlib import Path


def build_kohya_state_dict(unet, text_encoder=None,
                           adapter_name: str | None = None) -> dict:
    """Kohya-format state dict for the UNet adapter, plus the text encoder's
    if one is attached.

    The text-encoder keys need a second fix-up: diffusers emits the SDXL-style
    `lora_te1_` prefix, but SD 1.5 tooling (A1111, Draw Things) expects
    `lora_te_`. Both load in our own merger; we write the SD 1.5 convention so
    the file stays portable.
    """
    from diffusers.utils.state_dict_utils import convert_state_dict_to_kohya
    from peft.utils import get_peft_model_state_dict
    import torch

    if not hasattr(unet, "peft_config"):
        raise ValueError(
            "model has no LoRA adapter attached — call add_adapter() first")

    peft_sd = {f"unet.{k}": v for k, v in get_peft_model_state_dict(unet).items()}
    kohya = convert_state_dict_to_kohya(peft_sd)

    if text_encoder is not None and hasattr(text_encoder, "peft_config"):
        te_sd = {f"text_encoder.{k}": v
                 for k, v in get_peft_model_state_dict(text_encoder).items()}
        te_kohya = convert_state_dict_to_kohya(te_sd)
        for key, value in te_kohya.items():
            kohya[key.replace("lora_te1_", "lora_te_", 1)] = value

    return {k: v.to(torch.float16).contiguous() for k, v in kohya.items()}


def export_kohya_lora(unet, path: Path, metadata: dict | None = None,
                      text_encoder=None) -> int:
    """Write the adapter to `path`; returns the tensor count."""
    from safetensors.torch import save_file

    state = build_kohya_state_dict(unet, text_encoder)
    if not state:
        raise ValueError("adapter state dict is empty — was an adapter attached?")
    bad = [k for k in state if not k.startswith(("lora_unet_", "lora_te_"))]
    if bad:
        raise ValueError(
            f"{len(bad)} keys lack a lora_unet_/lora_te_ prefix (e.g. {bad[0]}) "
            "— the file would not load; check the re-prefix step")

    path.parent.mkdir(parents=True, exist_ok=True)
    meta = {str(k): str(v) for k, v in (metadata or {}).items()}
    save_file(state, str(path), metadata=meta)
    return len(state)


def training_metadata(*, rank: int, steps: int, learning_rate: float,
                      resolution: int, image_count: int, trigger: str,
                      class_token: str, caption: str, base_checkpoint: str,
                      style_family: str = "realistic",
                      mode: str = "character") -> dict:
    """kohya `ss_*` keys are what A1111/Draw Things read; `fanny_*` are ours."""
    return {
        "ss_network_dim": rank,
        "ss_network_alpha": rank,
        "ss_steps": steps,
        "ss_learning_rate": learning_rate,
        "ss_resolution": f"({resolution}, {resolution})",
        "ss_num_train_images": image_count,
        "fanny_trigger": trigger,
        "fanny_class_token": class_token,
        "fanny_caption": caption,
        "fanny_base_checkpoint": base_checkpoint,
        # Recorded so the Build tab can warn when a LoRA is merged into a
        # checkpoint from a different visual family, where it transfers poorly.
        "fanny_style_family": style_family,
        "fanny_mode": mode,
    }
