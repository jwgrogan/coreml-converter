# tests/core/merger/test_lora_compatibility.py
"""Real builds failed with a 400-character PEFT dump of "target modules not
found", which named neither the LoRA nor the cause. These check the layouts
are told apart before the loader is reached."""
import pytest
import torch
from safetensors.torch import save_file

from coreml_converter.core.merger.merger import describe_lora_incompatibility


def write_lora(path, keys):
    save_file({k: torch.zeros(2, 2) for k in keys}, str(path))
    return path


def test_diffusers_layout_is_accepted(tmp_path):
    f = write_lora(tmp_path / "ok.safetensors", [
        "lora_unet_down_blocks_0_attentions_0_proj_in.lora_up.weight",
        "lora_te_text_model_encoder_layers_0_mlp_fc1.lora_up.weight",
    ])
    describe_lora_incompatibility(f, "ok")  # must not raise


def test_original_sd_layout_is_rejected_with_a_usable_message(tmp_path):
    f = write_lora(tmp_path / "ldm.safetensors", [
        "lora_unet_input_blocks_0_0.lora_up.weight",
        "lora_unet_middle_block_1_proj_in.lora_up.weight",
    ])

    with pytest.raises(RuntimeError) as excinfo:
        describe_lora_incompatibility(f, "leah-lora")

    message = str(excinfo.value)
    assert "leah-lora" in message           # names the offending LoRA
    assert "input_blocks" in message        # names the actual cause
    assert "diffusers" in message           # says what does work


def test_a_lora_carrying_both_layouts_is_allowed(tmp_path):
    # Mixed keys mean diffusers has something to map; let the loader decide.
    f = write_lora(tmp_path / "mixed.safetensors", [
        "lora_unet_input_blocks_0_0.lora_up.weight",
        "lora_unet_down_blocks_0_attentions_0_proj_in.lora_up.weight",
    ])
    describe_lora_incompatibility(f, "mixed")


def test_loha_is_rejected(tmp_path):
    f = write_lora(tmp_path / "loha.safetensors", [
        "lora_unet_down_blocks_0.hada_w1_a",
    ])
    with pytest.raises(RuntimeError, match="LoHa"):
        describe_lora_incompatibility(f, "loha")


def test_lokr_is_rejected(tmp_path):
    f = write_lora(tmp_path / "lokr.safetensors", [
        "lora_unet_down_blocks_0.lokr_w1",
    ])
    with pytest.raises(RuntimeError, match="LoKr"):
        describe_lora_incompatibility(f, "lokr")


def test_missing_file_is_left_to_the_loader(tmp_path):
    describe_lora_incompatibility(tmp_path / "absent.safetensors", "absent")


def test_unreadable_file_is_left_to_the_loader(tmp_path):
    bad = tmp_path / "corrupt.safetensors"
    bad.write_bytes(b"not a safetensors file")
    describe_lora_incompatibility(bad, "corrupt")
