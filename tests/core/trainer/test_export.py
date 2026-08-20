"""Export-format regression tests.

These guard the two mistakes that silently produce a LoRA file which loads in
nothing: dropping the `unet.` re-prefix (so the `lora_unet_` header is never
written) and letting alpha drift away from rank.
"""
import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model

from coreml_converter.core.trainer.export import (
    build_kohya_state_dict, export_kohya_lora, training_metadata,
)


class TinyAttention(nn.Module):
    """Module names mirror the SD UNet attention blocks peft targets."""

    def __init__(self):
        super().__init__()
        self.to_q = nn.Linear(8, 8, bias=False)
        self.to_k = nn.Linear(8, 8, bias=False)
        self.to_v = nn.Linear(8, 8, bias=False)
        self.to_out = nn.ModuleList([nn.Linear(8, 8, bias=False)])

    def forward(self, x):
        return self.to_out[0](self.to_q(x) + self.to_k(x) + self.to_v(x))


def adapted():
    model = TinyAttention()
    return get_peft_model(model, LoraConfig(
        r=4, lora_alpha=4, init_lora_weights="gaussian",
        target_modules=["to_q", "to_k", "to_v", "to_out.0"]))


def test_every_key_carries_the_lora_unet_prefix():
    state = build_kohya_state_dict(adapted())
    assert state, "adapter produced no tensors"
    assert all(k.startswith("lora_unet_") for k in state), (
        "missing lora_unet_ header — the file would not load anywhere")


def test_down_up_and_alpha_present_for_each_module():
    state = build_kohya_state_dict(adapted())
    downs = {k.split(".")[0] for k in state if k.endswith("lora_down.weight")}
    ups = {k.split(".")[0] for k in state if k.endswith("lora_up.weight")}
    alphas = {k.split(".")[0] for k in state if k.endswith(".alpha")}
    assert downs == ups == alphas
    assert len(downs) == 4


def test_alpha_equals_rank():
    state = build_kohya_state_dict(adapted())
    for key, value in state.items():
        if key.endswith(".alpha"):
            assert int(value.item()) == 4


def test_weights_are_fp16():
    state = build_kohya_state_dict(adapted())
    assert all(v.dtype == torch.float16
               for k, v in state.items() if k.endswith("weight"))


def test_export_writes_file_and_metadata(tmp_path):
    out = tmp_path / "x.safetensors"
    count = export_kohya_lora(adapted(), out, {"ss_network_dim": 4, "fanny_trigger": "t"})
    assert out.is_file() and count == 12

    from safetensors import safe_open
    with safe_open(str(out), "pt") as f:
        assert f.metadata()["fanny_trigger"] == "t"
        assert all(k.startswith("lora_unet_") for k in f.keys())


def test_export_rejects_a_model_with_no_adapter(tmp_path):
    import pytest
    with pytest.raises(ValueError):
        export_kohya_lora(TinyAttention(), tmp_path / "y.safetensors")


def test_metadata_keeps_alpha_and_dim_in_lockstep():
    meta = training_metadata(
        rank=16, steps=1200, learning_rate=1e-4, resolution=512, image_count=11,
        trigger="t", class_token="woman", caption="photo of t woman",
        base_checkpoint="base.safetensors")
    assert meta["ss_network_dim"] == meta["ss_network_alpha"] == 16


class TinyTextEncoder(nn.Module):
    """Module names mirror the CLIP text encoder layers peft targets."""

    def __init__(self):
        super().__init__()
        self.k_proj = nn.Linear(8, 8, bias=False)
        self.q_proj = nn.Linear(8, 8, bias=False)
        self.v_proj = nn.Linear(8, 8, bias=False)
        self.out_proj = nn.Linear(8, 8, bias=False)

    def forward(self, x):
        return self.out_proj(self.k_proj(x) + self.q_proj(x) + self.v_proj(x))


def adapted_text_encoder():
    return get_peft_model(TinyTextEncoder(), LoraConfig(
        r=4, lora_alpha=4, init_lora_weights="gaussian",
        target_modules=["k_proj", "q_proj", "v_proj", "out_proj"]))


def test_text_encoder_keys_use_the_sd15_lora_te_prefix():
    """diffusers emits the SDXL-style lora_te1_; SD 1.5 tooling wants lora_te_."""
    state = build_kohya_state_dict(adapted(), adapted_text_encoder())
    te_keys = [k for k in state if k.startswith("lora_te_")]
    assert te_keys, "no text-encoder keys exported"
    assert not any(k.startswith("lora_te1_") for k in state)


def test_both_adapters_appear_when_the_text_encoder_is_trained():
    state = build_kohya_state_dict(adapted(), adapted_text_encoder())
    assert any(k.startswith("lora_unet_") for k in state)
    assert any(k.startswith("lora_te_") for k in state)


def test_text_encoder_is_omitted_when_not_trained():
    state = build_kohya_state_dict(adapted(), None)
    assert not any(k.startswith("lora_te_") for k in state)


def test_untrained_text_encoder_is_ignored_even_if_passed():
    """A bare module with no adapter must not break the export."""
    state = build_kohya_state_dict(adapted(), TinyTextEncoder())
    assert not any(k.startswith("lora_te_") for k in state)


def test_metadata_records_the_style_family_for_mismatch_warnings():
    """The Build tab needs this to warn when a LoRA meets a foreign family."""
    meta = training_metadata(
        rank=16, steps=1200, learning_rate=1e-4, resolution=512, image_count=11,
        trigger="t", class_token="woman", caption="photo of t woman",
        base_checkpoint="anylora.safetensors", style_family="anime", mode="character")
    assert meta["fanny_style_family"] == "anime"
    assert meta["fanny_mode"] == "character"


def test_metadata_defaults_to_realistic():
    meta = training_metadata(
        rank=16, steps=1200, learning_rate=1e-4, resolution=512, image_count=11,
        trigger="t", class_token="woman", caption="c", base_checkpoint="b")
    assert meta["fanny_style_family"] == "realistic"
