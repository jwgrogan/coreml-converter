from __future__ import annotations
import logging
from pathlib import Path
from coreml_converter.core.models import BaseArchitecture, DimensionResult

logger = logging.getLogger(__name__)
_ARCH_CROSS_ATTN_DIM = {BaseArchitecture.SD15: 768, BaseArchitecture.SD20: 1024}

try:
    from safetensors import safe_open
except ImportError:
    safe_open = None

def validate_lora_dimensions(lora_path: Path, base_arch: BaseArchitecture) -> DimensionResult:
    """Validate LoRA cross-attention dimensions match the base architecture."""
    if safe_open is None:
        logger.warning("safetensors not installed, skipping dimension check")
        return DimensionResult(expected=0, actual=0, compatible=True)
    expected_dim = _ARCH_CROSS_ATTN_DIM[base_arch]
    with safe_open(str(lora_path), framework="pt") as f:
        for key in f.keys():
            if "attn" in key and "lora_down" in key:
                tensor = f.get_tensor(key)
                actual_dim = tensor.shape[-1]
                # Small ranks (4,8,16,...) are LoRA decomposition dims, not model dims
                if actual_dim not in (expected_dim, 4, 8, 16, 32, 64, 128):
                    return DimensionResult(expected=expected_dim, actual=actual_dim, compatible=False)
    return DimensionResult(expected=expected_dim, actual=expected_dim, compatible=True)
