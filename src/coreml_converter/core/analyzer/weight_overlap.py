from __future__ import annotations
import logging
from itertools import combinations
from pathlib import Path
from coreml_converter.core.models import Conflict, Severity

logger = logging.getLogger(__name__)

try:
    import torch
    from safetensors import safe_open
except ImportError:
    torch = None
    safe_open = None

def _get_layer_norms(lora_path):
    norms = {}
    with safe_open(str(lora_path), framework="pt") as f:
        for key in f.keys():
            base_key = key.rsplit(".lora_", 1)[0] if ".lora_" in key else key
            tensor = f.get_tensor(key)
            norm = float(torch.norm(tensor.float()).item())
            norms[base_key] = norms.get(base_key, 0.0) + norm
    return norms

def detect_weight_overlap(loras, overlap_threshold=0.5):
    """Detect LoRA pairs with high weight overlap in the same layers."""
    if torch is None or safe_open is None:
        logger.warning("torch/safetensors not installed, skipping weight overlap check")
        return []
    if len(loras) < 2:
        return []
    lora_norms = []
    for name, path in loras:
        try:
            norms = _get_layer_norms(path)
            lora_norms.append((name, norms))
        except Exception as e:
            logger.warning(f"Failed to analyze {name}: {e}")
    conflicts = []
    for (name_a, norms_a), (name_b, norms_b) in combinations(lora_norms, 2):
        shared_keys = set(norms_a.keys()) & set(norms_b.keys())
        if not shared_keys:
            continue
        total_a = sum(norms_a.values())
        total_b = sum(norms_b.values())
        if total_a == 0 or total_b == 0:
            continue
        shared_mass_a = sum(norms_a[k] for k in shared_keys) / total_a
        shared_mass_b = sum(norms_b[k] for k in shared_keys) / total_b
        if shared_mass_a > overlap_threshold and shared_mass_b > overlap_threshold:
            conflicts.append(Conflict(lora_a=name_a, lora_b=name_b,
                reason=f"High weight overlap: {shared_mass_a:.0%} / {shared_mass_b:.0%} of weights in same layers",
                severity=Severity.WARNING))
    return conflicts
