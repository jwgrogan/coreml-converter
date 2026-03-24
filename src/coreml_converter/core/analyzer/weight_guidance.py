from __future__ import annotations
import re
from coreml_converter.core.models import ModelInfo

_CATEGORY_DEFAULTS = {
    "style": (0.6, 0.8), "character": (0.7, 0.9), "detail": (0.4, 0.6),
    "texture": (0.4, 0.6), "concept": (0.7, 1.0), "clothing": (0.7, 0.9),
    "pose": (0.6, 0.8), "background": (0.5, 0.7),
}

_TAG_TO_CATEGORY = {
    "style": "style", "anime": "style", "realistic": "style", "photorealistic": "style", "cartoon": "style",
    "character": "character", "person": "character", "face": "character", "portrait": "character",
    "detail": "detail", "details": "detail", "tweaker": "detail",
    "texture": "texture", "concept": "concept", "object": "concept",
    "clothing": "clothing", "outfit": "clothing",
    "pose": "pose", "action": "pose",
    "background": "background", "landscape": "background",
}

_WEIGHT_PATTERNS = [
    re.compile(r"(?:recommended?\s+)?weight[:\s]+(\d+\.?\d*)\s*[-\u2013]\s*(\d+\.?\d*)", re.IGNORECASE),
    re.compile(r"(?:recommended?\s+)?weight[:\s]+(\d+\.?\d*)", re.IGNORECASE),
    re.compile(r"(?:best|optimal)\s+(?:results?\s+)?(?:at|with)\s+(?:weight[:\s]+)?(\d+\.?\d*)\s*[-\u2013]\s*(\d+\.?\d*)", re.IGNORECASE),
    re.compile(r"(?:best|optimal)\s+(?:results?\s+)?(?:at|with)\s+(?:weight[:\s]+)?(\d+\.?\d*)", re.IGNORECASE),
]

def _parse_creator_weight(description):
    for pattern in _WEIGHT_PATTERNS:
        match = pattern.search(description)
        if match:
            groups = match.groups()
            if len(groups) == 2:
                low, high = float(groups[0]), float(groups[1])
                if 0.0 <= low <= 1.5 and 0.0 <= high <= 1.5:
                    return (low + high) / 2
            elif len(groups) == 1:
                val = float(groups[0])
                if 0.0 <= val <= 1.5:
                    return val
    return None

def _infer_category(model):
    for tag in model.tags:
        cat = _TAG_TO_CATEGORY.get(tag.lower())
        if cat:
            return cat
    return None

def get_recommended_weight(model: ModelInfo) -> tuple[float, str | None]:
    description = model.metadata.get("description", "")
    if description:
        creator_weight = _parse_creator_weight(description)
        if creator_weight is not None:
            return creator_weight, "creator"
    category = _infer_category(model)
    if category and category in _CATEGORY_DEFAULTS:
        low, high = _CATEGORY_DEFAULTS[category]
        return (low + high) / 2, "category_default"
    return 1.0, None
