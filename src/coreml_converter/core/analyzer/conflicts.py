from __future__ import annotations
from itertools import combinations
from coreml_converter.core.models import Conflict, LoRAEntry, Severity

_TAG_CATEGORIES = {
    "style": "style", "anime": "style", "realistic": "style", "photorealistic": "style",
    "cartoon": "style", "3d": "style", "illustration": "style", "painting": "style",
    "digital art": "style", "watercolor": "style", "oil painting": "style", "pixel art": "style",
    "character": "character", "male": "character", "female": "character", "person": "character",
    "face": "character", "portrait": "character",
    "concept": "concept", "object": "concept", "vehicle": "concept", "animal": "concept", "food": "concept",
    "clothing": "clothing", "outfit": "clothing", "armor": "clothing", "dress": "clothing", "uniform": "clothing",
    "pose": "pose", "action": "pose", "sitting": "pose", "standing": "pose",
    "background": "background", "landscape": "background", "interior": "background", "scenery": "background",
}

_COMPETING_PAIRS = {
    frozenset({"realistic", "anime"}), frozenset({"realistic", "cartoon"}),
    frozenset({"photorealistic", "anime"}), frozenset({"photorealistic", "cartoon"}),
    frozenset({"3d", "illustration"}),
}

def _categorize_lora(entry):
    categories = set()
    for tag in entry.model.tags:
        cat = _TAG_CATEGORIES.get(tag.lower())
        if cat:
            categories.add(cat)
    return categories

def _has_competing_tags(a, b):
    tags_a = {t.lower() for t in a.model.tags}
    tags_b = {t.lower() for t in b.model.tags}
    for pair in _COMPETING_PAIRS:
        if pair <= (tags_a | tags_b) and not pair <= tags_a and not pair <= tags_b:
            return True
    return False

def detect_tag_conflicts(loras):
    conflicts = []
    categorized = [(entry, _categorize_lora(entry)) for entry in loras]
    for (a, cats_a), (b, cats_b) in combinations(categorized, 2):
        shared = cats_a & cats_b
        if not shared:
            continue
        if _has_competing_tags(a, b):
            conflicts.append(Conflict(lora_a=a.model.name, lora_b=b.model.name,
                reason=f"Competing styles: {a.model.name} vs {b.model.name}", severity=Severity.WARNING))
        else:
            conflicts.append(Conflict(lora_a=a.model.name, lora_b=b.model.name,
                reason=f"Same category: {', '.join(shared)}", severity=Severity.INFO))
    return conflicts
