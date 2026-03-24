import pytest
from coreml_converter.core.models import (
    ModelInfo, ModelSource, BaseArchitecture, ModelType, LoRAEntry, Conflict, Severity,
)
from coreml_converter.core.analyzer.conflicts import detect_tag_conflicts

def _make_lora(name, tags):
    model = ModelInfo(source=ModelSource.CIVITAI, id=name, name=name,
        base_architecture=BaseArchitecture.SD15, model_type=ModelType.LORA,
        tags=tags, download_url="", metadata={})
    return LoRAEntry(model=model, weight=0.7)

class TestDetectTagConflicts:
    def test_no_conflicts_different_categories(self):
        loras = [_make_lora("Style LoRA", ["anime", "style"]), _make_lora("Character LoRA", ["character", "female"])]
        assert len(detect_tag_conflicts(loras)) == 0

    def test_same_category_info(self):
        loras = [_make_lora("Char A", ["character", "male"]), _make_lora("Char B", ["character", "female"])]
        conflicts = detect_tag_conflicts(loras)
        assert len(conflicts) == 1
        assert conflicts[0].severity == Severity.INFO

    def test_competing_styles_warning(self):
        loras = [_make_lora("Realistic", ["realistic", "photorealistic", "style"]),
                 _make_lora("Anime", ["anime", "cartoon", "style"])]
        warnings = [c for c in detect_tag_conflicts(loras) if c.severity == Severity.WARNING]
        assert len(warnings) >= 1

    def test_single_lora_no_conflicts(self):
        assert len(detect_tag_conflicts([_make_lora("Solo", ["style", "anime"])])) == 0

    def test_three_loras_same_category(self):
        loras = [_make_lora("A", ["style"]), _make_lora("B", ["style"]), _make_lora("C", ["style"])]
        assert len(detect_tag_conflicts(loras)) >= 2
