import pytest
from coreml_converter.core.models import (
    ModelInfo, ModelSource, BaseArchitecture, ModelType, LoRAEntry, CompatibilityReport,
)
from coreml_converter.core.analyzer.compatibility import check_compatibility

def _make_base(arch=BaseArchitecture.SD15):
    return ModelInfo(source=ModelSource.CIVITAI, id="1", name="Base",
        base_architecture=arch, model_type=ModelType.CHECKPOINT, tags=[], download_url="", metadata={})

def _make_lora(arch=BaseArchitecture.SD15, tags=None):
    model = ModelInfo(source=ModelSource.CIVITAI, id="2", name="LoRA",
        base_architecture=arch, model_type=ModelType.LORA, tags=tags or [], download_url="", metadata={})
    return LoRAEntry(model=model, weight=0.7)

class TestCheckCompatibility:
    def test_compatible_same_architecture(self):
        report = check_compatibility(_make_base(), [_make_lora()])
        assert report.is_compatible is True
        assert report.architecture_match is True

    def test_incompatible_architecture_mismatch(self):
        report = check_compatibility(_make_base(BaseArchitecture.SD15), [_make_lora(BaseArchitecture.SD20)])
        assert report.is_compatible is False
        assert report.architecture_match is False

    def test_no_loras_is_compatible(self):
        report = check_compatibility(_make_base(), [])
        assert report.is_compatible is True
        assert report.lora_count_warning is None

    def test_soft_warning_at_4_loras(self):
        loras = [_make_lora() for _ in range(4)]
        report = check_compatibility(_make_base(), loras)
        assert report.lora_count_warning is not None
        assert "may degrade" in report.lora_count_warning

    def test_hard_warning_at_6_loras(self):
        loras = [_make_lora() for _ in range(6)]
        report = check_compatibility(_make_base(), loras)
        assert report.lora_count_warning is not None
        assert "artifacts" in report.lora_count_warning

    def test_mixed_architectures_flag_specific_loras(self):
        loras = [_make_lora(BaseArchitecture.SD15), _make_lora(BaseArchitecture.SD20)]
        report = check_compatibility(_make_base(BaseArchitecture.SD15), loras)
        assert report.is_compatible is False
