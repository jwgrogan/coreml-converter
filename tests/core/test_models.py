import pytest
from pathlib import Path
from coreml_converter.core.models import (
    ModelSource, BaseArchitecture, ModelType, ModelInfo,
    LoRAEntry, ConversionConfig, ConversionResult, Recipe,
    BuildRecord, CompatibilityReport, Conflict, DimensionResult,
    Severity, RiskLevel, BuildStatus,
)


class TestEnums:
    def test_model_source_values(self):
        assert ModelSource.HUGGINGFACE == "huggingface"
        assert ModelSource.CIVITAI == "civitai"

    def test_base_architecture_values(self):
        assert BaseArchitecture.SD15 == "SD1.5"
        assert BaseArchitecture.SD20 == "SD2.0"

    def test_model_type_values(self):
        assert ModelType.CHECKPOINT == "checkpoint"
        assert ModelType.LORA == "lora"


class TestModelInfo:
    def test_create_checkpoint(self):
        info = ModelInfo(
            source=ModelSource.CIVITAI, id="12345", name="Realistic Vision V5.1",
            base_architecture=BaseArchitecture.SD15, model_type=ModelType.CHECKPOINT,
            tags=["realistic", "photorealistic"],
            download_url="https://civitai.com/api/download/models/12345",
            metadata={"download_count": 50000},
        )
        assert info.source == ModelSource.CIVITAI
        assert info.base_architecture == BaseArchitecture.SD15

    def test_create_lora(self):
        info = ModelInfo(
            source=ModelSource.HUGGINGFACE, id="user/lora-detail", name="Detail Tweaker",
            base_architecture=BaseArchitecture.SD15, model_type=ModelType.LORA,
            tags=["detail"], download_url="https://huggingface.co/user/lora-detail", metadata={},
        )
        assert info.model_type == ModelType.LORA


class TestLoRAEntry:
    def test_default_weight(self):
        model = ModelInfo(source=ModelSource.CIVITAI, id="1", name="Test",
            base_architecture=BaseArchitecture.SD15, model_type=ModelType.LORA,
            tags=[], download_url="", metadata={})
        entry = LoRAEntry(model=model)
        assert entry.weight == 1.0
        assert entry.recommended_weight is None
        assert entry.weight_source is None

    def test_custom_weight(self):
        model = ModelInfo(source=ModelSource.CIVITAI, id="1", name="Test",
            base_architecture=BaseArchitecture.SD15, model_type=ModelType.LORA,
            tags=[], download_url="", metadata={})
        entry = LoRAEntry(model=model, weight=0.7, recommended_weight=0.7, weight_source="creator")
        assert entry.weight == 0.7

    def test_weight_validation_bounds(self):
        model = ModelInfo(source=ModelSource.CIVITAI, id="1", name="Test",
            base_architecture=BaseArchitecture.SD15, model_type=ModelType.LORA,
            tags=[], download_url="", metadata={})
        with pytest.raises(ValueError):
            LoRAEntry(model=model, weight=-0.1)
        with pytest.raises(ValueError):
            LoRAEntry(model=model, weight=1.5)


class TestConversionConfig:
    def test_defaults(self):
        config = ConversionConfig(output_dir=Path("/tmp/output"), model_name="test-model")
        assert config.compute_units == "all"
        assert config.attention == "split_einsum"
        assert config.precision == "float16"
        assert config.include_safety_checker is False


class TestRecipe:
    def test_create_recipe(self):
        base = ModelInfo(source=ModelSource.CIVITAI, id="1", name="Base",
            base_architecture=BaseArchitecture.SD15, model_type=ModelType.CHECKPOINT,
            tags=[], download_url="", metadata={})
        config = ConversionConfig(output_dir=Path("/tmp"), model_name="test")
        recipe = Recipe(name="my-model", base_model=base, loras=[], conversion_config=config)
        assert recipe.name == "my-model"
        assert len(recipe.loras) == 0


class TestBuildRecord:
    def test_default_status(self):
        base = ModelInfo(source=ModelSource.CIVITAI, id="1", name="Base",
            base_architecture=BaseArchitecture.SD15, model_type=ModelType.CHECKPOINT,
            tags=[], download_url="", metadata={})
        config = ConversionConfig(output_dir=Path("/tmp"), model_name="test")
        recipe = Recipe(name="test", base_model=base, loras=[], conversion_config=config)
        record = BuildRecord(recipe=recipe)
        assert record.status == BuildStatus.PENDING
        assert record.id
        assert record.schema_version == 1


class TestCompatibilityReport:
    def test_compatible_report(self):
        report = CompatibilityReport(
            is_compatible=True, architecture_match=True, dimension_check=None,
            conflicts=[], lora_count_warning=None, overall_risk=RiskLevel.LOW,
        )
        assert report.is_compatible is True

    def test_incompatible_report(self):
        conflict = Conflict(lora_a="LoRA A", lora_b="LoRA B",
            reason="Same style category", severity=Severity.WARNING)
        report = CompatibilityReport(
            is_compatible=False, architecture_match=False,
            dimension_check=DimensionResult(expected=768, actual=1024, compatible=False),
            conflicts=[conflict], lora_count_warning="quality may degrade",
            overall_risk=RiskLevel.HIGH,
        )
        assert report.overall_risk == RiskLevel.HIGH
        assert len(report.conflicts) == 1
