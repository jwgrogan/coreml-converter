import pytest
from coreml_converter.core.models import ModelInfo, ModelSource, BaseArchitecture, ModelType
from coreml_converter.core.analyzer.weight_guidance import get_recommended_weight

def _make_lora(tags=None, metadata=None):
    return ModelInfo(source=ModelSource.CIVITAI, id="1", name="Test LoRA",
        base_architecture=BaseArchitecture.SD15, model_type=ModelType.LORA,
        tags=tags or [], download_url="", metadata=metadata or {})

class TestGetRecommendedWeight:
    def test_creator_specified_weight(self):
        model = _make_lora(metadata={"description": "Best results at weight: 0.6-0.8"})
        weight, source = get_recommended_weight(model)
        assert 0.6 <= weight <= 0.8
        assert source == "creator"

    def test_style_category_default(self):
        weight, source = get_recommended_weight(_make_lora(tags=["style", "anime"]))
        assert 0.6 <= weight <= 0.8
        assert source == "category_default"

    def test_character_category_default(self):
        weight, source = get_recommended_weight(_make_lora(tags=["character"]))
        assert 0.7 <= weight <= 0.9
        assert source == "category_default"

    def test_detail_category_default(self):
        weight, source = get_recommended_weight(_make_lora(tags=["detail"]))
        assert 0.4 <= weight <= 0.6
        assert source == "category_default"

    def test_unknown_tags_fallback_to_1(self):
        weight, source = get_recommended_weight(_make_lora(tags=["somethingunknown"]))
        assert weight == 1.0
        assert source is None

    def test_creator_weight_takes_priority(self):
        model = _make_lora(tags=["style", "anime"], metadata={"description": "Recommended weight: 0.3"})
        weight, source = get_recommended_weight(model)
        assert weight == pytest.approx(0.3, abs=0.05)
        assert source == "creator"
