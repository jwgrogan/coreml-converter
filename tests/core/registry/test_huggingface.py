import pytest
from unittest.mock import patch, MagicMock
from coreml_converter.core.models import ModelSource, BaseArchitecture, ModelType
from coreml_converter.core.registry.huggingface import HuggingFaceClient


class TestHuggingFaceClient:
    def test_search_returns_model_info_list(self):
        mock_model = MagicMock()
        mock_model.id = "runwayml/stable-diffusion-v1-5"
        mock_model.tags = ["stable-diffusion", "sd-1.5", "text-to-image"]
        mock_model.downloads = 100000
        mock_model.card_data = MagicMock()
        mock_model.card_data.tags = ["sd-1.5"]

        with patch("coreml_converter.core.registry.huggingface.HfApi") as mock_api:
            mock_api.return_value.list_models.return_value = [mock_model]
            client = HuggingFaceClient()
            results = client.search("stable diffusion", model_type=ModelType.CHECKPOINT)

        assert len(results) >= 1
        assert results[0].source == ModelSource.HUGGINGFACE
        assert results[0].id == "runwayml/stable-diffusion-v1-5"

    def test_search_filters_by_architecture(self):
        mock_sd15 = MagicMock()
        mock_sd15.id = "model/sd15"
        mock_sd15.tags = ["sd-1.5", "text-to-image"]
        mock_sd15.downloads = 100
        mock_sd15.card_data = MagicMock()
        mock_sd15.card_data.tags = ["sd-1.5"]

        mock_sd20 = MagicMock()
        mock_sd20.id = "model/sd20"
        mock_sd20.tags = ["sd-2.0", "text-to-image"]
        mock_sd20.downloads = 100
        mock_sd20.card_data = MagicMock()
        mock_sd20.card_data.tags = ["sd-2.0"]

        with patch("coreml_converter.core.registry.huggingface.HfApi") as mock_api:
            mock_api.return_value.list_models.return_value = [mock_sd15, mock_sd20]
            client = HuggingFaceClient()
            results = client.search("model", base_arch=BaseArchitecture.SD15)

        assert all(r.base_architecture == BaseArchitecture.SD15 for r in results)

    def test_infer_architecture_from_tags(self):
        client = HuggingFaceClient.__new__(HuggingFaceClient)
        assert client._infer_architecture(["sd-1.5", "other"]) == BaseArchitecture.SD15
        assert client._infer_architecture(["sd-2.0"]) == BaseArchitecture.SD20
        assert client._infer_architecture(["unrelated"]) is None
