import json
import pytest
import httpx
import respx
from coreml_converter.core.models import ModelSource, BaseArchitecture, ModelType
from coreml_converter.core.registry.civitai import CivitAIClient

CIVITAI_API = "https://civitai.com/api/v1"

MOCK_SEARCH_RESPONSE = {
    "items": [
        {
            "id": 4201,
            "name": "Realistic Vision V5.1",
            "type": "Checkpoint",
            "tags": ["realistic", "photorealistic"],
            "stats": {"downloadCount": 500000},
            "modelVersions": [
                {
                    "id": 29460,
                    "name": "V5.1",
                    "baseModel": "SD 1.5",
                    "files": [
                        {
                            "id": 1,
                            "name": "realisticVision.safetensors",
                            "downloadUrl": f"{CIVITAI_API}/download/models/29460",
                            "type": "Model",
                        }
                    ],
                    "images": [{"url": "https://example.com/img.png"}],
                }
            ],
        }
    ],
    "metadata": {"totalPages": 1, "currentPage": 1},
}


class TestCivitAIClient:
    @respx.mock
    def test_search_checkpoint(self):
        respx.get(f"{CIVITAI_API}/models").mock(
            return_value=httpx.Response(200, json=MOCK_SEARCH_RESPONSE)
        )
        client = CivitAIClient(api_key="test-key")
        results = client.search("realistic vision", model_type=ModelType.CHECKPOINT)
        assert len(results) == 1
        assert results[0].source == ModelSource.CIVITAI
        assert results[0].name == "Realistic Vision V5.1"
        assert results[0].base_architecture == BaseArchitecture.SD15

    @respx.mock
    def test_search_filters_by_architecture(self):
        response = {
            "items": [
                {
                    "id": 1, "name": "SD15 Model", "type": "Checkpoint", "tags": [],
                    "stats": {"downloadCount": 100},
                    "modelVersions": [{"id": 1, "name": "v1", "baseModel": "SD 1.5",
                        "files": [{"id": 1, "name": "m.safetensors", "downloadUrl": "http://x", "type": "Model"}],
                        "images": []}],
                },
                {
                    "id": 2, "name": "SD20 Model", "type": "Checkpoint", "tags": [],
                    "stats": {"downloadCount": 100},
                    "modelVersions": [{"id": 2, "name": "v1", "baseModel": "SD 2.0",
                        "files": [{"id": 2, "name": "m.safetensors", "downloadUrl": "http://x", "type": "Model"}],
                        "images": []}],
                },
            ],
            "metadata": {"totalPages": 1, "currentPage": 1},
        }
        respx.get(f"{CIVITAI_API}/models").mock(
            return_value=httpx.Response(200, json=response)
        )
        client = CivitAIClient(api_key="test-key")
        results = client.search("model", base_arch=BaseArchitecture.SD20)
        assert len(results) == 1
        assert results[0].base_architecture == BaseArchitecture.SD20

    @respx.mock
    def test_search_with_rate_limiting(self):
        respx.get(f"{CIVITAI_API}/models").mock(
            return_value=httpx.Response(200, json={"items": [], "metadata": {"totalPages": 1, "currentPage": 1}})
        )
        client = CivitAIClient(api_key="test-key")
        results = client.search("test")
        assert results == []

    def test_parse_base_model_string(self):
        client = CivitAIClient.__new__(CivitAIClient)
        assert client._parse_base_model("SD 1.5") == BaseArchitecture.SD15
        assert client._parse_base_model("SD 2.0") == BaseArchitecture.SD20
        assert client._parse_base_model("SD 2.1") == BaseArchitecture.SD20
        assert client._parse_base_model("SDXL") is None
