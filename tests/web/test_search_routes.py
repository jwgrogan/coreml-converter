# tests/web/test_search_routes.py
import pytest
from unittest.mock import MagicMock, patch
from httpx import AsyncClient, ASGITransport
from coreml_converter.web.app import create_app
from coreml_converter.core.models import (
    ModelInfo, ModelSource, BaseArchitecture, ModelType,
)


def _make_model(name: str) -> ModelInfo:
    return ModelInfo(
        source=ModelSource.CIVITAI, id="1", name=name,
        base_architecture=BaseArchitecture.SD15,
        model_type=ModelType.CHECKPOINT,
        tags=["realistic"], download_url="", metadata={"download_count": 1000},
    )


@pytest.fixture
def mock_registry():
    registry = MagicMock()
    registry.search.return_value = [_make_model("Test Model")]
    return registry


@pytest.fixture
def app(mock_registry):
    application = create_app()
    application.state.registry = mock_registry
    return application


class TestSearchPage:
    @pytest.mark.asyncio
    async def test_home_page_renders(self, app):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/")
        assert resp.status_code == 200
        assert "CoreML Converter" in resp.text

    @pytest.mark.asyncio
    async def test_search_returns_results(self, app, mock_registry):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/search", params={"q": "test"})
        assert resp.status_code == 200
        assert "Test Model" in resp.text

    @pytest.mark.asyncio
    async def test_search_empty_query(self, app):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/search", params={"q": ""})
        assert resp.status_code == 200
