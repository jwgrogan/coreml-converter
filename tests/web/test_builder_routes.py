# tests/web/test_builder_routes.py
import pytest
from unittest.mock import MagicMock
from httpx import AsyncClient, ASGITransport
from coreml_converter.web.app import create_app
from coreml_converter.core.models import (
    ModelInfo, ModelSource, BaseArchitecture, ModelType,
)
from coreml_converter.core.state import BuildStore


@pytest.fixture
def mock_registry():
    registry = MagicMock()
    registry.search.return_value = []
    return registry


@pytest.fixture
def app(mock_registry, tmp_path):
    application = create_app()
    application.state.registry = mock_registry
    application.state.build_store = BuildStore(tmp_path / "builds.json")
    return application


class TestBuilderPage:
    @pytest.mark.asyncio
    async def test_builder_renders(self, app):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/build")
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_builder_with_base_param(self, app, mock_registry):
        model = ModelInfo(
            source=ModelSource.CIVITAI, id="1", name="Test",
            base_architecture=BaseArchitecture.SD15,
            model_type=ModelType.CHECKPOINT,
            tags=[], download_url="", metadata={},
        )
        mock_registry.search.return_value = [model]

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/build", params={"base": "civitai:1"})
        assert resp.status_code == 200
