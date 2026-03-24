# tests/web/test_history_routes.py
import pytest
from unittest.mock import MagicMock
from httpx import AsyncClient, ASGITransport
from coreml_converter.web.app import create_app
from coreml_converter.core.models import (
    BuildRecord, Recipe, ModelInfo, ModelSource, BaseArchitecture,
    ModelType, ConversionConfig, BuildStatus,
)
from coreml_converter.core.state import BuildStore
from pathlib import Path


def _make_record(name: str = "test") -> BuildRecord:
    base = ModelInfo(
        source=ModelSource.CIVITAI, id="1", name="Base",
        base_architecture=BaseArchitecture.SD15,
        model_type=ModelType.CHECKPOINT, tags=[], download_url="", metadata={},
    )
    config = ConversionConfig(output_dir=Path("/tmp"), model_name=name)
    recipe = Recipe(name=name, base_model=base, loras=[], conversion_config=config)
    return BuildRecord(recipe=recipe, status=BuildStatus.COMPLETED)


@pytest.fixture
def app(tmp_path):
    application = create_app()
    application.state.registry = MagicMock()
    store = BuildStore(tmp_path / "builds.json")
    store.save(_make_record("build-1"))
    store.save(_make_record("build-2"))
    application.state.build_store = store
    return application


class TestHistoryPage:
    @pytest.mark.asyncio
    async def test_history_shows_builds(self, app):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/history")
        assert resp.status_code == 200
        assert "build-1" in resp.text
        assert "build-2" in resp.text
