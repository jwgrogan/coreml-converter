import json
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
from click.testing import CliRunner
from coreml_converter.cli.main import cli
from coreml_converter.core.models import (
    ModelInfo, ModelSource, BaseArchitecture, ModelType,
)

def _make_model(name, model_type=ModelType.CHECKPOINT):
    return ModelInfo(source=ModelSource.CIVITAI, id="1", name=name,
        base_architecture=BaseArchitecture.SD15, model_type=model_type,
        tags=["realistic"], download_url="http://example.com/model",
        metadata={"download_count": 1000})

class TestEndToEnd:
    @patch("coreml_converter.cli.commands.search.get_registry")
    def test_search_to_info_flow(self, mock_get_registry):
        mock_registry = MagicMock()
        mock_registry.search.return_value = [_make_model("Test Model")]
        mock_get_registry.return_value = mock_registry
        runner = CliRunner()
        result = runner.invoke(cli, ["search", "test", "--source", "civitai", "--type", "checkpoint"])
        assert result.exit_code == 0
        assert "Test Model" in result.output

    def test_config_roundtrip(self, tmp_path, monkeypatch):
        monkeypatch.setenv("COREML_CONVERTER_HOME", str(tmp_path))
        from coreml_converter.core.config import get_app_dir, load_config, save_config, Config
        app_dir = get_app_dir()
        config = Config(civitai_api_key="test-key-123")
        save_config(config, app_dir / "config.json")
        loaded = load_config(app_dir / "config.json")
        assert loaded.civitai_api_key == "test-key-123"

    def test_build_store_roundtrip(self, tmp_path):
        from coreml_converter.core.state import BuildStore
        from coreml_converter.core.models import BuildRecord, Recipe, ConversionConfig, BuildStatus
        store = BuildStore(tmp_path / "builds.json")
        base = _make_model("Base")
        config = ConversionConfig(output_dir=Path("/tmp"), model_name="test")
        recipe = Recipe(name="test", base_model=base, loras=[], conversion_config=config)
        record = BuildRecord(recipe=recipe)
        store.save(record)
        record.status = BuildStatus.COMPLETED
        store.save(record)
        loaded = store.get(record.id)
        assert loaded.status == BuildStatus.COMPLETED
        assert len(store.list_all()) == 1
