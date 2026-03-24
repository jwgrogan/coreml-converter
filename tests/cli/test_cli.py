import pytest
from click.testing import CliRunner
from unittest.mock import patch, MagicMock
from coreml_converter.cli.main import cli
from coreml_converter.core.models import ModelInfo, ModelSource, BaseArchitecture, ModelType

def _make_model(name):
    return ModelInfo(source=ModelSource.CIVITAI, id="1", name=name,
        base_architecture=BaseArchitecture.SD15, model_type=ModelType.CHECKPOINT,
        tags=["realistic"], download_url="", metadata={"download_count": 1000})

class TestCLIGroup:
    def test_help(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "CoreML Converter" in result.output

    def test_version(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["--version"])
        assert result.exit_code == 0
        assert "0.1.0" in result.output

class TestSearchCommand:
    @patch("coreml_converter.cli.commands.search.get_registry")
    def test_search_displays_results(self, mock_get_registry):
        mock_registry = MagicMock()
        mock_registry.search.return_value = [_make_model("Test Model")]
        mock_get_registry.return_value = mock_registry
        runner = CliRunner()
        result = runner.invoke(cli, ["search", "test"])
        assert result.exit_code == 0
        assert "Test Model" in result.output

    @patch("coreml_converter.cli.commands.search.get_registry")
    def test_search_no_results(self, mock_get_registry):
        mock_registry = MagicMock()
        mock_registry.search.return_value = []
        mock_get_registry.return_value = mock_registry
        runner = CliRunner()
        result = runner.invoke(cli, ["search", "nonexistent"])
        assert result.exit_code == 0
        assert "No results" in result.output

    @patch("coreml_converter.cli.commands.search.get_registry")
    def test_search_with_source_filter(self, mock_get_registry):
        mock_registry = MagicMock()
        mock_registry.search.return_value = []
        mock_get_registry.return_value = mock_registry
        runner = CliRunner()
        runner.invoke(cli, ["search", "test", "--source", "civitai"])
        mock_registry.search.assert_called_once()
        call_kwargs = mock_registry.search.call_args
        assert call_kwargs.kwargs.get("source") == ModelSource.CIVITAI or call_kwargs[1].get("source") == ModelSource.CIVITAI
