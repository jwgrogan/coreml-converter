import json
import pytest
from pathlib import Path
from coreml_converter.core.config import Config, get_app_dir, load_config, save_config


class TestConfig:
    def test_default_config(self):
        config = Config()
        assert config.compute_units == "all"
        assert config.attention == "split_einsum"
        assert config.civitai_api_key is None
        assert config.schema_version == 1

    def test_config_with_api_key(self):
        config = Config(civitai_api_key="test-key-123")
        assert config.civitai_api_key == "test-key-123"


class TestConfigPersistence:
    def test_save_and_load(self, tmp_path):
        config_path = tmp_path / "config.json"
        config = Config(civitai_api_key="my-key", compute_units="cpuAndGPU")
        save_config(config, config_path)
        loaded = load_config(config_path)
        assert loaded.civitai_api_key == "my-key"
        assert loaded.compute_units == "cpuAndGPU"

    def test_load_missing_file_returns_defaults(self, tmp_path):
        config_path = tmp_path / "nonexistent.json"
        loaded = load_config(config_path)
        assert loaded.civitai_api_key is None
        assert loaded.compute_units == "all"

    def test_civitai_key_from_env(self, tmp_path, monkeypatch):
        monkeypatch.setenv("CIVITAI_API_KEY", "env-key")
        config_path = tmp_path / "config.json"
        loaded = load_config(config_path)
        assert loaded.civitai_api_key == "env-key"

    def test_file_key_overrides_env(self, tmp_path, monkeypatch):
        monkeypatch.setenv("CIVITAI_API_KEY", "env-key")
        config_path = tmp_path / "config.json"
        save_config(Config(civitai_api_key="file-key"), config_path)
        loaded = load_config(config_path)
        assert loaded.civitai_api_key == "file-key"


class TestAppDir:
    def test_app_dir_exists(self, tmp_path, monkeypatch):
        monkeypatch.setenv("COREML_CONVERTER_HOME", str(tmp_path / "app"))
        app_dir = get_app_dir()
        assert app_dir.exists()
        assert (app_dir / "cache").exists()
