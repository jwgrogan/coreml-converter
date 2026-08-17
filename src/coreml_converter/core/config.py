from __future__ import annotations
import json
import os
from pathlib import Path
from pydantic import BaseModel


# Fanny spawns and health-checks the converter on this port; keeping the
# CLI default the same means a hand-started `ccml serve` is also the one
# Fanny finds.
DEFAULT_PORT = 8898


class Config(BaseModel):
    compute_units: str = "all"
    attention: str = "split_einsum"
    output_dir: str = "./output"
    civitai_api_key: str | None = None
    schema_version: int = 1


def get_app_dir() -> Path:
    app_dir = Path(os.environ.get("COREML_CONVERTER_HOME", Path.home() / ".coreml-converter"))
    app_dir.mkdir(parents=True, exist_ok=True)
    (app_dir / "cache").mkdir(exist_ok=True)
    return app_dir


def save_config(config: Config, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(config.model_dump(), indent=2))


def load_config(path: Path) -> Config:
    env_key = os.environ.get("CIVITAI_API_KEY")
    if path.exists():
        data = json.loads(path.read_text())
        config = Config(**data)
        if config.civitai_api_key is None and env_key:
            config.civitai_api_key = env_key
        return config
    if env_key:
        return Config(civitai_api_key=env_key)
    return Config()
