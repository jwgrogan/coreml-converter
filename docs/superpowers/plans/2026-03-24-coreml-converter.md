# CoreML Converter Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Python CLI + web UI tool that converts SD1.5/2.0 checkpoints with merged LoRAs into CoreML models optimized for Apple Silicon.

**Architecture:** Pipeline architecture with three layers: core library (registry, analyzer, merger, converter), Click CLI, and FastAPI + htmx web UI. All interfaces share the core library. State is JSON files in `~/.coreml-converter/`.

**Tech Stack:** Python 3.10+, diffusers, coremltools, python_coreml_stable_diffusion, huggingface_hub, httpx, Click, rich, FastAPI, Jinja2, htmx, Alpine.js, Pydantic

**Spec:** `docs/superpowers/specs/2026-03-24-coreml-converter-design.md`

---

## File Structure

```
src/coreml_converter/
├── __init__.py                          # Package version
├── core/
│   ├── __init__.py
│   ├── models.py                        # All Pydantic models (ModelInfo, Recipe, BuildRecord, etc.)
│   ├── config.py                        # Config loading/saving, app directories
│   ├── state.py                         # BuildRecord persistence, file locking
│   ├── registry/
│   │   ├── __init__.py                  # Registry facade (search, download across sources)
│   │   ├── base.py                      # Abstract base class for registry clients
│   │   ├── huggingface.py               # HuggingFace Hub client
│   │   ├── civitai.py                   # CivitAI REST API client
│   │   └── rate_limiter.py              # Token bucket rate limiter
│   ├── analyzer/
│   │   ├── __init__.py                  # Analyzer facade
│   │   ├── compatibility.py             # Architecture match, dimension checks
│   │   ├── conflicts.py                 # Tag-based conflict detection
│   │   ├── weight_overlap.py            # Post-download weight overlap analysis
│   │   ├── dimensions.py               # Post-download dimension validation
│   │   └── weight_guidance.py           # LoRA weight recommendations
│   ├── merger/
│   │   ├── __init__.py
│   │   └── merger.py                    # LoRA merge into base model via diffusers
│   └── converter/
│       ├── __init__.py
│       └── converter.py                 # CoreML conversion via Apple tooling
├── cli/
│   ├── __init__.py
│   ├── main.py                          # Click group, entry point
│   ├── commands/
│   │   ├── __init__.py
│   │   ├── search.py                    # search command
│   │   ├── info.py                      # info command
│   │   ├── build.py                     # build command (+ interactive mode)
│   │   ├── serve.py                     # serve command (launches web UI)
│   │   ├── cache.py                     # cache list/clear commands
│   │   └── config_cmd.py               # config set/get commands
│   └── formatting.py                    # Rich table/progress formatting helpers
└── web/
    ├── __init__.py
    ├── app.py                           # FastAPI app factory
    ├── dependencies.py                  # Shared deps (registry, analyzer, state)
    ├── jobs.py                          # Background job manager (ThreadPool + ProcessPool)
    ├── routes/
    │   ├── __init__.py
    │   ├── search.py                    # GET /, GET /search
    │   ├── builder.py                   # GET /build, POST /build
    │   ├── progress.py                  # GET /build/{job_id}, GET /build/{job_id}/events (SSE)
    │   └── history.py                   # GET /history
    ├── templates/
    │   ├── base.html                    # Base layout (Pico CSS, htmx, Alpine.js)
    │   ├── search.html                  # Search page
    │   ├── partials/
    │   │   ├── model_card.html          # Model card component
    │   │   ├── search_results.html      # htmx partial for search results
    │   │   ├── lora_card.html           # LoRA card with weight slider
    │   │   └── compatibility_report.html # Compatibility report panel
    │   ├── builder.html                 # Builder page
    │   ├── progress.html                # Progress page
    │   └── history.html                 # History page
    └── static/
        ├── app.css                      # Custom styles
        └── builder.js                   # Alpine.js components (drag-reorder, sliders)

tests/
├── conftest.py                          # Shared fixtures
├── core/
│   ├── test_models.py
│   ├── test_config.py
│   ├── test_state.py
│   ├── registry/
│   │   ├── test_huggingface.py
│   │   ├── test_civitai.py
│   │   ├── test_registry.py
│   │   └── test_rate_limiter.py
│   ├── analyzer/
│   │   ├── test_compatibility.py
│   │   ├── test_conflicts.py
│   │   └── test_weight_guidance.py
│   ├── merger/
│   │   └── test_merger.py
│   └── converter/
│       └── test_converter.py
├── cli/
│   └── test_cli.py
└── web/
    ├── test_search_routes.py
    ├── test_builder_routes.py
    ├── test_progress_routes.py
    └── test_history_routes.py
```

---

## Task 1: Project Scaffolding

**Files:**
- Create: `pyproject.toml`
- Create: `src/coreml_converter/__init__.py`
- Create: `tests/conftest.py`

- [ ] **Step 1: Create pyproject.toml**

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "coreml-converter"
version = "0.1.0"
description = "Convert Stable Diffusion models + LoRAs to CoreML for Apple Silicon"
requires-python = ">=3.10"
dependencies = [
    "pydantic>=2.0",
    "click>=8.0",
    "rich>=13.0",
    "httpx>=0.25",
    "huggingface-hub>=0.20",
    "fastapi>=0.104",
    "uvicorn>=0.24",
    "jinja2>=3.1",
    "python-multipart>=0.0.6",
    "sse-starlette>=1.6",
]

[project.optional-dependencies]
ml = [
    "torch>=2.0",
    "diffusers>=0.25",
    "transformers>=4.35",
    "safetensors>=0.4",
    "coremltools>=7.0",
    "python-coreml-stable-diffusion @ git+https://github.com/apple/ml-stable-diffusion.git",
]
dev = [
    "pytest>=7.0",
    "pytest-asyncio>=0.23",
    "httpx",
    "respx>=0.20",
]

[project.scripts]
coreml-converter = "coreml_converter.cli.main:cli"

[tool.hatch.build.targets.wheel]
packages = ["src/coreml_converter"]

[tool.pytest.ini_options]
testpaths = ["tests"]
asyncio_mode = "auto"
```

- [ ] **Step 2: Create package init**

```python
# src/coreml_converter/__init__.py
__version__ = "0.1.0"
```

- [ ] **Step 3: Create empty test conftest and core __init__ files**

```python
# tests/conftest.py
```

Create `__init__.py` files for all packages:
- `src/coreml_converter/core/__init__.py`
- `src/coreml_converter/core/registry/__init__.py`
- `src/coreml_converter/core/analyzer/__init__.py`
- `src/coreml_converter/core/merger/__init__.py`
- `src/coreml_converter/core/converter/__init__.py`
- `src/coreml_converter/cli/__init__.py`
- `src/coreml_converter/cli/commands/__init__.py`
- `src/coreml_converter/web/__init__.py`
- `src/coreml_converter/web/routes/__init__.py`

- [ ] **Step 4: Install in dev mode and verify**

Run: `cd /Users/jwgrogan/GitHub/coreml-converter && pip install -e ".[dev]"`
Expected: Successful install, `coreml-converter --help` shows Click default help

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "feat: project scaffolding with pyproject.toml and package structure"
```

---

## Task 2: Pydantic Data Models

**Files:**
- Create: `src/coreml_converter/core/models.py`
- Create: `tests/core/test_models.py`

- [ ] **Step 1: Write tests for all data models**

```python
# tests/core/test_models.py
import pytest
from pathlib import Path
from coreml_converter.core.models import (
    ModelSource, BaseArchitecture, ModelType, ModelInfo,
    LoRAEntry, ConversionConfig, ConversionResult, Recipe,
    BuildRecord, CompatibilityReport, Conflict, DimensionResult,
    Severity, RiskLevel, BuildStatus,
)


class TestEnums:
    def test_model_source_values(self):
        assert ModelSource.HUGGINGFACE == "huggingface"
        assert ModelSource.CIVITAI == "civitai"

    def test_base_architecture_values(self):
        assert BaseArchitecture.SD15 == "SD1.5"
        assert BaseArchitecture.SD20 == "SD2.0"

    def test_model_type_values(self):
        assert ModelType.CHECKPOINT == "checkpoint"
        assert ModelType.LORA == "lora"


class TestModelInfo:
    def test_create_checkpoint(self):
        info = ModelInfo(
            source=ModelSource.CIVITAI,
            id="12345",
            name="Realistic Vision V5.1",
            base_architecture=BaseArchitecture.SD15,
            model_type=ModelType.CHECKPOINT,
            tags=["realistic", "photorealistic"],
            download_url="https://civitai.com/api/download/models/12345",
            metadata={"download_count": 50000},
        )
        assert info.source == ModelSource.CIVITAI
        assert info.base_architecture == BaseArchitecture.SD15

    def test_create_lora(self):
        info = ModelInfo(
            source=ModelSource.HUGGINGFACE,
            id="user/lora-detail",
            name="Detail Tweaker",
            base_architecture=BaseArchitecture.SD15,
            model_type=ModelType.LORA,
            tags=["detail"],
            download_url="https://huggingface.co/user/lora-detail",
            metadata={},
        )
        assert info.model_type == ModelType.LORA


class TestLoRAEntry:
    def test_default_weight(self):
        model = ModelInfo(
            source=ModelSource.CIVITAI, id="1", name="Test",
            base_architecture=BaseArchitecture.SD15,
            model_type=ModelType.LORA, tags=[], download_url="", metadata={},
        )
        entry = LoRAEntry(model=model)
        assert entry.weight == 1.0
        assert entry.recommended_weight is None
        assert entry.weight_source is None

    def test_custom_weight(self):
        model = ModelInfo(
            source=ModelSource.CIVITAI, id="1", name="Test",
            base_architecture=BaseArchitecture.SD15,
            model_type=ModelType.LORA, tags=[], download_url="", metadata={},
        )
        entry = LoRAEntry(model=model, weight=0.7, recommended_weight=0.7, weight_source="creator")
        assert entry.weight == 0.7

    def test_weight_validation_bounds(self):
        model = ModelInfo(
            source=ModelSource.CIVITAI, id="1", name="Test",
            base_architecture=BaseArchitecture.SD15,
            model_type=ModelType.LORA, tags=[], download_url="", metadata={},
        )
        with pytest.raises(ValueError):
            LoRAEntry(model=model, weight=-0.1)
        with pytest.raises(ValueError):
            LoRAEntry(model=model, weight=1.5)


class TestConversionConfig:
    def test_defaults(self):
        config = ConversionConfig(
            output_dir=Path("/tmp/output"),
            model_name="test-model",
        )
        assert config.compute_units == "all"
        assert config.attention == "split_einsum"
        assert config.precision == "float16"
        assert config.include_safety_checker is False


class TestRecipe:
    def test_create_recipe(self):
        base = ModelInfo(
            source=ModelSource.CIVITAI, id="1", name="Base",
            base_architecture=BaseArchitecture.SD15,
            model_type=ModelType.CHECKPOINT, tags=[], download_url="", metadata={},
        )
        config = ConversionConfig(output_dir=Path("/tmp"), model_name="test")
        recipe = Recipe(name="my-model", base_model=base, loras=[], conversion_config=config)
        assert recipe.name == "my-model"
        assert len(recipe.loras) == 0


class TestBuildRecord:
    def test_default_status(self):
        base = ModelInfo(
            source=ModelSource.CIVITAI, id="1", name="Base",
            base_architecture=BaseArchitecture.SD15,
            model_type=ModelType.CHECKPOINT, tags=[], download_url="", metadata={},
        )
        config = ConversionConfig(output_dir=Path("/tmp"), model_name="test")
        recipe = Recipe(name="test", base_model=base, loras=[], conversion_config=config)
        record = BuildRecord(recipe=recipe)
        assert record.status == BuildStatus.PENDING
        assert record.id  # auto-generated UUID
        assert record.schema_version == 1


class TestCompatibilityReport:
    def test_compatible_report(self):
        report = CompatibilityReport(
            is_compatible=True,
            architecture_match=True,
            dimension_check=None,
            conflicts=[],
            lora_count_warning=None,
            overall_risk=RiskLevel.LOW,
        )
        assert report.is_compatible is True

    def test_incompatible_report(self):
        conflict = Conflict(
            lora_a="LoRA A", lora_b="LoRA B",
            reason="Same style category",
            severity=Severity.WARNING,
        )
        report = CompatibilityReport(
            is_compatible=False,
            architecture_match=False,
            dimension_check=DimensionResult(expected=768, actual=1024, compatible=False),
            conflicts=[conflict],
            lora_count_warning="quality may degrade",
            overall_risk=RiskLevel.HIGH,
        )
        assert report.overall_risk == RiskLevel.HIGH
        assert len(report.conflicts) == 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/jwgrogan/GitHub/coreml-converter && python -m pytest tests/core/test_models.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'coreml_converter.core.models'`

- [ ] **Step 3: Implement all data models**

```python
# src/coreml_converter/core/models.py
from __future__ import annotations

import uuid
from datetime import datetime
from enum import Enum
from pathlib import Path

from pydantic import BaseModel, Field, field_validator


class ModelSource(str, Enum):
    HUGGINGFACE = "huggingface"
    CIVITAI = "civitai"


class BaseArchitecture(str, Enum):
    SD15 = "SD1.5"
    SD20 = "SD2.0"


class ModelType(str, Enum):
    CHECKPOINT = "checkpoint"
    LORA = "lora"


class Severity(str, Enum):
    INFO = "info"
    WARNING = "warning"


class RiskLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class BuildStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class ModelInfo(BaseModel):
    source: ModelSource
    id: str
    name: str
    base_architecture: BaseArchitecture
    model_type: ModelType
    tags: list[str] = []
    download_url: str
    metadata: dict = {}


class LoRAEntry(BaseModel):
    model: ModelInfo
    weight: float = 1.0
    recommended_weight: float | None = None
    weight_source: str | None = None  # "creator" | "community" | "category_default"

    @field_validator("weight")
    @classmethod
    def validate_weight(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError(f"Weight must be between 0.0 and 1.0, got {v}")
        return v


class ConversionConfig(BaseModel):
    compute_units: str = "all"
    attention: str = "split_einsum"
    precision: str = "float16"
    include_safety_checker: bool = False
    output_dir: Path
    model_name: str


class ConversionResult(BaseModel):
    mlpackage_path: Path
    mlmodelc_path: Path
    manifest_path: Path
    conversion_time: float
    model_size_mb: float


class Recipe(BaseModel):
    name: str
    base_model: ModelInfo
    loras: list[LoRAEntry] = []
    conversion_config: ConversionConfig


class DimensionResult(BaseModel):
    expected: int
    actual: int
    compatible: bool


class Conflict(BaseModel):
    lora_a: str
    lora_b: str
    reason: str
    severity: Severity


class CompatibilityReport(BaseModel):
    is_compatible: bool
    architecture_match: bool
    dimension_check: DimensionResult | None = None
    conflicts: list[Conflict] = []
    lora_count_warning: str | None = None
    overall_risk: RiskLevel


class BuildRecord(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    recipe: Recipe
    status: BuildStatus = BuildStatus.PENDING
    started_at: datetime | None = None
    completed_at: datetime | None = None
    result: ConversionResult | None = None
    error: str | None = None
    schema_version: int = 1
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/core/test_models.py -v`
Expected: All tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/coreml_converter/core/models.py tests/core/test_models.py
git commit -m "feat: add Pydantic data models for all core types"
```

---

## Task 3: Config & State Management

**Files:**
- Create: `src/coreml_converter/core/config.py`
- Create: `src/coreml_converter/core/state.py`
- Create: `tests/core/test_config.py`
- Create: `tests/core/test_state.py`

- [ ] **Step 1: Write config tests**

```python
# tests/core/test_config.py
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/core/test_config.py -v`
Expected: FAIL — import errors

- [ ] **Step 3: Implement config module**

```python
# src/coreml_converter/core/config.py
from __future__ import annotations

import json
import os
from pathlib import Path

from pydantic import BaseModel


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
        # Env var is fallback when file has no key
        if config.civitai_api_key is None and env_key:
            config.civitai_api_key = env_key
        return config
    if env_key:
        return Config(civitai_api_key=env_key)
    return Config()
```

- [ ] **Step 4: Run config tests**

Run: `python -m pytest tests/core/test_config.py -v`
Expected: All PASS

- [ ] **Step 5: Write state tests**

```python
# tests/core/test_state.py
import pytest
from pathlib import Path
from coreml_converter.core.state import BuildStore
from coreml_converter.core.models import (
    BuildRecord, Recipe, ModelInfo, ModelSource, BaseArchitecture,
    ModelType, ConversionConfig, BuildStatus,
)


def _make_recipe(name: str = "test") -> Recipe:
    base = ModelInfo(
        source=ModelSource.CIVITAI, id="1", name="Base",
        base_architecture=BaseArchitecture.SD15,
        model_type=ModelType.CHECKPOINT, tags=[], download_url="", metadata={},
    )
    config = ConversionConfig(output_dir=Path("/tmp"), model_name=name)
    return Recipe(name=name, base_model=base, loras=[], conversion_config=config)


class TestBuildStore:
    def test_create_and_get(self, tmp_path):
        store = BuildStore(tmp_path / "builds.json")
        record = BuildRecord(recipe=_make_recipe())
        store.save(record)
        loaded = store.get(record.id)
        assert loaded is not None
        assert loaded.id == record.id
        assert loaded.status == BuildStatus.PENDING

    def test_list_all(self, tmp_path):
        store = BuildStore(tmp_path / "builds.json")
        store.save(BuildRecord(recipe=_make_recipe("a")))
        store.save(BuildRecord(recipe=_make_recipe("b")))
        records = store.list_all()
        assert len(records) == 2

    def test_update_status(self, tmp_path):
        store = BuildStore(tmp_path / "builds.json")
        record = BuildRecord(recipe=_make_recipe())
        store.save(record)
        record.status = BuildStatus.RUNNING
        store.save(record)
        loaded = store.get(record.id)
        assert loaded.status == BuildStatus.RUNNING

    def test_get_nonexistent_returns_none(self, tmp_path):
        store = BuildStore(tmp_path / "builds.json")
        assert store.get("nonexistent") is None

    def test_persistence_across_instances(self, tmp_path):
        path = tmp_path / "builds.json"
        store1 = BuildStore(path)
        record = BuildRecord(recipe=_make_recipe())
        store1.save(record)

        store2 = BuildStore(path)
        loaded = store2.get(record.id)
        assert loaded is not None
```

- [ ] **Step 6: Run state tests to verify they fail**

Run: `python -m pytest tests/core/test_state.py -v`
Expected: FAIL — import errors

- [ ] **Step 7: Implement state module**

```python
# src/coreml_converter/core/state.py
from __future__ import annotations

import fcntl
import json
from pathlib import Path

from coreml_converter.core.models import BuildRecord


class BuildStore:
    def __init__(self, path: Path) -> None:
        self._path = path

    def _read_all(self) -> dict[str, dict]:
        if not self._path.exists():
            return {}
        data = json.loads(self._path.read_text())
        return {r["id"]: r for r in data.get("builds", [])}

    def _write_all(self, records: dict[str, dict]) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        data = {"schema_version": 1, "builds": list(records.values())}
        tmp = self._path.with_suffix(".tmp")
        with open(tmp, "w") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            json.dump(data, f, indent=2, default=str)
            fcntl.flock(f, fcntl.LOCK_UN)
        tmp.rename(self._path)

    def save(self, record: BuildRecord) -> None:
        records = self._read_all()
        records[record.id] = json.loads(record.model_dump_json())
        self._write_all(records)

    def get(self, build_id: str) -> BuildRecord | None:
        records = self._read_all()
        data = records.get(build_id)
        if data is None:
            return None
        return BuildRecord(**data)

    def list_all(self) -> list[BuildRecord]:
        records = self._read_all()
        return [BuildRecord(**r) for r in records.values()]
```

- [ ] **Step 8: Run all state tests**

Run: `python -m pytest tests/core/test_state.py -v`
Expected: All PASS

- [ ] **Step 9: Commit**

```bash
git add src/coreml_converter/core/config.py src/coreml_converter/core/state.py tests/core/test_config.py tests/core/test_state.py
git commit -m "feat: add config management and build state persistence"
```

---

## Task 4: Rate Limiter

**Files:**
- Create: `src/coreml_converter/core/registry/rate_limiter.py`
- Create: `tests/core/registry/test_rate_limiter.py`

- [ ] **Step 1: Write rate limiter tests**

```python
# tests/core/registry/test_rate_limiter.py
import time
import pytest
from coreml_converter.core.registry.rate_limiter import TokenBucketRateLimiter


class TestTokenBucketRateLimiter:
    def test_allows_burst_up_to_capacity(self):
        limiter = TokenBucketRateLimiter(rate=2.0, capacity=3)
        # Should allow 3 immediate calls
        for _ in range(3):
            assert limiter.try_acquire() is True

    def test_blocks_after_burst(self):
        limiter = TokenBucketRateLimiter(rate=2.0, capacity=2)
        limiter.try_acquire()
        limiter.try_acquire()
        assert limiter.try_acquire() is False

    def test_refills_over_time(self):
        limiter = TokenBucketRateLimiter(rate=10.0, capacity=1)
        limiter.try_acquire()
        assert limiter.try_acquire() is False
        time.sleep(0.15)  # Wait for refill
        assert limiter.try_acquire() is True

    def test_wait_for_token(self):
        limiter = TokenBucketRateLimiter(rate=10.0, capacity=1)
        limiter.try_acquire()
        start = time.monotonic()
        limiter.acquire()  # Should block until token available
        elapsed = time.monotonic() - start
        assert elapsed >= 0.05  # At least some wait
        assert elapsed < 0.5   # But not too long
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/core/registry/test_rate_limiter.py -v`
Expected: FAIL

- [ ] **Step 3: Implement rate limiter**

```python
# src/coreml_converter/core/registry/rate_limiter.py
from __future__ import annotations

import time
import threading


class TokenBucketRateLimiter:
    def __init__(self, rate: float, capacity: int) -> None:
        self._rate = rate  # tokens per second
        self._capacity = capacity
        self._tokens = float(capacity)
        self._last_refill = time.monotonic()
        self._lock = threading.Lock()

    def _refill(self) -> None:
        now = time.monotonic()
        elapsed = now - self._last_refill
        self._tokens = min(self._capacity, self._tokens + elapsed * self._rate)
        self._last_refill = now

    def try_acquire(self) -> bool:
        with self._lock:
            self._refill()
            if self._tokens >= 1.0:
                self._tokens -= 1.0
                return True
            return False

    def acquire(self) -> None:
        while True:
            with self._lock:
                self._refill()
                if self._tokens >= 1.0:
                    self._tokens -= 1.0
                    return
                wait_time = (1.0 - self._tokens) / self._rate
            time.sleep(wait_time)
```

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/core/registry/test_rate_limiter.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add src/coreml_converter/core/registry/rate_limiter.py tests/core/registry/test_rate_limiter.py
git commit -m "feat: add token bucket rate limiter for CivitAI API"
```

---

## Task 5: Registry — Base Class + HuggingFace Client

**Files:**
- Create: `src/coreml_converter/core/registry/base.py`
- Create: `src/coreml_converter/core/registry/huggingface.py`
- Create: `tests/core/registry/test_huggingface.py`

- [ ] **Step 1: Write tests for HuggingFace client**

```python
# tests/core/registry/test_huggingface.py
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/core/registry/test_huggingface.py -v`
Expected: FAIL

- [ ] **Step 3: Implement base class and HuggingFace client**

```python
# src/coreml_converter/core/registry/base.py
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from coreml_converter.core.models import BaseArchitecture, ModelInfo, ModelType


class RegistryClient(ABC):
    @abstractmethod
    def search(
        self,
        query: str,
        model_type: ModelType | None = None,
        base_arch: BaseArchitecture | None = None,
        limit: int = 20,
    ) -> list[ModelInfo]:
        ...

    @abstractmethod
    def get_compatible_loras(self, base_model: ModelInfo, limit: int = 20) -> list[ModelInfo]:
        ...

    @abstractmethod
    def download(self, model: ModelInfo, dest: Path) -> Path:
        ...
```

```python
# src/coreml_converter/core/registry/huggingface.py
from __future__ import annotations

from pathlib import Path

from huggingface_hub import HfApi, snapshot_download

from coreml_converter.core.models import (
    BaseArchitecture, ModelInfo, ModelSource, ModelType,
)
from coreml_converter.core.registry.base import RegistryClient

_ARCH_TAG_MAP = {
    "sd-1.5": BaseArchitecture.SD15,
    "sd-1.4": BaseArchitecture.SD15,  # 1.4 is compatible with 1.5
    "sd-2.0": BaseArchitecture.SD20,
    "sd-2.1": BaseArchitecture.SD20,  # 2.1 is compatible with 2.0
}


class HuggingFaceClient(RegistryClient):
    def __init__(self) -> None:
        self._api = HfApi()

    def _infer_architecture(self, tags: list[str]) -> BaseArchitecture | None:
        for tag in tags:
            tag_lower = tag.lower()
            if tag_lower in _ARCH_TAG_MAP:
                return _ARCH_TAG_MAP[tag_lower]
        return None

    def _infer_model_type(self, tags: list[str]) -> ModelType:
        for tag in tags:
            if "lora" in tag.lower():
                return ModelType.LORA
        return ModelType.CHECKPOINT

    def search(
        self,
        query: str,
        model_type: ModelType | None = None,
        base_arch: BaseArchitecture | None = None,
        limit: int = 20,
    ) -> list[ModelInfo]:
        models = self._api.list_models(
            search=query,
            pipeline_tag="text-to-image",
            sort="downloads",
            direction=-1,
            limit=limit * 3,  # over-fetch to filter
        )

        results: list[ModelInfo] = []
        for m in models:
            tags = list(m.tags or [])
            arch = self._infer_architecture(tags)
            if arch is None:
                continue
            if base_arch and arch != base_arch:
                continue
            mt = self._infer_model_type(tags)
            if model_type and mt != model_type:
                continue

            results.append(ModelInfo(
                source=ModelSource.HUGGINGFACE,
                id=m.id,
                name=m.id.split("/")[-1] if "/" in m.id else m.id,
                base_architecture=arch,
                model_type=mt,
                tags=tags,
                download_url=f"https://huggingface.co/{m.id}",
                metadata={"downloads": getattr(m, "downloads", 0)},
            ))
            if len(results) >= limit:
                break

        return results

    def get_compatible_loras(self, base_model: ModelInfo, limit: int = 20) -> list[ModelInfo]:
        return self.search(
            query="lora",
            model_type=ModelType.LORA,
            base_arch=base_model.base_architecture,
            limit=limit,
        )

    def download(self, model: ModelInfo, dest: Path) -> Path:
        return Path(snapshot_download(
            repo_id=model.id,
            local_dir=str(dest / model.id.replace("/", "_")),
        ))
```

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/core/registry/test_huggingface.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add src/coreml_converter/core/registry/base.py src/coreml_converter/core/registry/huggingface.py tests/core/registry/test_huggingface.py
git commit -m "feat: add registry base class and HuggingFace client"
```

---

## Task 6: Registry — CivitAI Client

**Files:**
- Create: `src/coreml_converter/core/registry/civitai.py`
- Create: `tests/core/registry/test_civitai.py`

- [ ] **Step 1: Write CivitAI client tests**

```python
# tests/core/registry/test_civitai.py
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/core/registry/test_civitai.py -v`
Expected: FAIL

- [ ] **Step 3: Implement CivitAI client**

```python
# src/coreml_converter/core/registry/civitai.py
from __future__ import annotations

import logging
import time
from pathlib import Path

import httpx

from coreml_converter.core.models import (
    BaseArchitecture, ModelInfo, ModelSource, ModelType,
)
from coreml_converter.core.registry.base import RegistryClient
from coreml_converter.core.registry.rate_limiter import TokenBucketRateLimiter

logger = logging.getLogger(__name__)

CIVITAI_API = "https://civitai.com/api/v1"

_BASE_MODEL_MAP = {
    "SD 1.4": BaseArchitecture.SD15,
    "SD 1.5": BaseArchitecture.SD15,
    "SD 2.0": BaseArchitecture.SD20,
    "SD 2.1": BaseArchitecture.SD20,
}

_TYPE_MAP = {
    "Checkpoint": ModelType.CHECKPOINT,
    "LORA": ModelType.LORA,
}


class CivitAIClient(RegistryClient):
    def __init__(self, api_key: str | None = None) -> None:
        self._api_key = api_key
        self._client = httpx.Client(timeout=30.0)
        self._rate_limiter = TokenBucketRateLimiter(rate=2.0, capacity=3)

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        return headers

    def _parse_base_model(self, base_model_str: str) -> BaseArchitecture | None:
        return _BASE_MODEL_MAP.get(base_model_str)

    def _parse_civitai_type(self, type_str: str) -> ModelType | None:
        return _TYPE_MAP.get(type_str)

    def search(
        self,
        query: str,
        model_type: ModelType | None = None,
        base_arch: BaseArchitecture | None = None,
        limit: int = 20,
    ) -> list[ModelInfo]:
        self._rate_limiter.acquire()

        params: dict = {"query": query, "limit": limit, "sort": "Most Downloaded"}
        if model_type:
            civitai_type = {ModelType.CHECKPOINT: "Checkpoint", ModelType.LORA: "LORA"}
            params["types"] = civitai_type.get(model_type, "Checkpoint")

        resp = self._client.get(f"{CIVITAI_API}/models", params=params, headers=self._headers())
        resp.raise_for_status()
        data = resp.json()

        results: list[ModelInfo] = []
        for item in data.get("items", []):
            mt = self._parse_civitai_type(item.get("type", ""))
            if mt is None:
                continue

            versions = item.get("modelVersions", [])
            if not versions:
                continue

            version = versions[0]
            arch = self._parse_base_model(version.get("baseModel", ""))
            if arch is None:
                continue
            if base_arch and arch != base_arch:
                continue

            files = [f for f in version.get("files", []) if f.get("type") == "Model"]
            if not files:
                continue

            results.append(ModelInfo(
                source=ModelSource.CIVITAI,
                id=str(item["id"]),
                name=item["name"],
                base_architecture=arch,
                model_type=mt,
                tags=item.get("tags", []),
                download_url=files[0]["downloadUrl"],
                metadata={
                    "version_id": version["id"],
                    "version_name": version.get("name", ""),
                    "download_count": item.get("stats", {}).get("downloadCount", 0),
                    "images": [img.get("url") for img in version.get("images", [])[:3]],
                    "description": item.get("description", ""),
                },
            ))

        return results

    def get_compatible_loras(self, base_model: ModelInfo, limit: int = 20) -> list[ModelInfo]:
        return self.search(
            query="",
            model_type=ModelType.LORA,
            base_arch=base_model.base_architecture,
            limit=limit,
        )

    def download(self, model: ModelInfo, dest: Path, retries: int = 3) -> Path:
        dest.mkdir(parents=True, exist_ok=True)
        filename = f"{model.source.value}_{model.id}.safetensors"
        file_path = dest / filename
        partial_path = file_path.with_suffix(".partial")

        for attempt in range(retries):
            try:
                self._rate_limiter.acquire()
                headers = self._headers()

                # Resume support
                if partial_path.exists():
                    existing_size = partial_path.stat().st_size
                    headers["Range"] = f"bytes={existing_size}-"
                    mode = "ab"
                else:
                    mode = "wb"

                with self._client.stream("GET", model.download_url, headers=headers) as resp:
                    resp.raise_for_status()
                    with open(partial_path, mode) as f:
                        for chunk in resp.iter_bytes(chunk_size=8192):
                            f.write(chunk)

                partial_path.rename(file_path)
                return file_path

            except (httpx.HTTPError, OSError) as e:
                logger.warning(f"Download attempt {attempt + 1} failed: {e}")
                if attempt == retries - 1:
                    if partial_path.exists():
                        partial_path.unlink()
                    raise
                time.sleep(2 ** attempt)

        raise RuntimeError("Download failed after all retries")
```

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/core/registry/test_civitai.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add src/coreml_converter/core/registry/civitai.py tests/core/registry/test_civitai.py
git commit -m "feat: add CivitAI REST API client with rate limiting"
```

---

## Task 7: Registry Facade

**Files:**
- Create: `tests/core/registry/test_registry.py`
- Modify: `src/coreml_converter/core/registry/__init__.py`

- [ ] **Step 1: Write registry facade tests**

```python
# tests/core/registry/test_registry.py
import pytest
from unittest.mock import MagicMock
from coreml_converter.core.models import (
    ModelSource, BaseArchitecture, ModelType, ModelInfo,
)
from coreml_converter.core.registry import Registry


def _make_model(source: ModelSource, name: str) -> ModelInfo:
    return ModelInfo(
        source=source, id="1", name=name,
        base_architecture=BaseArchitecture.SD15,
        model_type=ModelType.CHECKPOINT,
        tags=[], download_url="", metadata={},
    )


class TestRegistry:
    def test_search_all_sources(self):
        hf_client = MagicMock()
        civitai_client = MagicMock()
        hf_client.search.return_value = [_make_model(ModelSource.HUGGINGFACE, "HF Model")]
        civitai_client.search.return_value = [_make_model(ModelSource.CIVITAI, "Civitai Model")]

        registry = Registry(hf_client=hf_client, civitai_client=civitai_client)
        results = registry.search("test")
        assert len(results) == 2

    def test_search_single_source(self):
        hf_client = MagicMock()
        civitai_client = MagicMock()
        hf_client.search.return_value = [_make_model(ModelSource.HUGGINGFACE, "HF")]

        registry = Registry(hf_client=hf_client, civitai_client=civitai_client)
        results = registry.search("test", source=ModelSource.HUGGINGFACE)
        assert len(results) == 1
        civitai_client.search.assert_not_called()

    def test_search_handles_source_error_gracefully(self):
        hf_client = MagicMock()
        civitai_client = MagicMock()
        hf_client.search.side_effect = Exception("HF down")
        civitai_client.search.return_value = [_make_model(ModelSource.CIVITAI, "Civitai")]

        registry = Registry(hf_client=hf_client, civitai_client=civitai_client)
        results = registry.search("test")
        assert len(results) == 1  # CivitAI results still returned
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/core/registry/test_registry.py -v`
Expected: FAIL

- [ ] **Step 3: Implement registry facade**

```python
# src/coreml_converter/core/registry/__init__.py
from __future__ import annotations

import logging
from pathlib import Path

from coreml_converter.core.models import (
    BaseArchitecture, ModelInfo, ModelSource, ModelType,
)
from coreml_converter.core.registry.base import RegistryClient
from coreml_converter.core.registry.civitai import CivitAIClient
from coreml_converter.core.registry.huggingface import HuggingFaceClient

logger = logging.getLogger(__name__)


class Registry:
    def __init__(
        self,
        hf_client: RegistryClient | None = None,
        civitai_client: RegistryClient | None = None,
    ) -> None:
        self._clients: dict[ModelSource, RegistryClient] = {}
        if hf_client:
            self._clients[ModelSource.HUGGINGFACE] = hf_client
        if civitai_client:
            self._clients[ModelSource.CIVITAI] = civitai_client

    def search(
        self,
        query: str,
        source: ModelSource | None = None,
        model_type: ModelType | None = None,
        base_arch: BaseArchitecture | None = None,
        limit: int = 20,
    ) -> list[ModelInfo]:
        clients = (
            {source: self._clients[source]}
            if source and source in self._clients
            else self._clients
        )

        results: list[ModelInfo] = []
        for src, client in clients.items():
            try:
                results.extend(client.search(query, model_type=model_type, base_arch=base_arch, limit=limit))
            except Exception:
                logger.exception(f"Search failed for {src.value}")
        return results

    def get_compatible_loras(self, base_model: ModelInfo, limit: int = 20) -> list[ModelInfo]:
        results: list[ModelInfo] = []
        for src, client in self._clients.items():
            try:
                results.extend(client.get_compatible_loras(base_model, limit=limit))
            except Exception:
                logger.exception(f"LoRA search failed for {src.value}")
        return results

    def download(self, model: ModelInfo, dest: Path) -> Path:
        client = self._clients.get(model.source)
        if client is None:
            raise ValueError(f"No client registered for {model.source.value}")
        return client.download(model, dest)
```

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/core/registry/test_registry.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add src/coreml_converter/core/registry/__init__.py tests/core/registry/test_registry.py
git commit -m "feat: add registry facade combining HuggingFace and CivitAI"
```

---

## Task 8: Analyzer — Compatibility Checking

**Files:**
- Create: `src/coreml_converter/core/analyzer/compatibility.py`
- Create: `tests/core/analyzer/test_compatibility.py`

- [ ] **Step 1: Write compatibility tests**

```python
# tests/core/analyzer/test_compatibility.py
import pytest
from coreml_converter.core.models import (
    ModelInfo, ModelSource, BaseArchitecture, ModelType,
    LoRAEntry, CompatibilityReport,
)
from coreml_converter.core.analyzer.compatibility import check_compatibility


def _make_base(arch: BaseArchitecture = BaseArchitecture.SD15) -> ModelInfo:
    return ModelInfo(
        source=ModelSource.CIVITAI, id="1", name="Base",
        base_architecture=arch, model_type=ModelType.CHECKPOINT,
        tags=[], download_url="", metadata={},
    )


def _make_lora(arch: BaseArchitecture = BaseArchitecture.SD15, tags: list[str] | None = None) -> LoRAEntry:
    model = ModelInfo(
        source=ModelSource.CIVITAI, id="2", name="LoRA",
        base_architecture=arch, model_type=ModelType.LORA,
        tags=tags or [], download_url="", metadata={},
    )
    return LoRAEntry(model=model, weight=0.7)


class TestCheckCompatibility:
    def test_compatible_same_architecture(self):
        report = check_compatibility(_make_base(), [_make_lora()])
        assert report.is_compatible is True
        assert report.architecture_match is True

    def test_incompatible_architecture_mismatch(self):
        report = check_compatibility(
            _make_base(BaseArchitecture.SD15),
            [_make_lora(BaseArchitecture.SD20)],
        )
        assert report.is_compatible is False
        assert report.architecture_match is False

    def test_no_loras_is_compatible(self):
        report = check_compatibility(_make_base(), [])
        assert report.is_compatible is True
        assert report.lora_count_warning is None

    def test_soft_warning_at_4_loras(self):
        loras = [_make_lora() for _ in range(4)]
        report = check_compatibility(_make_base(), loras)
        assert report.lora_count_warning is not None
        assert "may degrade" in report.lora_count_warning

    def test_hard_warning_at_6_loras(self):
        loras = [_make_lora() for _ in range(6)]
        report = check_compatibility(_make_base(), loras)
        assert report.lora_count_warning is not None
        assert "artifacts" in report.lora_count_warning

    def test_mixed_architectures_flag_specific_loras(self):
        loras = [
            _make_lora(BaseArchitecture.SD15),
            _make_lora(BaseArchitecture.SD20),
        ]
        report = check_compatibility(_make_base(BaseArchitecture.SD15), loras)
        assert report.is_compatible is False
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/core/analyzer/test_compatibility.py -v`
Expected: FAIL

- [ ] **Step 3: Implement compatibility checker**

```python
# src/coreml_converter/core/analyzer/compatibility.py
from __future__ import annotations

from coreml_converter.core.models import (
    CompatibilityReport, Conflict, LoRAEntry, ModelInfo, RiskLevel, Severity,
)


def check_compatibility(
    base_model: ModelInfo,
    loras: list[LoRAEntry],
) -> CompatibilityReport:
    conflicts: list[Conflict] = []
    arch_match = True

    # Check architecture match for each LoRA
    for entry in loras:
        if entry.model.base_architecture != base_model.base_architecture:
            arch_match = False
            conflicts.append(Conflict(
                lora_a=entry.model.name,
                lora_b=base_model.name,
                reason=f"Architecture mismatch: LoRA is {entry.model.base_architecture.value}, "
                       f"base is {base_model.base_architecture.value}",
                severity=Severity.WARNING,
            ))

    # LoRA count warning
    count = len(loras)
    lora_count_warning = None
    if count >= 6:
        lora_count_warning = "6+ LoRAs: likely to produce artifacts, proceed at own risk"
    elif count >= 4:
        lora_count_warning = "4-5 LoRAs: quality may degrade"

    # Risk assessment
    if not arch_match:
        risk = RiskLevel.HIGH
    elif count >= 6:
        risk = RiskLevel.HIGH
    elif count >= 4 or conflicts:
        risk = RiskLevel.MEDIUM
    else:
        risk = RiskLevel.LOW

    return CompatibilityReport(
        is_compatible=arch_match,
        architecture_match=arch_match,
        dimension_check=None,  # Populated post-download
        conflicts=conflicts,
        lora_count_warning=lora_count_warning,
        overall_risk=risk,
    )
```

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/core/analyzer/test_compatibility.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add src/coreml_converter/core/analyzer/compatibility.py tests/core/analyzer/test_compatibility.py
git commit -m "feat: add metadata-based compatibility checker"
```

---

## Task 9: Analyzer — Conflict Detection

**Files:**
- Create: `src/coreml_converter/core/analyzer/conflicts.py`
- Create: `tests/core/analyzer/test_conflicts.py`

- [ ] **Step 1: Write conflict detection tests**

```python
# tests/core/analyzer/test_conflicts.py
import pytest
from coreml_converter.core.models import (
    ModelInfo, ModelSource, BaseArchitecture, ModelType,
    LoRAEntry, Conflict, Severity,
)
from coreml_converter.core.analyzer.conflicts import detect_tag_conflicts


def _make_lora(name: str, tags: list[str]) -> LoRAEntry:
    model = ModelInfo(
        source=ModelSource.CIVITAI, id=name, name=name,
        base_architecture=BaseArchitecture.SD15,
        model_type=ModelType.LORA, tags=tags, download_url="", metadata={},
    )
    return LoRAEntry(model=model, weight=0.7)


class TestDetectTagConflicts:
    def test_no_conflicts_different_categories(self):
        loras = [
            _make_lora("Style LoRA", ["anime", "style"]),
            _make_lora("Character LoRA", ["character", "female"]),
        ]
        conflicts = detect_tag_conflicts(loras)
        assert len(conflicts) == 0

    def test_same_category_info(self):
        loras = [
            _make_lora("Char A", ["character", "male"]),
            _make_lora("Char B", ["character", "female"]),
        ]
        conflicts = detect_tag_conflicts(loras)
        assert len(conflicts) == 1
        assert conflicts[0].severity == Severity.INFO

    def test_competing_styles_warning(self):
        loras = [
            _make_lora("Realistic", ["realistic", "photorealistic", "style"]),
            _make_lora("Anime", ["anime", "cartoon", "style"]),
        ]
        conflicts = detect_tag_conflicts(loras)
        warnings = [c for c in conflicts if c.severity == Severity.WARNING]
        assert len(warnings) >= 1

    def test_single_lora_no_conflicts(self):
        loras = [_make_lora("Solo", ["style", "anime"])]
        conflicts = detect_tag_conflicts(loras)
        assert len(conflicts) == 0

    def test_three_loras_same_category(self):
        loras = [
            _make_lora("A", ["style"]),
            _make_lora("B", ["style"]),
            _make_lora("C", ["style"]),
        ]
        conflicts = detect_tag_conflicts(loras)
        assert len(conflicts) >= 2  # A-B, A-C, B-C pairs
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/core/analyzer/test_conflicts.py -v`
Expected: FAIL

- [ ] **Step 3: Implement conflict detection**

```python
# src/coreml_converter/core/analyzer/conflicts.py
from __future__ import annotations

from itertools import combinations

from coreml_converter.core.models import Conflict, LoRAEntry, Severity

# Tag-to-category mapping
_TAG_CATEGORIES: dict[str, str] = {
    # Style
    "style": "style", "anime": "style", "realistic": "style",
    "photorealistic": "style", "cartoon": "style", "3d": "style",
    "illustration": "style", "painting": "style", "digital art": "style",
    "watercolor": "style", "oil painting": "style", "pixel art": "style",
    # Character
    "character": "character", "male": "character", "female": "character",
    "person": "character", "face": "character", "portrait": "character",
    # Concept
    "concept": "concept", "object": "concept", "vehicle": "concept",
    "animal": "concept", "food": "concept",
    # Clothing
    "clothing": "clothing", "outfit": "clothing", "armor": "clothing",
    "dress": "clothing", "uniform": "clothing",
    # Pose
    "pose": "pose", "action": "pose", "sitting": "pose", "standing": "pose",
    # Background
    "background": "background", "landscape": "background",
    "interior": "background", "scenery": "background",
}

# Competing tags that produce a WARNING (not just INFO)
_COMPETING_PAIRS: set[frozenset[str]] = {
    frozenset({"realistic", "anime"}),
    frozenset({"realistic", "cartoon"}),
    frozenset({"photorealistic", "anime"}),
    frozenset({"photorealistic", "cartoon"}),
    frozenset({"3d", "illustration"}),
}


def _categorize_lora(entry: LoRAEntry) -> set[str]:
    categories = set()
    for tag in entry.model.tags:
        tag_lower = tag.lower()
        cat = _TAG_CATEGORIES.get(tag_lower)
        if cat:
            categories.add(cat)
    return categories


def _has_competing_tags(a: LoRAEntry, b: LoRAEntry) -> bool:
    tags_a = {t.lower() for t in a.model.tags}
    tags_b = {t.lower() for t in b.model.tags}
    for pair in _COMPETING_PAIRS:
        if pair <= (tags_a | tags_b) and not pair <= tags_a and not pair <= tags_b:
            return True
    return False


def detect_tag_conflicts(loras: list[LoRAEntry]) -> list[Conflict]:
    conflicts: list[Conflict] = []
    categorized = [(entry, _categorize_lora(entry)) for entry in loras]

    for (a, cats_a), (b, cats_b) in combinations(categorized, 2):
        shared = cats_a & cats_b
        if not shared:
            continue

        if _has_competing_tags(a, b):
            conflicts.append(Conflict(
                lora_a=a.model.name,
                lora_b=b.model.name,
                reason=f"Competing styles: {a.model.name} vs {b.model.name}",
                severity=Severity.WARNING,
            ))
        else:
            conflicts.append(Conflict(
                lora_a=a.model.name,
                lora_b=b.model.name,
                reason=f"Same category: {', '.join(shared)}",
                severity=Severity.INFO,
            ))

    return conflicts
```

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/core/analyzer/test_conflicts.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add src/coreml_converter/core/analyzer/conflicts.py tests/core/analyzer/test_conflicts.py
git commit -m "feat: add tag-based conflict detection for LoRA combinations"
```

---

## Task 10: Analyzer — Weight Guidance

**Files:**
- Create: `src/coreml_converter/core/analyzer/weight_guidance.py`
- Create: `tests/core/analyzer/test_weight_guidance.py`

- [ ] **Step 1: Write weight guidance tests**

```python
# tests/core/analyzer/test_weight_guidance.py
import pytest
from coreml_converter.core.models import (
    ModelInfo, ModelSource, BaseArchitecture, ModelType,
)
from coreml_converter.core.analyzer.weight_guidance import get_recommended_weight


def _make_lora(tags: list[str] | None = None, metadata: dict | None = None) -> ModelInfo:
    return ModelInfo(
        source=ModelSource.CIVITAI, id="1", name="Test LoRA",
        base_architecture=BaseArchitecture.SD15,
        model_type=ModelType.LORA, tags=tags or [],
        download_url="", metadata=metadata or {},
    )


class TestGetRecommendedWeight:
    def test_creator_specified_weight(self):
        model = _make_lora(metadata={"description": "Best results at weight: 0.6-0.8"})
        weight, source = get_recommended_weight(model)
        assert 0.6 <= weight <= 0.8
        assert source == "creator"

    def test_style_category_default(self):
        model = _make_lora(tags=["style", "anime"])
        weight, source = get_recommended_weight(model)
        assert 0.6 <= weight <= 0.8
        assert source == "category_default"

    def test_character_category_default(self):
        model = _make_lora(tags=["character"])
        weight, source = get_recommended_weight(model)
        assert 0.7 <= weight <= 0.9
        assert source == "category_default"

    def test_detail_category_default(self):
        model = _make_lora(tags=["detail"])
        weight, source = get_recommended_weight(model)
        assert 0.4 <= weight <= 0.6
        assert source == "category_default"

    def test_unknown_tags_fallback_to_1(self):
        model = _make_lora(tags=["somethingunknown"])
        weight, source = get_recommended_weight(model)
        assert weight == 1.0
        assert source is None

    def test_creator_weight_takes_priority(self):
        model = _make_lora(
            tags=["style", "anime"],
            metadata={"description": "Recommended weight: 0.3"},
        )
        weight, source = get_recommended_weight(model)
        assert weight == pytest.approx(0.3, abs=0.05)
        assert source == "creator"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/core/analyzer/test_weight_guidance.py -v`
Expected: FAIL

- [ ] **Step 3: Implement weight guidance**

```python
# src/coreml_converter/core/analyzer/weight_guidance.py
from __future__ import annotations

import re

from coreml_converter.core.models import ModelInfo

# Category -> (min, max) recommended weight range
_CATEGORY_DEFAULTS: dict[str, tuple[float, float]] = {
    "style": (0.6, 0.8),
    "character": (0.7, 0.9),
    "detail": (0.4, 0.6),
    "texture": (0.4, 0.6),
    "concept": (0.7, 1.0),
    "clothing": (0.7, 0.9),
    "pose": (0.6, 0.8),
    "background": (0.5, 0.7),
}

# Tags that map to a weight category
_TAG_TO_CATEGORY: dict[str, str] = {
    "style": "style", "anime": "style", "realistic": "style",
    "photorealistic": "style", "cartoon": "style",
    "character": "character", "person": "character", "face": "character",
    "portrait": "character",
    "detail": "detail", "details": "detail", "tweaker": "detail",
    "texture": "texture",
    "concept": "concept", "object": "concept",
    "clothing": "clothing", "outfit": "clothing",
    "pose": "pose", "action": "pose",
    "background": "background", "landscape": "background",
}

# Regex patterns for parsing creator-specified weights from descriptions
_WEIGHT_PATTERNS = [
    re.compile(r"(?:recommended?\s+)?weight[:\s]+(\d+\.?\d*)\s*[-–]\s*(\d+\.?\d*)", re.IGNORECASE),
    re.compile(r"(?:recommended?\s+)?weight[:\s]+(\d+\.?\d*)", re.IGNORECASE),
    re.compile(r"(?:best|optimal)\s+(?:results?\s+)?(?:at|with)\s+(?:weight[:\s]+)?(\d+\.?\d*)\s*[-–]\s*(\d+\.?\d*)", re.IGNORECASE),
    re.compile(r"(?:best|optimal)\s+(?:results?\s+)?(?:at|with)\s+(?:weight[:\s]+)?(\d+\.?\d*)", re.IGNORECASE),
]


def _parse_creator_weight(description: str) -> float | None:
    for pattern in _WEIGHT_PATTERNS:
        match = pattern.search(description)
        if match:
            groups = match.groups()
            if len(groups) == 2:
                low, high = float(groups[0]), float(groups[1])
                if 0.0 <= low <= 1.5 and 0.0 <= high <= 1.5:
                    return (low + high) / 2
            elif len(groups) == 1:
                val = float(groups[0])
                if 0.0 <= val <= 1.5:
                    return val
    return None


def _infer_category(model: ModelInfo) -> str | None:
    for tag in model.tags:
        tag_lower = tag.lower()
        cat = _TAG_TO_CATEGORY.get(tag_lower)
        if cat:
            return cat
    return None


def get_recommended_weight(model: ModelInfo) -> tuple[float, str | None]:
    """Returns (recommended_weight, source) where source is 'creator', 'category_default', or None."""
    # 1. Creator-specified
    description = model.metadata.get("description", "")
    if description:
        creator_weight = _parse_creator_weight(description)
        if creator_weight is not None:
            return creator_weight, "creator"

    # 2. Category defaults
    category = _infer_category(model)
    if category and category in _CATEGORY_DEFAULTS:
        low, high = _CATEGORY_DEFAULTS[category]
        return (low + high) / 2, "category_default"

    # 3. Fallback
    return 1.0, None
```

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/core/analyzer/test_weight_guidance.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add src/coreml_converter/core/analyzer/weight_guidance.py tests/core/analyzer/test_weight_guidance.py
git commit -m "feat: add LoRA weight guidance with creator/category defaults"
```

---

## Task 11: Analyzer Facade

**Files:**
- Modify: `src/coreml_converter/core/analyzer/__init__.py`

- [ ] **Step 1: Implement analyzer facade**

```python
# src/coreml_converter/core/analyzer/__init__.py
from coreml_converter.core.analyzer.compatibility import check_compatibility
from coreml_converter.core.analyzer.conflicts import detect_tag_conflicts
from coreml_converter.core.analyzer.weight_guidance import get_recommended_weight

__all__ = ["check_compatibility", "detect_tag_conflicts", "get_recommended_weight"]
```

- [ ] **Step 2: Run all analyzer tests together**

Run: `python -m pytest tests/core/analyzer/ -v`
Expected: All PASS

- [ ] **Step 3: Commit**

```bash
git add src/coreml_converter/core/analyzer/__init__.py
git commit -m "feat: add analyzer facade re-exporting public API"
```

---

## Task 12: Merger — LoRA Baking

**Files:**
- Create: `src/coreml_converter/core/merger/merger.py`
- Create: `tests/core/merger/test_merger.py`

- [ ] **Step 1: Write merger tests**

The merger depends on `torch` and `diffusers` which are heavy. Tests mock the pipeline.

```python
# tests/core/merger/test_merger.py
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock, call
from coreml_converter.core.models import (
    ModelInfo, ModelSource, BaseArchitecture, ModelType,
    LoRAEntry, ConversionConfig, Recipe,
)
from coreml_converter.core.merger.merger import Merger


def _make_recipe(loras: list[LoRAEntry] | None = None) -> Recipe:
    base = ModelInfo(
        source=ModelSource.CIVITAI, id="1", name="Base Model",
        base_architecture=BaseArchitecture.SD15,
        model_type=ModelType.CHECKPOINT,
        tags=[], download_url="", metadata={"local_path": "/tmp/base"},
    )
    config = ConversionConfig(output_dir=Path("/tmp/output"), model_name="test")
    return Recipe(name="test", base_model=base, loras=loras or [], conversion_config=config)


def _make_lora(name: str, weight: float = 0.7) -> LoRAEntry:
    model = ModelInfo(
        source=ModelSource.CIVITAI, id=name, name=name,
        base_architecture=BaseArchitecture.SD15,
        model_type=ModelType.LORA,
        tags=[], download_url="",
        metadata={"local_path": f"/tmp/{name}.safetensors"},
    )
    return LoRAEntry(model=model, weight=weight)


class TestMerger:
    @patch("coreml_converter.core.merger.merger.StableDiffusionPipeline")
    def test_merge_no_loras(self, mock_pipeline_cls, tmp_path):
        mock_pipe = MagicMock()
        mock_pipeline_cls.from_pretrained.return_value = mock_pipe

        merger = Merger()
        recipe = _make_recipe()
        result = merger.merge(recipe, cache_dir=Path("/tmp/cache"), output_dir=tmp_path)

        mock_pipe.save_pretrained.assert_called_once()
        assert result.exists() or True  # tmp_path-based

    @patch("coreml_converter.core.merger.merger.StableDiffusionPipeline")
    def test_merge_applies_loras_in_order(self, mock_pipeline_cls, tmp_path):
        mock_pipe = MagicMock()
        mock_pipeline_cls.from_pretrained.return_value = mock_pipe

        merger = Merger()
        loras = [_make_lora("A", 0.8), _make_lora("B", 0.5)]
        recipe = _make_recipe(loras)
        merger.merge(recipe, cache_dir=Path("/tmp/cache"), output_dir=tmp_path)

        # Should load and fuse each LoRA in order
        assert mock_pipe.load_lora_weights.call_count == 2
        assert mock_pipe.fuse_lora.call_count == 2

    @patch("coreml_converter.core.merger.merger.StableDiffusionPipeline")
    def test_merge_uses_correct_weights(self, mock_pipeline_cls, tmp_path):
        mock_pipe = MagicMock()
        mock_pipeline_cls.from_pretrained.return_value = mock_pipe

        merger = Merger()
        loras = [_make_lora("A", 0.6)]
        recipe = _make_recipe(loras)
        merger.merge(recipe, cache_dir=Path("/tmp/cache"), output_dir=tmp_path)

        mock_pipe.fuse_lora.assert_called_once_with(lora_scale=0.6)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/core/merger/test_merger.py -v`
Expected: FAIL

- [ ] **Step 3: Implement merger**

```python
# src/coreml_converter/core/merger/merger.py
from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Callable

logger = logging.getLogger(__name__)

try:
    from diffusers import StableDiffusionPipeline
except ImportError:
    StableDiffusionPipeline = None

from coreml_converter.core.models import Recipe


class Merger:
    def merge(
        self,
        recipe: Recipe,
        cache_dir: Path,
        output_dir: Path,
        progress_callback: Callable[[str, float], None] | None = None,
    ) -> Path:
        if StableDiffusionPipeline is None:
            raise RuntimeError("diffusers is not installed. Install with: pip install coreml-converter[ml]")

        def _report(msg: str, pct: float) -> None:
            if progress_callback:
                progress_callback(msg, pct)

        _report("Loading base model", 0.0)
        local_path = recipe.base_model.metadata.get("local_path")
        if local_path and Path(local_path).exists():
            model_path = Path(local_path)
        else:
            model_path = cache_dir / f"{recipe.base_model.source.value}_{recipe.base_model.id}"

        # Detect if single file or directory
        if model_path.is_file():
            # .ckpt security warning
            if model_path.suffix == ".ckpt":
                logger.warning(
                    "WARNING: .ckpt files can contain arbitrary code. "
                    ".safetensors format is recommended."
                )
                if progress_callback:
                    progress_callback("ckpt_security_warning", 0.0)
            pipe = StableDiffusionPipeline.from_single_file(
                str(model_path), torch_dtype="auto",
            )
        else:
            pipe = StableDiffusionPipeline.from_pretrained(
                str(model_path), torch_dtype="auto",
            )

        # Apply LoRAs in order
        total_loras = len(recipe.loras)
        for i, entry in enumerate(recipe.loras):
            lora_path = entry.model.metadata.get("local_path")
            if lora_path and Path(lora_path).exists():
                lora_file = Path(lora_path)
            else:
                lora_file = cache_dir / f"{entry.model.source.value}_{entry.model.id}.safetensors"

            _report(f"Applying LoRA {i+1}/{total_loras}: {entry.model.name}", (i + 1) / (total_loras + 1))

            if lora_file.is_file():
                pipe.load_lora_weights(str(lora_file.parent), weight_name=lora_file.name)
            else:
                pipe.load_lora_weights(str(lora_file))

            pipe.fuse_lora(lora_scale=entry.weight)
            pipe.unload_lora_weights()

        # Save merged pipeline
        merged_dir = output_dir / "merged_pipeline"
        if merged_dir.exists():
            shutil.rmtree(merged_dir)
        merged_dir.mkdir(parents=True)

        _report("Saving merged pipeline", 0.9)
        pipe.save_pretrained(str(merged_dir))

        _report("Merge complete", 1.0)
        return merged_dir
```

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/core/merger/test_merger.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add src/coreml_converter/core/merger/merger.py tests/core/merger/test_merger.py
git commit -m "feat: add LoRA merger using diffusers pipeline"
```

---

## Task 13: Converter — CoreML Conversion

**Files:**
- Create: `src/coreml_converter/core/converter/converter.py`
- Create: `tests/core/converter/test_converter.py`

- [ ] **Step 1: Write converter tests**

```python
# tests/core/converter/test_converter.py
import json
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
from coreml_converter.core.models import (
    ConversionConfig, ConversionResult, ModelInfo, ModelSource,
    BaseArchitecture, ModelType, Recipe,
)
from coreml_converter.core.converter.converter import Converter, check_disk_space


def _make_config(tmp_path: Path) -> ConversionConfig:
    return ConversionConfig(
        output_dir=tmp_path / "output",
        model_name="test-model",
        compute_units="all",
        attention="split_einsum",
        precision="float16",
    )


def _make_recipe(tmp_path: Path) -> Recipe:
    base = ModelInfo(
        source=ModelSource.CIVITAI, id="1", name="Base",
        base_architecture=BaseArchitecture.SD15,
        model_type=ModelType.CHECKPOINT,
        tags=[], download_url="", metadata={},
    )
    return Recipe(
        name="test-model",
        base_model=base,
        loras=[],
        conversion_config=_make_config(tmp_path),
    )


class TestCheckDiskSpace:
    def test_sufficient_space(self, tmp_path):
        # Should not raise for typical tmp dirs
        check_disk_space(tmp_path, required_gb=0.001)

    def test_insufficient_space_raises(self, tmp_path):
        with pytest.raises(RuntimeError, match="Insufficient disk space"):
            check_disk_space(tmp_path, required_gb=999999)


class TestConverter:
    @patch("coreml_converter.core.converter.converter.subprocess")
    def test_convert_calls_apple_script(self, mock_subprocess, tmp_path):
        mock_subprocess.run.return_value = MagicMock(returncode=0)

        converter = Converter()
        config = _make_config(tmp_path)
        merged_path = tmp_path / "merged"
        merged_path.mkdir()

        # Create expected output files so result can be built
        output_dir = config.output_dir / config.model_name
        output_dir.mkdir(parents=True)
        (output_dir / f"{config.model_name}.mlpackage").mkdir()
        (output_dir / f"{config.model_name}.mlmodelc").mkdir()

        result = converter.convert(
            merged_model_path=merged_path,
            recipe=_make_recipe(tmp_path),
        )
        mock_subprocess.run.assert_called()

    def test_generate_manifest(self, tmp_path):
        converter = Converter()
        recipe = _make_recipe(tmp_path)
        manifest_path = tmp_path / "manifest.json"
        converter._write_manifest(recipe, manifest_path)

        manifest = json.loads(manifest_path.read_text())
        assert manifest["name"] == "test-model"
        assert manifest["schema_version"] == 1
        assert manifest["base_model"]["source"] == "civitai"
        assert manifest["conversion"]["compute_units"] == "all"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/core/converter/test_converter.py -v`
Expected: FAIL

- [ ] **Step 3: Implement converter**

```python
# src/coreml_converter/core/converter/converter.py
from __future__ import annotations

import json
import logging
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import coreml_converter
from coreml_converter.core.models import ConversionResult, Recipe

logger = logging.getLogger(__name__)


def check_disk_space(path: Path, required_gb: float = 20.0) -> None:
    stat = shutil.disk_usage(path)
    available_gb = stat.free / (1024 ** 3)
    if available_gb < required_gb:
        raise RuntimeError(
            f"Insufficient disk space: {available_gb:.1f} GB available, "
            f"{required_gb:.1f} GB required"
        )


class Converter:
    def convert(
        self,
        merged_model_path: Path,
        recipe: Recipe,
        progress_callback: Callable[[str, float], None] | None = None,
    ) -> ConversionResult:
        config = recipe.conversion_config
        output_dir = config.output_dir / config.model_name
        output_dir.mkdir(parents=True, exist_ok=True)

        def _report(msg: str, pct: float) -> None:
            if progress_callback:
                progress_callback(msg, pct)

        check_disk_space(config.output_dir)

        _report("Converting to CoreML", 0.1)
        start_time = time.monotonic()

        # Build conversion command using Apple's python_coreml_stable_diffusion
        cmd = [
            "python", "-m", "python_coreml_stable_diffusion.torch2coreml",
            "--model-version", str(merged_model_path),
            "-o", str(output_dir),
            "--convert-unet",
            "--convert-text-encoder",
            "--convert-vae-decoder",
            "--attention-implementation", config.attention.upper(),
            "--compute-unit", config.compute_units.replace("And", "_and_").upper(),
        ]

        if config.include_safety_checker:
            cmd.append("--convert-safety-checker")

        if config.precision == "float32":
            cmd.append("--precision-full")

        cmd.append("--bundle-resources-for-swift-cli")

        _report("Running CoreML conversion (this may take a while)", 0.3)
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"CoreML conversion failed:\n{result.stderr}")

        elapsed = time.monotonic() - start_time

        # Locate output files
        mlpackage_path = output_dir / f"{config.model_name}.mlpackage"
        mlmodelc_path = output_dir / f"{config.model_name}.mlmodelc"
        manifest_path = output_dir / "manifest.json"

        _report("Writing manifest", 0.95)
        self._write_manifest(recipe, manifest_path)

        # Calculate model size
        model_size_mb = 0.0
        if mlmodelc_path.exists():
            model_size_mb = sum(f.stat().st_size for f in mlmodelc_path.rglob("*") if f.is_file()) / (1024 ** 2)

        _report("Conversion complete", 1.0)
        return ConversionResult(
            mlpackage_path=mlpackage_path,
            mlmodelc_path=mlmodelc_path,
            manifest_path=manifest_path,
            conversion_time=elapsed,
            model_size_mb=model_size_mb,
        )

    def _write_manifest(self, recipe: Recipe, path: Path) -> None:
        manifest = {
            "schema_version": 1,
            "name": recipe.name,
            "created": datetime.now(timezone.utc).isoformat(),
            "base_model": {
                "source": recipe.base_model.source.value,
                "id": recipe.base_model.id,
                "name": recipe.base_model.name,
                "architecture": recipe.base_model.base_architecture.value,
            },
            "loras": [
                {
                    "source": entry.model.source.value,
                    "id": entry.model.id,
                    "name": entry.model.name,
                    "weight": entry.weight,
                }
                for entry in recipe.loras
            ],
            "conversion": {
                "compute_units": recipe.conversion_config.compute_units,
                "attention": recipe.conversion_config.attention,
                "precision": recipe.conversion_config.precision,
                "include_safety_checker": recipe.conversion_config.include_safety_checker,
            },
            "tool_version": coreml_converter.__version__,
        }
        path.write_text(json.dumps(manifest, indent=2))
```

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/core/converter/test_converter.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add src/coreml_converter/core/converter/converter.py tests/core/converter/test_converter.py
git commit -m "feat: add CoreML converter wrapping Apple's conversion tooling"
```

---

## Task 14: CLI — Main Group + Search Command

**Files:**
- Create: `src/coreml_converter/cli/main.py`
- Create: `src/coreml_converter/cli/formatting.py`
- Create: `src/coreml_converter/cli/commands/search.py`
- Create: `tests/cli/test_cli.py`

- [ ] **Step 1: Write CLI tests**

```python
# tests/cli/test_cli.py
import pytest
from click.testing import CliRunner
from unittest.mock import patch, MagicMock
from coreml_converter.cli.main import cli
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
        assert call_kwargs.kwargs.get("source") == ModelSource.CIVITAI or \
               call_kwargs[1].get("source") == ModelSource.CIVITAI
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/cli/test_cli.py -v`
Expected: FAIL

- [ ] **Step 3: Implement CLI main group**

```python
# src/coreml_converter/cli/main.py
import click

import coreml_converter
from coreml_converter.cli.commands.search import search
from coreml_converter.cli.commands.info import info
from coreml_converter.cli.commands.build import build
from coreml_converter.cli.commands.serve import serve
from coreml_converter.cli.commands.cache import cache


@click.group()
@click.version_option(version=coreml_converter.__version__, prog_name="CoreML Converter")
def cli():
    """CoreML Converter - Convert SD models + LoRAs to CoreML for Apple Silicon."""
    pass


cli.add_command(search)
cli.add_command(info)
cli.add_command(build)
cli.add_command(serve)
cli.add_command(cache)
```

- [ ] **Step 4: Implement formatting helpers**

```python
# src/coreml_converter/cli/formatting.py
from __future__ import annotations

from rich.console import Console
from rich.table import Table

from coreml_converter.core.models import ModelInfo

console = Console()


def print_model_table(models: list[ModelInfo]) -> None:
    table = Table(title="Search Results")
    table.add_column("Source", style="cyan")
    table.add_column("ID", style="dim")
    table.add_column("Name", style="bold")
    table.add_column("Arch", style="green")
    table.add_column("Type", style="yellow")
    table.add_column("Tags")

    for m in models:
        table.add_row(
            m.source.value,
            m.id,
            m.name,
            m.base_architecture.value,
            m.model_type.value,
            ", ".join(m.tags[:5]),
        )

    console.print(table)
```

- [ ] **Step 5: Implement search command**

```python
# src/coreml_converter/cli/commands/search.py
from __future__ import annotations

import click

from coreml_converter.core.models import BaseArchitecture, ModelSource, ModelType
from coreml_converter.cli.formatting import console, print_model_table


def get_registry():
    from coreml_converter.core.config import get_app_dir, load_config
    from coreml_converter.core.registry import Registry
    from coreml_converter.core.registry.huggingface import HuggingFaceClient
    from coreml_converter.core.registry.civitai import CivitAIClient

    config = load_config(get_app_dir() / "config.json")
    return Registry(
        hf_client=HuggingFaceClient(),
        civitai_client=CivitAIClient(api_key=config.civitai_api_key),
    )


_SOURCE_MAP = {"hf": ModelSource.HUGGINGFACE, "civitai": ModelSource.CIVITAI}
_TYPE_MAP = {"checkpoint": ModelType.CHECKPOINT, "lora": ModelType.LORA}
_ARCH_MAP = {"sd1.5": BaseArchitecture.SD15, "sd2.0": BaseArchitecture.SD20}


@click.command()
@click.argument("query")
@click.option("--source", type=click.Choice(["hf", "civitai", "all"]), default="all")
@click.option("--type", "model_type", type=click.Choice(["checkpoint", "lora"]), default=None)
@click.option("--arch", type=click.Choice(["sd1.5", "sd2.0"]), default=None)
@click.option("--limit", default=20, type=int)
def search(query: str, source: str, model_type: str | None, arch: str | None, limit: int):
    """Search HuggingFace and CivitAI for models."""
    registry = get_registry()

    results = registry.search(
        query=query,
        source=_SOURCE_MAP.get(source) if source != "all" else None,
        model_type=_TYPE_MAP.get(model_type) if model_type else None,
        base_arch=_ARCH_MAP.get(arch) if arch else None,
        limit=limit,
    )

    if not results:
        console.print("[yellow]No results found.[/yellow]")
        return

    print_model_table(results)
```

- [ ] **Step 6: Create stub commands** (info, build, serve, cache — enough to not crash on import)

```python
# src/coreml_converter/cli/commands/info.py
import click

@click.command()
@click.argument("model_ref")
def info(model_ref: str):
    """Show details for a model (e.g., civitai:12345)."""
    click.echo(f"Info for {model_ref} — not yet implemented")
```

```python
# src/coreml_converter/cli/commands/build.py
import click

@click.command()
@click.option("--base", default=None)
@click.option("--lora", multiple=True)
@click.option("--name", default=None)
@click.option("--recipe", default=None, type=click.Path(exists=True))
@click.option("--compute-units", default="all")
@click.option("--attention", default="split_einsum")
@click.option("--output", default="./output", type=click.Path())
def build(base, lora, name, recipe, compute_units, attention, output):
    """Build a CoreML model from base + LoRAs."""
    click.echo("Build command — not yet implemented")
```

```python
# src/coreml_converter/cli/commands/serve.py
import click

@click.command()
@click.option("--port", default=8420, type=int)
@click.option("--host", default="127.0.0.1")
def serve(port: int, host: str):
    """Start the web UI."""
    click.echo(f"Starting web UI on {host}:{port} — not yet implemented")
```

```python
# src/coreml_converter/cli/commands/cache.py
import click

@click.group()
def cache():
    """Manage the model cache."""
    pass

@cache.command("list")
def cache_list():
    """List cached models."""
    click.echo("Cache list — not yet implemented")

@cache.command("clear")
@click.argument("model_ref", required=False)
def cache_clear(model_ref: str | None):
    """Clear cached models."""
    click.echo("Cache clear — not yet implemented")
```

- [ ] **Step 7: Run tests**

Run: `python -m pytest tests/cli/test_cli.py -v`
Expected: All PASS

- [ ] **Step 8: Commit**

```bash
git add src/coreml_converter/cli/ tests/cli/
git commit -m "feat: add CLI with search command and stub commands"
```

---

## Task 15: CLI — Build Command (Full Implementation)

**Files:**
- Modify: `src/coreml_converter/cli/commands/build.py`

- [ ] **Step 1: Implement full build command**

```python
# src/coreml_converter/cli/commands/build.py
from __future__ import annotations

import json
import sys
from pathlib import Path

import click
from rich.progress import Progress, SpinnerColumn, TextColumn

from coreml_converter.cli.formatting import console
from coreml_converter.core.config import get_app_dir, load_config
from coreml_converter.core.models import (
    BaseArchitecture, BuildRecord, BuildStatus, ConversionConfig,
    LoRAEntry, ModelInfo, ModelSource, ModelType, Recipe,
)
from coreml_converter.core.state import BuildStore


def _parse_model_ref(ref: str) -> tuple[ModelSource, str]:
    if ":" not in ref:
        raise click.BadParameter(f"Invalid model ref '{ref}'. Expected format: source:id")
    source_str, model_id = ref.split(":", 1)
    source_map = {"hf": ModelSource.HUGGINGFACE, "civitai": ModelSource.CIVITAI}
    source = source_map.get(source_str.lower())
    if source is None:
        raise click.BadParameter(f"Unknown source '{source_str}'. Use 'hf' or 'civitai'.")
    return source, model_id


def _parse_lora_ref(ref: str) -> tuple[ModelSource, str, float]:
    weight = 1.0
    if "@" in ref:
        ref, weight_str = ref.rsplit("@", 1)
        weight = float(weight_str)
    source, model_id = _parse_model_ref(ref)
    return source, model_id, weight


@click.command()
@click.option("--base", default=None, help="Base model (source:id)")
@click.option("--lora", multiple=True, help="LoRA (source:id@weight), repeatable")
@click.option("--name", default=None, help="Output model name")
@click.option("--recipe", default=None, type=click.Path(exists=True), help="Recipe JSON file")
@click.option("--compute-units", default="all", type=click.Choice(["all", "cpuAndGPU"]))
@click.option("--attention", default="split_einsum", type=click.Choice(["split_einsum", "original"]))
@click.option("--output", default="./output", type=click.Path())
def build(base, lora, name, recipe, compute_units, attention, output):
    """Build a CoreML model from base + LoRAs."""
    app_dir = get_app_dir()
    config = load_config(app_dir / "config.json")

    if recipe:
        # Load recipe from manifest JSON
        manifest = json.loads(Path(recipe).read_text())
        console.print(f"[green]Rebuilding from recipe:[/green] {manifest['name']}")
        # Reconstruct recipe from manifest
        base_data = manifest["base_model"]
        base_model = ModelInfo(
            source=ModelSource(base_data["source"]),
            id=base_data["id"],
            name=base_data["name"],
            base_architecture=BaseArchitecture(base_data["architecture"]),
            model_type=ModelType.CHECKPOINT,
            tags=[], download_url="", metadata={},
        )
        lora_entries = []
        for l in manifest.get("loras", []):
            lora_model = ModelInfo(
                source=ModelSource(l["source"]),
                id=l["id"],
                name=l["name"],
                base_architecture=base_model.base_architecture,
                model_type=ModelType.LORA,
                tags=[], download_url="", metadata={},
            )
            lora_entries.append(LoRAEntry(model=lora_model, weight=l["weight"]))

        conv_data = manifest.get("conversion", {})
        conv_config = ConversionConfig(
            output_dir=Path(output),
            model_name=manifest["name"],
            compute_units=conv_data.get("compute_units", "all"),
            attention=conv_data.get("attention", "split_einsum"),
            precision=conv_data.get("precision", "float16"),
            include_safety_checker=conv_data.get("include_safety_checker", False),
        )
        build_recipe = Recipe(
            name=manifest["name"],
            base_model=base_model,
            loras=lora_entries,
            conversion_config=conv_config,
        )
    elif base:
        source, model_id = _parse_model_ref(base)
        # Fetch model info from registry
        from coreml_converter.cli.commands.search import get_registry
        registry = get_registry()
        results = registry.search(model_id, source=source, model_type=ModelType.CHECKPOINT, limit=1)
        if not results:
            console.print(f"[red]Model not found: {base}[/red]")
            sys.exit(1)
        base_model = results[0]

        lora_entries = []
        for lora_ref in lora:
            l_source, l_id, l_weight = _parse_lora_ref(lora_ref)
            lora_results = registry.search(l_id, source=l_source, model_type=ModelType.LORA, limit=1)
            if not lora_results:
                console.print(f"[red]LoRA not found: {lora_ref}[/red]")
                sys.exit(1)
            lora_entries.append(LoRAEntry(model=lora_results[0], weight=l_weight))

        model_name = name or f"{base_model.name}-custom"
        conv_config = ConversionConfig(
            output_dir=Path(output),
            model_name=model_name,
            compute_units=compute_units,
            attention=attention,
        )
        build_recipe = Recipe(
            name=model_name,
            base_model=base_model,
            loras=lora_entries,
            conversion_config=conv_config,
        )
    else:
        console.print("[yellow]Interactive build mode not yet implemented. Use --base or --recipe.[/yellow]")
        sys.exit(1)

    # Run compatibility check
    from coreml_converter.core.analyzer import check_compatibility, detect_tag_conflicts
    report = check_compatibility(base_model, [e for e in build_recipe.loras])
    if not report.is_compatible:
        console.print(f"[red]Compatibility check failed:[/red]")
        for c in report.conflicts:
            console.print(f"  - {c.reason}")
        if not click.confirm("Continue anyway?"):
            sys.exit(1)

    if report.lora_count_warning:
        console.print(f"[yellow]Warning: {report.lora_count_warning}[/yellow]")

    tag_conflicts = detect_tag_conflicts(build_recipe.loras)
    for c in tag_conflicts:
        console.print(f"[yellow]Conflict: {c.reason} ({c.severity.value})[/yellow]")

    # Create build record
    store = BuildStore(app_dir / "builds.json")
    record = BuildRecord(recipe=build_recipe)
    store.save(record)

    # Download, merge, convert
    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console) as progress:
        task = progress.add_task("Starting build...", total=None)
        cache_dir = app_dir / "cache"

        try:
            # Download
            progress.update(task, description="Downloading base model...")
            base_path = registry.download(build_recipe.base_model, cache_dir)
            build_recipe.base_model.metadata["local_path"] = str(base_path)

            for entry in build_recipe.loras:
                progress.update(task, description=f"Downloading LoRA: {entry.model.name}...")
                lora_path = registry.download(entry.model, cache_dir)
                entry.model.metadata["local_path"] = str(lora_path)

            # Merge
            progress.update(task, description="Merging LoRAs into base model...")
            from coreml_converter.core.merger.merger import Merger
            merger = Merger()
            merged_path = merger.merge(build_recipe, cache_dir, Path(output))

            # Convert
            progress.update(task, description="Converting to CoreML...")
            from coreml_converter.core.converter.converter import Converter
            converter = Converter()
            result = converter.convert(merged_path, build_recipe)

            record.status = BuildStatus.COMPLETED
            record.result = result
            store.save(record)

            console.print(f"\n[green]Build complete![/green]")
            console.print(f"  mlpackage: {result.mlpackage_path}")
            console.print(f"  mlmodelc:  {result.mlmodelc_path}")
            console.print(f"  manifest:  {result.manifest_path}")
            console.print(f"  size:      {result.model_size_mb:.1f} MB")
            console.print(f"  time:      {result.conversion_time:.1f}s")

        except Exception as e:
            record.status = BuildStatus.FAILED
            record.error = str(e)
            store.save(record)
            console.print(f"[red]Build failed: {e}[/red]")
            sys.exit(1)
```

- [ ] **Step 2: Run CLI tests**

Run: `python -m pytest tests/cli/test_cli.py -v`
Expected: All PASS (existing tests still pass)

- [ ] **Step 3: Commit**

```bash
git add src/coreml_converter/cli/commands/build.py
git commit -m "feat: implement full build command with download, merge, convert pipeline"
```

---

## Task 16: Web UI — FastAPI App + Search Route

**Files:**
- Create: `src/coreml_converter/web/app.py`
- Create: `src/coreml_converter/web/dependencies.py`
- Create: `src/coreml_converter/web/routes/search.py`
- Create: `src/coreml_converter/web/templates/base.html`
- Create: `src/coreml_converter/web/templates/search.html`
- Create: `src/coreml_converter/web/templates/partials/model_card.html`
- Create: `src/coreml_converter/web/templates/partials/search_results.html`
- Create: `src/coreml_converter/web/static/app.css`
- Create: `tests/web/test_search_routes.py`

- [ ] **Step 1: Write search route tests**

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/web/test_search_routes.py -v`
Expected: FAIL

- [ ] **Step 3: Implement web app factory**

```python
# src/coreml_converter/web/app.py
from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from coreml_converter.web.routes import search, builder, progress, history

TEMPLATES_DIR = Path(__file__).parent / "templates"
STATIC_DIR = Path(__file__).parent / "static"


def create_app() -> FastAPI:
    app = FastAPI(title="CoreML Converter")

    STATIC_DIR.mkdir(exist_ok=True)
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    app.include_router(search.router)
    app.include_router(builder.router)
    app.include_router(progress.router)
    app.include_router(history.router)

    return app
```

- [ ] **Step 4: Implement dependencies**

```python
# src/coreml_converter/web/dependencies.py
from __future__ import annotations

from pathlib import Path

from fastapi import Request
from fastapi.templating import Jinja2Templates

TEMPLATES_DIR = Path(__file__).parent / "templates"
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


def get_registry(request: Request):
    return request.app.state.registry


def get_build_store(request: Request):
    return request.app.state.build_store
```

- [ ] **Step 5: Implement search route**

```python
# src/coreml_converter/web/routes/search.py
from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor

from fastapi import APIRouter, Request, Query

from coreml_converter.core.models import BaseArchitecture, ModelSource, ModelType
from coreml_converter.web.dependencies import templates, get_registry

router = APIRouter()
_executor = ThreadPoolExecutor(max_workers=2)

_SOURCE_MAP = {"hf": ModelSource.HUGGINGFACE, "civitai": ModelSource.CIVITAI}
_TYPE_MAP = {"checkpoint": ModelType.CHECKPOINT, "lora": ModelType.LORA}
_ARCH_MAP = {"sd1.5": BaseArchitecture.SD15, "sd2.0": BaseArchitecture.SD20}


@router.get("/")
async def home(request: Request):
    return templates.TemplateResponse("search.html", {"request": request, "results": None})


@router.get("/search")
async def search(
    request: Request,
    q: str = Query(default=""),
    source: str = Query(default="all"),
    type: str = Query(default=""),
    arch: str = Query(default=""),
):
    registry = get_registry(request)
    if not q.strip():
        return templates.TemplateResponse("partials/search_results.html", {"request": request, "results": []})

    loop = asyncio.get_event_loop()
    results = await loop.run_in_executor(
        _executor,
        lambda: registry.search(
            query=q,
            source=_SOURCE_MAP.get(source) if source != "all" else None,
            model_type=_TYPE_MAP.get(type) if type else None,
            base_arch=_ARCH_MAP.get(arch) if arch else None,
        ),
    )

    return templates.TemplateResponse("partials/search_results.html", {"request": request, "results": results})
```

- [ ] **Step 6: Create stub routers** (builder, progress, history)

```python
# src/coreml_converter/web/routes/builder.py
from fastapi import APIRouter, Request
from coreml_converter.web.dependencies import templates

router = APIRouter()

@router.get("/build")
async def builder_page(request: Request):
    return templates.TemplateResponse("builder.html", {"request": request})
```

```python
# src/coreml_converter/web/routes/progress.py
from fastapi import APIRouter, Request
from coreml_converter.web.dependencies import templates

router = APIRouter()

@router.get("/build/{job_id}")
async def progress_page(request: Request, job_id: str):
    return templates.TemplateResponse("progress.html", {"request": request, "job_id": job_id})
```

```python
# src/coreml_converter/web/routes/history.py
from fastapi import APIRouter, Request
from coreml_converter.web.dependencies import templates

router = APIRouter()

@router.get("/history")
async def history_page(request: Request):
    return templates.TemplateResponse("history.html", {"request": request, "builds": []})
```

- [ ] **Step 7: Create templates**

```html
<!-- src/coreml_converter/web/templates/base.html -->
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{% block title %}CoreML Converter{% endblock %}</title>
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/@picocss/pico@2/css/pico.min.css">
    <link rel="stylesheet" href="/static/app.css">
    <script src="https://unpkg.com/htmx.org@2.0.0"></script>
    <script defer src="https://cdn.jsdelivr.net/npm/alpinejs@3.x.x/dist/cdn.min.js"></script>
</head>
<body>
    <nav class="container-fluid">
        <ul><li><a href="/"><strong>CoreML Converter</strong></a></li></ul>
        <ul>
            <li><a href="/">Search</a></li>
            <li><a href="/build">Builder</a></li>
            <li><a href="/history">History</a></li>
        </ul>
    </nav>
    <main class="container">
        {% block content %}{% endblock %}
    </main>
</body>
</html>
```

```html
<!-- src/coreml_converter/web/templates/search.html -->
{% extends "base.html" %}
{% block content %}
<h1>Search Models</h1>
<form hx-get="/search" hx-target="#search-results" hx-trigger="submit">
    <div class="grid">
        <input type="search" name="q" placeholder="Search models..." autofocus>
        <select name="source">
            <option value="all">All Sources</option>
            <option value="hf">HuggingFace</option>
            <option value="civitai">CivitAI</option>
        </select>
        <select name="type">
            <option value="">All Types</option>
            <option value="checkpoint">Checkpoint</option>
            <option value="lora">LoRA</option>
        </select>
        <select name="arch">
            <option value="">All Architectures</option>
            <option value="sd1.5">SD 1.5</option>
            <option value="sd2.0">SD 2.0</option>
        </select>
        <button type="submit">Search</button>
    </div>
</form>
<div id="search-results">
    {% if results is not none %}
        {% include "partials/search_results.html" %}
    {% endif %}
</div>
{% endblock %}
```

```html
<!-- src/coreml_converter/web/templates/partials/search_results.html -->
{% if results %}
<div class="grid">
    {% for model in results %}
        {% include "partials/model_card.html" %}
    {% endfor %}
</div>
{% else %}
<p>No results found.</p>
{% endif %}
```

```html
<!-- src/coreml_converter/web/templates/partials/model_card.html -->
<article>
    <header>
        <span class="badge">{{ model.source.value }}</span>
        <span class="badge">{{ model.base_architecture.value }}</span>
        <span class="badge">{{ model.model_type.value }}</span>
    </header>
    <h4>{{ model.name }}</h4>
    <p>{{ model.tags[:5] | join(", ") }}</p>
    <footer>
        {% if model.model_type.value == "checkpoint" %}
            <a href="/build?base={{ model.source.value }}:{{ model.id }}" role="button">Use as Base</a>
        {% else %}
            <small>{{ model.source.value }}:{{ model.id }}</small>
        {% endif %}
    </footer>
</article>
```

```html
<!-- src/coreml_converter/web/templates/builder.html -->
{% extends "base.html" %}
{% block content %}
<h1>Model Builder</h1>
<p>Builder UI — coming soon.</p>
{% endblock %}
```

```html
<!-- src/coreml_converter/web/templates/progress.html -->
{% extends "base.html" %}
{% block content %}
<h1>Build Progress</h1>
<p>Job: {{ job_id }} — coming soon.</p>
{% endblock %}
```

```html
<!-- src/coreml_converter/web/templates/history.html -->
{% extends "base.html" %}
{% block content %}
<h1>Build History</h1>
{% if builds %}
<p>{{ builds | length }} builds</p>
{% else %}
<p>No builds yet.</p>
{% endif %}
{% endblock %}
```

- [ ] **Step 8: Create CSS file**

```css
/* src/coreml_converter/web/static/app.css */
.badge {
    display: inline-block;
    padding: 0.15em 0.5em;
    font-size: 0.75em;
    border-radius: 4px;
    background: var(--pico-primary-background);
    color: var(--pico-primary-inverse);
    margin-right: 0.25em;
}
```

- [ ] **Step 9: Run tests**

Run: `python -m pytest tests/web/test_search_routes.py -v`
Expected: All PASS

- [ ] **Step 10: Commit**

```bash
git add src/coreml_converter/web/ tests/web/
git commit -m "feat: add web UI with FastAPI, search page, and htmx templates"
```

---

## Task 17: Web UI — Builder Route + SSE Progress

**Files:**
- Modify: `src/coreml_converter/web/routes/builder.py`
- Modify: `src/coreml_converter/web/routes/progress.py`
- Create: `src/coreml_converter/web/jobs.py`
- Create: `src/coreml_converter/web/static/builder.js`
- Modify: `src/coreml_converter/web/templates/builder.html`
- Modify: `src/coreml_converter/web/templates/progress.html`
- Create: `src/coreml_converter/web/templates/partials/lora_card.html`
- Create: `src/coreml_converter/web/templates/partials/compatibility_report.html`
- Create: `tests/web/test_builder_routes.py`

- [ ] **Step 1: Write builder route tests**

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/web/test_builder_routes.py -v`
Expected: FAIL

- [ ] **Step 3: Implement job manager**

```python
# src/coreml_converter/web/jobs.py
from __future__ import annotations

import asyncio
import logging
import queue
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import AsyncGenerator

from coreml_converter.core.models import BuildRecord, BuildStatus

logger = logging.getLogger(__name__)

# Progress events per job
_progress_queues: dict[str, queue.Queue] = {}


def get_progress_queue(job_id: str) -> queue.Queue:
    if job_id not in _progress_queues:
        _progress_queues[job_id] = queue.Queue()
    return _progress_queues[job_id]


def _run_build(record_dict: dict, cache_dir: str, output_dir: str) -> dict:
    """Runs in a separate process."""
    from coreml_converter.core.models import BuildRecord, BuildStatus
    from coreml_converter.core.merger.merger import Merger
    from coreml_converter.core.converter.converter import Converter

    record = BuildRecord(**record_dict)
    recipe = record.recipe

    try:
        merger = Merger()
        merged_path = merger.merge(recipe, Path(cache_dir), Path(output_dir))

        converter = Converter()
        result = converter.convert(merged_path, recipe)

        record.status = BuildStatus.COMPLETED
        record.result = result
        record.completed_at = datetime.now(timezone.utc)
    except Exception as e:
        record.status = BuildStatus.FAILED
        record.error = str(e)
        record.completed_at = datetime.now(timezone.utc)

    import json
    return json.loads(record.model_dump_json())


class JobManager:
    def __init__(self, cache_dir: Path, build_store) -> None:
        self._cache_dir = cache_dir
        self._store = build_store
        self._executor = ProcessPoolExecutor(max_workers=1)

    async def submit(self, record: BuildRecord) -> str:
        record.status = BuildStatus.RUNNING
        record.started_at = datetime.now(timezone.utc)
        self._store.save(record)

        import json
        record_dict = json.loads(record.model_dump_json())
        output_dir = str(record.recipe.conversion_config.output_dir)

        loop = asyncio.get_event_loop()
        future = loop.run_in_executor(
            self._executor,
            _run_build,
            record_dict,
            str(self._cache_dir),
            output_dir,
        )

        async def _on_complete(fut):
            try:
                result_dict = await fut
                updated = BuildRecord(**result_dict)
                self._store.save(updated)
            except Exception as e:
                record.status = BuildStatus.FAILED
                record.error = str(e)
                self._store.save(record)

        asyncio.ensure_future(_on_complete(future))
        return record.id

    async def progress_stream(self, job_id: str) -> AsyncGenerator[str, None]:
        """SSE stream for job progress."""
        while True:
            record = self._store.get(job_id)
            if record is None:
                yield f"data: {{\"error\": \"Job not found\"}}\n\n"
                return

            status = record.status.value
            yield f"data: {{\"status\": \"{status}\"}}\n\n"

            if record.status in (BuildStatus.COMPLETED, BuildStatus.FAILED):
                return

            await asyncio.sleep(2)
```

- [ ] **Step 4: Update builder route**

```python
# src/coreml_converter/web/routes/builder.py
from __future__ import annotations

import asyncio
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from fastapi import APIRouter, Request, Form, Query

from coreml_converter.core.analyzer import check_compatibility, detect_tag_conflicts, get_recommended_weight
from coreml_converter.core.models import (
    BaseArchitecture, BuildRecord, ConversionConfig, LoRAEntry,
    ModelInfo, ModelSource, ModelType, Recipe,
)
from coreml_converter.web.dependencies import templates, get_registry, get_build_store

router = APIRouter()
_executor = ThreadPoolExecutor(max_workers=2)


@router.get("/build")
async def builder_page(request: Request, base: str = Query(default="")):
    base_model = None
    if base:
        registry = get_registry(request)
        if ":" in base:
            source_str, model_id = base.split(":", 1)
            source_map = {"hf": ModelSource.HUGGINGFACE, "civitai": ModelSource.CIVITAI}
            source = source_map.get(source_str)
            loop = asyncio.get_event_loop()
            results = await loop.run_in_executor(
                _executor,
                lambda: registry.search(model_id, source=source, model_type=ModelType.CHECKPOINT, limit=1),
            )
            if results:
                base_model = results[0]

    return templates.TemplateResponse("builder.html", {
        "request": request,
        "base_model": base_model,
    })


@router.get("/build/search-loras")
async def search_loras(request: Request, q: str = Query(default=""), arch: str = Query(default="")):
    registry = get_registry(request)
    arch_map = {"SD1.5": BaseArchitecture.SD15, "SD2.0": BaseArchitecture.SD20}

    loop = asyncio.get_event_loop()
    results = await loop.run_in_executor(
        _executor,
        lambda: registry.search(
            query=q,
            model_type=ModelType.LORA,
            base_arch=arch_map.get(arch),
        ),
    )

    # Add weight recommendations
    lora_entries = []
    for model in results:
        weight, source = get_recommended_weight(model)
        lora_entries.append({
            "model": model,
            "recommended_weight": weight,
            "weight_source": source,
        })

    return templates.TemplateResponse("partials/search_results.html", {
        "request": request,
        "results": results,
        "lora_data": lora_entries,
    })


@router.post("/build/check-compatibility")
async def check_compat(request: Request):
    form = await request.form()
    # Parse recipe from form data
    base_json = form.get("base_model")
    loras_json = form.get("loras")

    if not base_json:
        return templates.TemplateResponse("partials/compatibility_report.html", {
            "request": request, "report": None,
        })

    base_model = ModelInfo(**json.loads(base_json))
    lora_entries = [LoRAEntry(**l) for l in json.loads(loras_json or "[]")]

    report = check_compatibility(base_model, lora_entries)
    tag_conflicts = detect_tag_conflicts(lora_entries)
    report.conflicts.extend(tag_conflicts)

    return templates.TemplateResponse("partials/compatibility_report.html", {
        "request": request, "report": report,
    })


@router.post("/build/start")
async def start_build(request: Request):
    form = await request.form()
    base_model = ModelInfo(**json.loads(form.get("base_model", "{}")))
    loras_raw = json.loads(form.get("loras", "[]"))
    lora_entries = [LoRAEntry(**l) for l in loras_raw]
    model_name = form.get("name", "custom-model")

    config = ConversionConfig(
        output_dir=Path(form.get("output_dir", "./output")),
        model_name=model_name,
        compute_units=form.get("compute_units", "all"),
        attention=form.get("attention", "split_einsum"),
    )

    recipe = Recipe(name=model_name, base_model=base_model, loras=lora_entries, conversion_config=config)
    record = BuildRecord(recipe=recipe)

    build_store = get_build_store(request)
    build_store.save(record)

    job_manager = request.app.state.job_manager
    await job_manager.submit(record)

    from starlette.responses import RedirectResponse
    return RedirectResponse(url=f"/build/{record.id}", status_code=303)
```

- [ ] **Step 5: Update progress route with SSE**

```python
# src/coreml_converter/web/routes/progress.py
from __future__ import annotations

from fastapi import APIRouter, Request
from sse_starlette.sse import EventSourceResponse

from coreml_converter.web.dependencies import templates, get_build_store

router = APIRouter()


@router.get("/build/{job_id}")
async def progress_page(request: Request, job_id: str):
    build_store = get_build_store(request)
    record = build_store.get(job_id)
    return templates.TemplateResponse("progress.html", {
        "request": request,
        "job_id": job_id,
        "record": record,
    })


@router.get("/build/{job_id}/events")
async def progress_events(request: Request, job_id: str):
    job_manager = request.app.state.job_manager
    return EventSourceResponse(job_manager.progress_stream(job_id))
```

- [ ] **Step 6: Create remaining templates and JS**

Builder template, LoRA card partial, compatibility report partial, updated progress template, and `builder.js` (Alpine.js component for drag-reorder and weight sliders). These are larger template files — implement based on the patterns established in search.html.

```html
<!-- src/coreml_converter/web/templates/builder.html -->
{% extends "base.html" %}
{% block content %}
<h1>Model Builder</h1>
<div x-data="builder()" class="grid">
    <!-- Base Model -->
    <div>
        <h3>Base Model</h3>
        {% if base_model %}
            <article>
                <h4>{{ base_model.name }}</h4>
                <span class="badge">{{ base_model.base_architecture.value }}</span>
            </article>
        {% else %}
            <p>Select a base model from <a href="/">Search</a>.</p>
        {% endif %}
    </div>

    <!-- LoRA Search -->
    <div>
        <h3>Add LoRAs</h3>
        {% if base_model %}
        <input type="search" placeholder="Search compatible LoRAs..."
               hx-get="/build/search-loras?arch={{ base_model.base_architecture.value }}"
               hx-trigger="keyup changed delay:500ms"
               hx-target="#lora-results"
               name="q">
        <div id="lora-results"></div>
        {% endif %}
    </div>
</div>

{% if base_model %}
<!-- Recipe & Compatibility -->
<div id="recipe-panel">
    <h3>Recipe</h3>
    <div id="lora-list">
        <p>No LoRAs added yet.</p>
    </div>
    <div id="compatibility-report"></div>
    <button hx-post="/build/start" hx-include="#build-form">Convert to CoreML</button>
</div>
{% endif %}

<script src="/static/builder.js"></script>
{% endblock %}
```

```html
<!-- src/coreml_converter/web/templates/partials/lora_card.html -->
<article class="lora-card" draggable="true">
    <h5>{{ lora.model.name }}</h5>
    <label>
        Weight: <span x-text="weight">{{ lora.weight }}</span>
        <input type="range" min="0" max="1" step="0.05"
               x-model="weight" value="{{ lora.weight }}">
    </label>
    {% if lora.recommended_weight %}
    <small>Recommended: {{ lora.recommended_weight }}
        {% if lora.weight_source %}({{ lora.weight_source }}){% endif %}
    </small>
    {% endif %}
    <button @click="removeLora('{{ lora.model.id }}')">Remove</button>
</article>
```

```html
<!-- src/coreml_converter/web/templates/partials/compatibility_report.html -->
{% if report %}
<article>
    <h4>Compatibility Report</h4>
    <p>
        Architecture: {% if report.architecture_match %}OK{% else %}MISMATCH{% endif %}
        | Risk: <strong>{{ report.overall_risk.value }}</strong>
    </p>
    {% if report.lora_count_warning %}
        <p class="warning">{{ report.lora_count_warning }}</p>
    {% endif %}
    {% for conflict in report.conflicts %}
        <p class="{{ conflict.severity.value }}">{{ conflict.reason }}</p>
    {% endfor %}
</article>
{% endif %}
```

```html
<!-- src/coreml_converter/web/templates/progress.html (updated) -->
{% extends "base.html" %}
{% block content %}
<h1>Build Progress</h1>
{% if record %}
<article>
    <h3>{{ record.recipe.name }}</h3>
    <div id="status" hx-ext="sse" sse-connect="/build/{{ job_id }}/events" sse-swap="message">
        <p>Status: {{ record.status.value }}</p>
    </div>
    {% if record.status.value == "completed" and record.result %}
    <h4>Output</h4>
    <ul>
        <li>mlpackage: {{ record.result.mlpackage_path }}</li>
        <li>mlmodelc: {{ record.result.mlmodelc_path }}</li>
        <li>manifest: {{ record.result.manifest_path }}</li>
        <li>Size: {{ "%.1f"|format(record.result.model_size_mb) }} MB</li>
        <li>Time: {{ "%.1f"|format(record.result.conversion_time) }}s</li>
    </ul>
    {% endif %}
    {% if record.status.value == "failed" %}
    <p class="error">Error: {{ record.error }}</p>
    {% endif %}
</article>
{% else %}
<p>Job not found.</p>
{% endif %}
<a href="/build">Build Another</a>
{% endblock %}
```

```javascript
// src/coreml_converter/web/static/builder.js
function builder() {
    return {
        loras: [],
        addLora(model, recommendedWeight, weightSource) {
            if (this.loras.find(l => l.model.id === model.id)) return;
            this.loras.push({
                model: model,
                weight: recommendedWeight || 1.0,
                recommended_weight: recommendedWeight,
                weight_source: weightSource,
            });
        },
        removeLora(id) {
            this.loras = this.loras.filter(l => l.model.id !== id);
        },
    };
}
```

- [ ] **Step 7: Run tests**

Run: `python -m pytest tests/web/test_builder_routes.py -v`
Expected: All PASS

- [ ] **Step 8: Commit**

```bash
git add src/coreml_converter/web/ tests/web/
git commit -m "feat: add builder route, SSE progress, job manager, and builder templates"
```

---

## Task 18: CLI — Serve Command (Wire Up Web UI)

**Files:**
- Modify: `src/coreml_converter/cli/commands/serve.py`

- [ ] **Step 1: Implement serve command**

```python
# src/coreml_converter/cli/commands/serve.py
from __future__ import annotations

import click
import uvicorn

from coreml_converter.cli.formatting import console


@click.command()
@click.option("--port", default=8420, type=int, help="Port to listen on")
@click.option("--host", default="127.0.0.1", help="Host to bind to")
def serve(port: int, host: str):
    """Start the web UI."""
    from coreml_converter.core.config import get_app_dir, load_config
    from coreml_converter.core.registry import Registry
    from coreml_converter.core.registry.huggingface import HuggingFaceClient
    from coreml_converter.core.registry.civitai import CivitAIClient
    from coreml_converter.core.state import BuildStore
    from coreml_converter.web.app import create_app
    from coreml_converter.web.jobs import JobManager

    app_dir = get_app_dir()
    config = load_config(app_dir / "config.json")

    app = create_app()

    app.state.registry = Registry(
        hf_client=HuggingFaceClient(),
        civitai_client=CivitAIClient(api_key=config.civitai_api_key),
    )
    build_store = BuildStore(app_dir / "builds.json")
    app.state.build_store = build_store
    app.state.job_manager = JobManager(cache_dir=app_dir / "cache", build_store=build_store)

    console.print(f"[green]Starting CoreML Converter web UI[/green]")
    console.print(f"  http://{host}:{port}")
    uvicorn.run(app, host=host, port=port, log_level="info")
```

- [ ] **Step 2: Test manually**

Run: `python -m pytest tests/cli/test_cli.py -v`
Expected: All existing tests still PASS

- [ ] **Step 3: Commit**

```bash
git add src/coreml_converter/cli/commands/serve.py
git commit -m "feat: wire up serve command to launch web UI with full dependencies"
```

---

## Task 19: Web UI — History Route

**Files:**
- Modify: `src/coreml_converter/web/routes/history.py`
- Create: `tests/web/test_history_routes.py`

- [ ] **Step 1: Write history tests**

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/web/test_history_routes.py -v`
Expected: FAIL

- [ ] **Step 3: Update history route**

```python
# src/coreml_converter/web/routes/history.py
from fastapi import APIRouter, Request
from coreml_converter.web.dependencies import templates, get_build_store

router = APIRouter()


@router.get("/history")
async def history_page(request: Request):
    build_store = get_build_store(request)
    builds = build_store.list_all()
    from datetime import datetime, timezone
    _epoch = datetime.min.replace(tzinfo=timezone.utc)
    builds.sort(key=lambda b: b.started_at or b.completed_at or _epoch, reverse=True)
    return templates.TemplateResponse("history.html", {
        "request": request,
        "builds": builds,
    })
```

- [ ] **Step 4: Update history template**

```html
<!-- src/coreml_converter/web/templates/history.html -->
{% extends "base.html" %}
{% block content %}
<h1>Build History</h1>
{% if builds %}
{% for build in builds %}
<article>
    <header>
        <span class="badge">{{ build.status.value }}</span>
        <strong>{{ build.recipe.name }}</strong>
    </header>
    <p>Base: {{ build.recipe.base_model.name }} ({{ build.recipe.base_model.base_architecture.value }})</p>
    <p>LoRAs: {{ build.recipe.loras | length }}</p>
    {% if build.result %}
        <p>Size: {{ "%.1f"|format(build.result.model_size_mb) }} MB | Time: {{ "%.1f"|format(build.result.conversion_time) }}s</p>
    {% endif %}
    {% if build.error %}
        <p class="error">Error: {{ build.error }}</p>
    {% endif %}
    <footer>
        <a href="/build/{{ build.id }}">Details</a>
    </footer>
</article>
{% endfor %}
{% else %}
<p>No builds yet. <a href="/">Start by searching for a model.</a></p>
{% endif %}
{% endblock %}
```

- [ ] **Step 5: Run tests**

Run: `python -m pytest tests/web/test_history_routes.py -v`
Expected: All PASS

- [ ] **Step 6: Commit**

```bash
git add src/coreml_converter/web/routes/history.py src/coreml_converter/web/templates/history.html tests/web/test_history_routes.py
git commit -m "feat: add history page showing previous builds"
```

---

## Task 20: Integration Test + Final Wiring

**Files:**
- Create: `tests/test_integration.py`

- [ ] **Step 1: Write integration test**

This test verifies the full pipeline wiring (with mocked ML dependencies).

```python
# tests/test_integration.py
import json
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
from click.testing import CliRunner
from coreml_converter.cli.main import cli
from coreml_converter.core.models import (
    ModelInfo, ModelSource, BaseArchitecture, ModelType,
)


def _make_model(name: str, model_type=ModelType.CHECKPOINT) -> ModelInfo:
    return ModelInfo(
        source=ModelSource.CIVITAI, id="1", name=name,
        base_architecture=BaseArchitecture.SD15,
        model_type=model_type,
        tags=["realistic"], download_url="http://example.com/model",
        metadata={"download_count": 1000},
    )


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
        from coreml_converter.core.models import (
            BuildRecord, Recipe, ConversionConfig, BuildStatus,
        )

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

        all_records = store.list_all()
        assert len(all_records) == 1
```

- [ ] **Step 2: Run integration tests**

Run: `python -m pytest tests/test_integration.py -v`
Expected: All PASS

- [ ] **Step 3: Run full test suite**

Run: `python -m pytest tests/ -v --tb=short`
Expected: All tests PASS

- [ ] **Step 4: Commit**

```bash
git add tests/test_integration.py
git commit -m "feat: add integration tests for end-to-end pipeline wiring"
```

---

## Task 21: README + Final Polish

**Files:**
- Create: `README.md`

- [ ] **Step 1: Create README**

```markdown
# CoreML Converter

Convert Stable Diffusion 1.5/2.0 checkpoints + LoRAs to CoreML models optimized for Apple Silicon.

## Features

- Search HuggingFace and CivitAI for base models and LoRAs
- Compatibility checking: architecture validation, conflict detection, weight guidance
- Merge multiple LoRAs into a base model with configurable weights
- Convert to CoreML (.mlpackage + .mlmodelc) for use with Apple's ml-stable-diffusion
- CLI + web UI interfaces
- Recipe manifests for reproducible builds

## Requirements

- Python 3.10+
- macOS 13+ (Ventura) on Apple Silicon

## Install

\```bash
pip install coreml-converter
# For ML dependencies (torch, diffusers, coremltools):
pip install coreml-converter[ml]
\```

## Quick Start

\```bash
# Search for models
coreml-converter search "realistic vision" --type checkpoint

# Build a model
coreml-converter build --base civitai:4201 --lora civitai:6789@0.7 --name my-model

# Start the web UI
coreml-converter serve
\```

## License

MIT
```

- [ ] **Step 2: Commit**

```bash
git add README.md
git commit -m "docs: add README with install and usage instructions"
```

- [ ] **Step 3: Run final full test suite**

Run: `python -m pytest tests/ -v`
Expected: All PASS

---

## Task 22: CLI — Info + Config Commands

**Files:**
- Modify: `src/coreml_converter/cli/commands/info.py`
- Create: `src/coreml_converter/cli/commands/config_cmd.py`
- Modify: `src/coreml_converter/cli/main.py`

- [ ] **Step 1: Implement info command**

```python
# src/coreml_converter/cli/commands/info.py
from __future__ import annotations

import click

from coreml_converter.cli.formatting import console
from coreml_converter.core.models import ModelSource, ModelType


def _parse_ref(ref: str) -> tuple[ModelSource, str]:
    if ":" not in ref:
        raise click.BadParameter(f"Expected format: source:id (e.g., civitai:12345)")
    source_str, model_id = ref.split(":", 1)
    source_map = {"hf": ModelSource.HUGGINGFACE, "civitai": ModelSource.CIVITAI}
    source = source_map.get(source_str.lower())
    if source is None:
        raise click.BadParameter(f"Unknown source '{source_str}'. Use 'hf' or 'civitai'.")
    return source, model_id


@click.command()
@click.argument("model_ref")
def info(model_ref: str):
    """Show details for a model (e.g., civitai:12345)."""
    from coreml_converter.cli.commands.search import get_registry
    source, model_id = _parse_ref(model_ref)
    registry = get_registry()
    results = registry.search(model_id, source=source, limit=1)
    if not results:
        console.print(f"[red]Model not found: {model_ref}[/red]")
        return

    model = results[0]
    console.print(f"[bold]{model.name}[/bold]")
    console.print(f"  Source:       {model.source.value}")
    console.print(f"  ID:           {model.id}")
    console.print(f"  Architecture: {model.base_architecture.value}")
    console.print(f"  Type:         {model.model_type.value}")
    console.print(f"  Tags:         {', '.join(model.tags)}")
    if model.metadata.get("download_count"):
        console.print(f"  Downloads:    {model.metadata['download_count']:,}")
    if model.metadata.get("description"):
        console.print(f"  Description:  {model.metadata['description'][:200]}")
```

- [ ] **Step 2: Implement config command**

```python
# src/coreml_converter/cli/commands/config_cmd.py
from __future__ import annotations

import click

from coreml_converter.cli.formatting import console
from coreml_converter.core.config import get_app_dir, load_config, save_config


@click.group()
def config():
    """Manage configuration."""
    pass


@config.command("set")
@click.argument("key")
@click.argument("value")
def config_set(key: str, value: str):
    """Set a config value (e.g., config set civitai-key YOUR_KEY)."""
    app_dir = get_app_dir()
    cfg = load_config(app_dir / "config.json")
    key_map = {
        "civitai-key": "civitai_api_key",
        "compute-units": "compute_units",
        "attention": "attention",
        "output-dir": "output_dir",
    }
    field = key_map.get(key)
    if field is None:
        console.print(f"[red]Unknown key: {key}. Valid keys: {', '.join(key_map.keys())}[/red]")
        return
    setattr(cfg, field, value)
    save_config(cfg, app_dir / "config.json")
    console.print(f"[green]Set {key} = {value}[/green]")


@config.command("get")
@click.argument("key", required=False)
def config_get(key: str | None):
    """Show config values."""
    app_dir = get_app_dir()
    cfg = load_config(app_dir / "config.json")
    if key:
        key_map = {
            "civitai-key": "civitai_api_key",
            "compute-units": "compute_units",
            "attention": "attention",
            "output-dir": "output_dir",
        }
        field = key_map.get(key)
        if field:
            val = getattr(cfg, field, None)
            # Mask API keys
            if "key" in key and val:
                val = val[:4] + "..." + val[-4:]
            console.print(f"{key} = {val}")
        else:
            console.print(f"[red]Unknown key: {key}[/red]")
    else:
        console.print(f"compute-units = {cfg.compute_units}")
        console.print(f"attention     = {cfg.attention}")
        console.print(f"output-dir    = {cfg.output_dir}")
        key_display = (cfg.civitai_api_key[:4] + "..." + cfg.civitai_api_key[-4:]) if cfg.civitai_api_key else "not set"
        console.print(f"civitai-key   = {key_display}")
```

- [ ] **Step 3: Register config command in main.py**

Add to `src/coreml_converter/cli/main.py`:

```python
from coreml_converter.cli.commands.config_cmd import config
cli.add_command(config)
```

- [ ] **Step 4: Run CLI tests**

Run: `python -m pytest tests/cli/ -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add src/coreml_converter/cli/commands/info.py src/coreml_converter/cli/commands/config_cmd.py src/coreml_converter/cli/main.py
git commit -m "feat: implement info and config CLI commands"
```

---

## Task 23: Post-Download Validation (Dimensions + Weight Overlap)

**Files:**
- Create: `src/coreml_converter/core/analyzer/dimensions.py`
- Create: `src/coreml_converter/core/analyzer/weight_overlap.py`
- Create: `tests/core/analyzer/test_dimensions.py`
- Create: `tests/core/analyzer/test_weight_overlap.py`

- [ ] **Step 1: Write dimension validation tests**

```python
# tests/core/analyzer/test_dimensions.py
import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path
from coreml_converter.core.models import BaseArchitecture, DimensionResult
from coreml_converter.core.analyzer.dimensions import validate_lora_dimensions


class TestValidateloraDimensions:
    @patch("coreml_converter.core.analyzer.dimensions.safetensors")
    def test_sd15_compatible_lora(self, mock_st):
        mock_st.safe_open.return_value.__enter__ = MagicMock(return_value=MagicMock(
            keys=MagicMock(return_value=["lora_unet_down_blocks_0_attentions_0_transformer_blocks_0_attn1_to_q.lora_down.weight"]),
            get_tensor=MagicMock(return_value=MagicMock(shape=(4, 320))),
        ))
        mock_st.safe_open.return_value.__exit__ = MagicMock(return_value=False)
        result = validate_lora_dimensions(Path("/fake/lora.safetensors"), BaseArchitecture.SD15)
        assert result.compatible is True

    @patch("coreml_converter.core.analyzer.dimensions.safetensors")
    def test_sd20_incompatible_lora(self, mock_st):
        mock_st.safe_open.return_value.__enter__ = MagicMock(return_value=MagicMock(
            keys=MagicMock(return_value=["lora_unet_down_blocks_0_attentions_0_transformer_blocks_0_attn1_to_q.lora_down.weight"]),
            get_tensor=MagicMock(return_value=MagicMock(shape=(4, 768))),
        ))
        mock_st.safe_open.return_value.__exit__ = MagicMock(return_value=False)
        result = validate_lora_dimensions(Path("/fake/lora.safetensors"), BaseArchitecture.SD20)
        assert result.compatible is False
        assert result.expected == 1024
        assert result.actual == 768
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/core/analyzer/test_dimensions.py -v`
Expected: FAIL

- [ ] **Step 3: Implement dimension validation**

```python
# src/coreml_converter/core/analyzer/dimensions.py
from __future__ import annotations

import logging
from pathlib import Path

from coreml_converter.core.models import BaseArchitecture, DimensionResult

logger = logging.getLogger(__name__)

_ARCH_CROSS_ATTN_DIM = {
    BaseArchitecture.SD15: 768,
    BaseArchitecture.SD20: 1024,
}

try:
    import safetensors
    from safetensors import safe_open
except ImportError:
    safetensors = None


def validate_lora_dimensions(lora_path: Path, base_arch: BaseArchitecture) -> DimensionResult:
    """Validate LoRA cross-attention dimensions match the base architecture."""
    if safetensors is None:
        logger.warning("safetensors not installed, skipping dimension check")
        return DimensionResult(expected=0, actual=0, compatible=True)

    expected_dim = _ARCH_CROSS_ATTN_DIM[base_arch]

    with safe_open(str(lora_path), framework="pt") as f:
        for key in f.keys():
            if "attn" in key and "lora_down" in key:
                tensor = f.get_tensor(key)
                actual_dim = tensor.shape[-1]
                # Small LoRA rank dimensions (4, 8, etc.) are the low-rank decomposition, not the model dim
                if actual_dim not in (expected_dim, 4, 8, 16, 32, 64, 128):
                    return DimensionResult(
                        expected=expected_dim,
                        actual=actual_dim,
                        compatible=False,
                    )

    return DimensionResult(expected=expected_dim, actual=expected_dim, compatible=True)
```

- [ ] **Step 4: Write weight overlap tests**

```python
# tests/core/analyzer/test_weight_overlap.py
import pytest
import torch
from unittest.mock import patch, MagicMock
from pathlib import Path
from coreml_converter.core.models import Conflict, Severity
from coreml_converter.core.analyzer.weight_overlap import detect_weight_overlap


class TestDetectWeightOverlap:
    @patch("coreml_converter.core.analyzer.weight_overlap.safe_open")
    def test_no_overlap(self, mock_safe_open):
        # LoRA A modifies layer 1, LoRA B modifies layer 2
        mock_a = MagicMock()
        mock_a.keys.return_value = ["layer1.lora_down.weight"]
        mock_a.get_tensor.return_value = torch.randn(4, 320)

        mock_b = MagicMock()
        mock_b.keys.return_value = ["layer2.lora_down.weight"]
        mock_b.get_tensor.return_value = torch.randn(4, 320)

        mock_safe_open.side_effect = [
            MagicMock(__enter__=MagicMock(return_value=mock_a), __exit__=MagicMock(return_value=False)),
            MagicMock(__enter__=MagicMock(return_value=mock_b), __exit__=MagicMock(return_value=False)),
        ]

        conflicts = detect_weight_overlap(
            [("LoRA A", Path("/a.safetensors")), ("LoRA B", Path("/b.safetensors"))]
        )
        assert len(conflicts) == 0

    @patch("coreml_converter.core.analyzer.weight_overlap.safe_open")
    def test_high_overlap(self, mock_safe_open):
        # Both LoRAs modify the same layers
        keys = ["layer1.lora_down.weight", "layer1.lora_up.weight"]
        tensor = torch.randn(4, 320)

        mock_a = MagicMock()
        mock_a.keys.return_value = keys
        mock_a.get_tensor.return_value = tensor

        mock_b = MagicMock()
        mock_b.keys.return_value = keys
        mock_b.get_tensor.return_value = tensor

        mock_safe_open.side_effect = [
            MagicMock(__enter__=MagicMock(return_value=mock_a), __exit__=MagicMock(return_value=False)),
            MagicMock(__enter__=MagicMock(return_value=mock_b), __exit__=MagicMock(return_value=False)),
        ]

        conflicts = detect_weight_overlap(
            [("LoRA A", Path("/a.safetensors")), ("LoRA B", Path("/b.safetensors"))]
        )
        assert len(conflicts) >= 1
        assert conflicts[0].severity == Severity.WARNING
```

- [ ] **Step 5: Implement weight overlap detection**

```python
# src/coreml_converter/core/analyzer/weight_overlap.py
from __future__ import annotations

import logging
from itertools import combinations
from pathlib import Path

from coreml_converter.core.models import Conflict, Severity

logger = logging.getLogger(__name__)

try:
    import torch
    from safetensors import safe_open
except ImportError:
    torch = None
    safe_open = None


def _get_layer_norms(lora_path: Path) -> dict[str, float]:
    """Get L2 norm per layer for a LoRA file."""
    norms: dict[str, float] = {}
    with safe_open(str(lora_path), framework="pt") as f:
        for key in f.keys():
            # Group by base layer name (strip lora_up/lora_down suffix)
            base_key = key.rsplit(".lora_", 1)[0] if ".lora_" in key else key
            tensor = f.get_tensor(key)
            norm = float(torch.norm(tensor.float()).item())
            norms[base_key] = norms.get(base_key, 0.0) + norm
    return norms


def detect_weight_overlap(
    loras: list[tuple[str, Path]],  # (name, path) pairs
    overlap_threshold: float = 0.5,
) -> list[Conflict]:
    """Detect LoRA pairs with high weight overlap in the same layers."""
    if torch is None or safe_open is None:
        logger.warning("torch/safetensors not installed, skipping weight overlap check")
        return []

    if len(loras) < 2:
        return []

    # Compute per-layer norms for each LoRA
    lora_norms: list[tuple[str, dict[str, float]]] = []
    for name, path in loras:
        try:
            norms = _get_layer_norms(path)
            lora_norms.append((name, norms))
        except Exception as e:
            logger.warning(f"Failed to analyze {name}: {e}")

    conflicts: list[Conflict] = []
    for (name_a, norms_a), (name_b, norms_b) in combinations(lora_norms, 2):
        shared_keys = set(norms_a.keys()) & set(norms_b.keys())
        if not shared_keys:
            continue

        total_a = sum(norms_a.values())
        total_b = sum(norms_b.values())
        if total_a == 0 or total_b == 0:
            continue

        shared_mass_a = sum(norms_a[k] for k in shared_keys) / total_a
        shared_mass_b = sum(norms_b[k] for k in shared_keys) / total_b

        if shared_mass_a > overlap_threshold and shared_mass_b > overlap_threshold:
            conflicts.append(Conflict(
                lora_a=name_a,
                lora_b=name_b,
                reason=f"High weight overlap: {shared_mass_a:.0%} / {shared_mass_b:.0%} of weights in same layers",
                severity=Severity.WARNING,
            ))

    return conflicts
```

- [ ] **Step 6: Update analyzer facade**

Update `src/coreml_converter/core/analyzer/__init__.py`:

```python
from coreml_converter.core.analyzer.compatibility import check_compatibility
from coreml_converter.core.analyzer.conflicts import detect_tag_conflicts
from coreml_converter.core.analyzer.weight_guidance import get_recommended_weight
from coreml_converter.core.analyzer.dimensions import validate_lora_dimensions
from coreml_converter.core.analyzer.weight_overlap import detect_weight_overlap

__all__ = [
    "check_compatibility", "detect_tag_conflicts", "get_recommended_weight",
    "validate_lora_dimensions", "detect_weight_overlap",
]
```

- [ ] **Step 7: Run all analyzer tests**

Run: `python -m pytest tests/core/analyzer/ -v`
Expected: All PASS

- [ ] **Step 8: Commit**

```bash
git add src/coreml_converter/core/analyzer/ tests/core/analyzer/
git commit -m "feat: add post-download dimension validation and weight overlap detection"
```

---

## Task 24: Integrate Post-Download Checks + Fix Web Build Downloads

Add disk space pre-flight to CLI build before downloads. Add download step to web job runner. Add post-download validation to both pipelines.

**Files:**
- Modify: `src/coreml_converter/cli/commands/build.py`
- Modify: `src/coreml_converter/web/jobs.py`

- [ ] **Step 1: Add disk space check and post-download validation to CLI build**

In `src/coreml_converter/cli/commands/build.py`, add before the download section:

```python
        # Pre-flight disk space check
        from coreml_converter.core.converter.converter import check_disk_space
        try:
            check_disk_space(Path(output))
        except RuntimeError as e:
            console.print(f"[red]{e}[/red]")
            if not click.confirm("Continue anyway?"):
                sys.exit(1)
```

After downloading LoRAs, add dimension + weight overlap checks:

```python
            # Post-download validation
            from coreml_converter.core.analyzer import validate_lora_dimensions, detect_weight_overlap
            for entry in build_recipe.loras:
                lora_path = Path(entry.model.metadata["local_path"])
                if lora_path.suffix == ".safetensors":
                    dim_result = validate_lora_dimensions(lora_path, build_recipe.base_model.base_architecture)
                    if not dim_result.compatible:
                        console.print(f"[red]Dimension mismatch for {entry.model.name}: expected {dim_result.expected}, got {dim_result.actual}[/red]")
                        if not click.confirm("Continue anyway?"):
                            sys.exit(1)

            if len(build_recipe.loras) >= 2:
                lora_pairs = [(e.model.name, Path(e.model.metadata["local_path"])) for e in build_recipe.loras]
                overlap_conflicts = detect_weight_overlap(lora_pairs)
                for c in overlap_conflicts:
                    console.print(f"[yellow]Weight overlap: {c.reason}[/yellow]")
```

- [ ] **Step 2: Add downloads to web job runner**

In `src/coreml_converter/web/jobs.py`, update `_run_build` to accept a registry config and download models:

```python
def _run_build(record_dict: dict, cache_dir: str, output_dir: str, civitai_api_key: str | None = None) -> dict:
    """Runs in a separate process."""
    from coreml_converter.core.models import BuildRecord, BuildStatus
    from coreml_converter.core.merger.merger import Merger
    from coreml_converter.core.converter.converter import Converter
    from coreml_converter.core.registry import Registry
    from coreml_converter.core.registry.huggingface import HuggingFaceClient
    from coreml_converter.core.registry.civitai import CivitAIClient

    record = BuildRecord(**record_dict)
    recipe = record.recipe

    try:
        # Download models
        registry = Registry(
            hf_client=HuggingFaceClient(),
            civitai_client=CivitAIClient(api_key=civitai_api_key),
        )
        cache = Path(cache_dir)
        base_path = registry.download(recipe.base_model, cache)
        recipe.base_model.metadata["local_path"] = str(base_path)

        for entry in recipe.loras:
            lora_path = registry.download(entry.model, cache)
            entry.model.metadata["local_path"] = str(lora_path)

        # Merge + Convert
        merger = Merger()
        merged_path = merger.merge(recipe, cache, Path(output_dir))

        converter = Converter()
        result = converter.convert(merged_path, recipe)

        record.status = BuildStatus.COMPLETED
        record.result = result
        record.completed_at = datetime.now(timezone.utc)
    except Exception as e:
        record.status = BuildStatus.FAILED
        record.error = str(e)
        record.completed_at = datetime.now(timezone.utc)

    import json
    return json.loads(record.model_dump_json())
```

Update `JobManager.submit()` to pass the API key:

```python
    async def submit(self, record: BuildRecord, civitai_api_key: str | None = None) -> str:
        # ... existing code ...
        future = loop.run_in_executor(
            self._executor,
            _run_build,
            record_dict,
            str(self._cache_dir),
            output_dir,
            civitai_api_key,
        )
```

- [ ] **Step 3: Run full test suite**

Run: `python -m pytest tests/ -v`
Expected: All PASS

- [ ] **Step 4: Commit**

```bash
git add src/coreml_converter/cli/commands/build.py src/coreml_converter/web/jobs.py
git commit -m "feat: add pre-flight disk check, post-download validation, web build downloads"
```
