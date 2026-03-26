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
    weight_source: str | None = None

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
    studio: bool = False


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
