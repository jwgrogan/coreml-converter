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


class TrainStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


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


class TrainingMode(str, Enum):
    """What the LoRA is meant to capture.

    These are not interchangeable presets over one algorithm: the captioning
    strategy inverts between CHARACTER and POSE. Whatever you caption stays
    promptable; whatever you leave uncaptioned is absorbed into the trigger.
    """

    CHARACTER = "character"
    POSE = "pose"
    STYLE = "style"


# Attention-only is right for identity — cross-attention to_k/to_v is what
# binds the trigger token to appearance. Pose is spatial so it wants the
# self-attention path too; style is diffuse and wants feed-forward as well.
MODE_TARGET_MODULES: dict[str, list[str]] = {
    TrainingMode.CHARACTER.value: ["to_k", "to_q", "to_v", "to_out.0"],
    TrainingMode.POSE.value: ["to_k", "to_q", "to_v", "to_out.0"],
    TrainingMode.STYLE.value: ["to_k", "to_q", "to_v", "to_out.0",
                               "ff.net.0.proj", "ff.net.2"],
}

# Step counts assume the text-encoder setting alongside them — the two
# interact strongly and must not be tuned independently. Training the text
# encoder makes every step do considerably more work (the trigger token's own
# embedding moves, not just the UNet's response to it), so a TE-enabled preset
# needs roughly a quarter of the steps. Measured: with TE on and 11 images,
# quality peaked at ~300 steps and was visibly damaged by 600 — blotched skin,
# rigid faces, collapsed composition. See F28/F29.
MODE_DEFAULTS: dict[str, dict] = {
    TrainingMode.CHARACTER.value: {"rank": 16, "steps": 400, "train_text_encoder": True},
    TrainingMode.POSE.value: {"rank": 12, "steps": 1200, "train_text_encoder": False},
    TrainingMode.STYLE.value: {"rank": 32, "steps": 1500, "train_text_encoder": False},
}

# Above this, a text-encoder run is very likely past its peak.
TE_STEP_CEILING = 800


def overtraining_warning(steps: int, train_text_encoder: bool) -> str | None:
    """Flag settings measured to overtrain, so the UI can say so up front."""
    if train_text_encoder and steps > TE_STEP_CEILING:
        return (f"{steps} steps with text-encoder training is likely to overtrain — "
                f"quality peaked near 300 steps in testing. Consider {TE_STEP_CEILING} "
                f"or fewer, or turn the text encoder off for longer runs.")
    return None


def caption_for(mode: str, trigger: str, class_token: str, suffix: str = "") -> str:
    """Build the training caption for a mode.

    CHARACTER omits any description of the face so identity is absorbed by the
    trigger. POSE deliberately does the opposite for the subject, keeping the
    person promptable while the (uncaptioned) pose binds to the trigger.
    """
    if mode == TrainingMode.STYLE.value:
        caption = f"{trigger} style"
    elif mode == TrainingMode.POSE.value:
        caption = f"a {class_token}, {trigger}".strip()
    else:
        caption = f"photo of {trigger} {class_token}".strip()
    suffix = (suffix or "").strip().lstrip(",").strip()
    return f"{caption}, {suffix}" if suffix else caption


class StyleFamily(str, Enum):
    """The visual family a LoRA is trained for.

    LoRAs transfer well *within* a family and poorly across one. Anime SD 1.5
    checkpoints almost all descend from the NovelAI finetune and sit a long way
    from base SD 1.5 in weight space, so a LoRA trained on a photoreal base
    lands weakly on them and fights the style. Pick the family, train on a
    neutral member of it, deploy across the rest.
    """

    PHOTOREAL = "photoreal"
    SEMI_REAL = "semi_real"
    ANIME = "anime"
    ILLUSTRATION = "illustration"
    GENERAL = "general"


# Always the *least stylised* competent member of the family. A checkpoint that
# is heavily biased toward your subject flatters the result but teaches the
# adapter less of the actual identity — the base is supplying the look, so the
# LoRA under-specifies it and falls apart on any other checkpoint. Competent at
# the category is what you want; idealised toward it is not. See F19/F25.
RECOMMENDED_BASES: dict[str, dict] = {
    StyleFamily.PHOTOREAL.value: {
        "label": "Photoreal",
        "description": "Photographs of real people or places.",
        "name": "epiCRealism",
        "alternatives": ["Realistic Vision v5.1", "CyberRealistic"],
        "hint": "epiCRealism renders natural, unretouched skin, so an ordinary face survives training. "
                "Realistic Vision is the safer all-rounder if you want the widest compatibility.",
        "avoid": "heavily idealised merges (URPM and similar) — their beauty prior competes with a real likeness",
        "dataset": "15-25 photos of one person, varied angle and lighting, 512px+ on the face after cropping",
        "search": "epicrealism",
    },
    StyleFamily.SEMI_REAL.value: {
        "label": "Semi-real / 2.5D",
        "description": "Stylised but grounded — game-art and cinematic character looks.",
        "name": "DreamShaper 8",
        "alternatives": ["Deliberate v2"],
        "hint": "DreamShaper spans photoreal and painterly, which is what makes it a forgiving base for this family.",
        "avoid": "checkpoints tuned hard toward one look; they narrow what the LoRA can express",
        "dataset": "15-25 images in a consistent degree of stylisation — do not mix photos with rendered art",
        "search": "dreamshaper",
    },
    StyleFamily.ANIME.value: {
        "label": "Anime",
        "description": "Cel-shaded and anime-styled characters.",
        "name": "AnyLoRA",
        "alternatives": ["Anything v4.5"],
        "hint": "AnyLoRA exists specifically to be a neutral anime training base — that is its whole purpose.",
        "avoid": "high-contrast stylised merges (AbyssOrangeMix and similar) — they bake their look into the LoRA",
        "dataset": "15-30 images of the character, consistent design; avoid heavy compression artefacts",
        "search": "anylora",
    },
    StyleFamily.ILLUSTRATION.value: {
        "label": "Graphic novel / fantasy",
        "description": "Painterly, comic and fantasy illustration.",
        "name": "DreamShaper 8",
        "alternatives": ["RevAnimated", "Deliberate v2"],
        "hint": "This family overlaps semi-real; RevAnimated leans further into fantasy if that is the target.",
        "avoid": "photoreal merges — they flatten painterly brushwork toward photography",
        "dataset": "15-30 images with consistent rendering style; mixed media confuses the adapter",
        "search": "dreamshaper",
    },
    StyleFamily.GENERAL.value: {
        "label": "Not sure",
        "description": "Unknown or mixed target checkpoints.",
        "name": "Stable Diffusion v1.5",
        "alternatives": [],
        "hint": "maximum neutrality; the safe default when the deployment checkpoint is unknown",
        "avoid": "any finetune with a strong house style",
        "dataset": "15-25 images, consistent subject, varied framing",
        "search": "stable diffusion 1.5",
    },
}


def recommended_base(family: str) -> dict:
    """What to train against for a given target family."""
    return RECOMMENDED_BASES.get(family, RECOMMENDED_BASES[StyleFamily.GENERAL.value])


class TrainingParams(BaseModel):
    """Tunables for a LoRA training run.

    Defaults are the "character preset" from the plan, informed by the Phase 0
    measurements: rank 16 / alpha 16 (they must match — the kohya exporter
    writes alpha = rank unconditionally), gradient accumulation 1 so every step
    is an optimizer update, and gradient checkpointing on because the
    memory-heavy alternative collapsed mid-run on a 24GB machine.
    """

    rank: int = 16
    steps: int = 1200
    learning_rate: float = 1e-4
    resolution: int = 512
    train_text_encoder: bool = False
    precision: str = "fp32"          # "fp32" | "fp16w"
    gradient_checkpointing: bool = True
    grad_accum: int = 1
    snr_gamma: float = 5.0
    warmup_steps: int = 100
    save_every: int = 250
    flip_augmentation: bool = True
    seed: int = 42
    target_modules: list[str] | None = None   # None -> resolved from the mode

    def resolved_targets(self, mode: str) -> list[str]:
        return self.target_modules or MODE_TARGET_MODULES.get(
            mode, MODE_TARGET_MODULES[TrainingMode.CHARACTER.value])

    @classmethod
    def for_mode(cls, mode: str, **overrides) -> "TrainingParams":
        values = dict(MODE_DEFAULTS.get(mode, MODE_DEFAULTS[TrainingMode.CHARACTER.value]))
        values.update({k: v for k, v in overrides.items() if v is not None})
        return cls(**values)


class TrainRequest(BaseModel):
    name: str
    trigger: str
    mode: TrainingMode = TrainingMode.CHARACTER
    # Recorded in the LoRA metadata so the Build tab can warn when a LoRA is
    # merged into a checkpoint from a different family.
    style_family: StyleFamily = StyleFamily.PHOTOREAL
    class_token: str = "woman"
    caption_suffix: str = ""
    image_paths: list[str]
    output_dir: Path
    base_model: ModelInfo | None = None
    base_path: Path | None = None
    params: TrainingParams = Field(default_factory=TrainingParams)


class TrainResult(BaseModel):
    lora_path: Path
    steps_completed: int
    training_time: float
    file_size_mb: float
    seconds_per_step: float
    loss_first: float | None = None
    loss_last: float | None = None
    images_used: int = 0


class TrainRecord(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    request: TrainRequest
    status: TrainStatus = TrainStatus.PENDING
    started_at: datetime | None = None
    completed_at: datetime | None = None
    result: TrainResult | None = None
    error: str | None = None
    schema_version: int = 1
