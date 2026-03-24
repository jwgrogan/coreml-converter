# CoreML Converter — Design Spec

## Overview

A Python tool (CLI + local web UI) that converts Stable Diffusion 1.5/2.0 checkpoints with merged LoRAs into CoreML models optimized for Apple Silicon. Searches HuggingFace and CivitAI for models, validates compatibility, guides users through building custom recipes, and outputs compiled `.mlmodelc` bundles ready for use with Apple's `ml-stable-diffusion` Swift package.

**Distribution:** Open source, installed via `pip` / Homebrew.
**Target user:** Developers building Swift apps with `ml-stable-diffusion` who want custom models from the broader HF/CivitAI ecosystem.

## Architecture

**Approach: Pipeline Architecture** — three layers with clean interfaces:

1. **Core library** (`coreml_converter.core`) — model registry, compatibility analyzer, LoRA merger, CoreML converter
2. **CLI** (`coreml_converter.cli`) — Click-based, thin wrapper over core
3. **Web UI** (`coreml_converter.web`) — FastAPI + Jinja2 + htmx/Alpine.js

All interfaces share the core library. No duplication of business logic. A Textual TUI is a potential future addition but is not part of v1.

**Requirements:** Python 3.10+, macOS 13+ (Ventura) on Apple Silicon.

## Project Structure

```
coreml-converter/
├── pyproject.toml
├── src/
│   └── coreml_converter/
│       ├── __init__.py
│       ├── core/
│       │   ├── registry/       # HuggingFace + CivitAI API clients
│       │   ├── analyzer/       # Compatibility checking, conflict detection
│       │   ├── merger/         # LoRA weight merging into base model
│       │   ├── converter/      # diffusers -> CoreML conversion
│       │   └── models.py       # Shared Pydantic data models
│       ├── cli/
│       │   └── commands/       # search, build, serve, cache, info
│       ├── web/
│       │   ├── routes/
│       │   ├── templates/      # Jinja2 + htmx
│       │   └── static/
│       └── tui/                # Future (not v1)
├── tests/
└── docs/
```

## Core Library

### Registry — HuggingFace + CivitAI Clients

**HuggingFace:**
- Uses `huggingface_hub` SDK
- Search filtered by pipeline tag (`stable-diffusion`, `text-to-image`) and tags (`sd-1.5`, `sd-2.0`)
- Download via `snapshot_download` to local cache

**CivitAI:**
- REST API (`https://civitai.com/api/v1/models`)
- Search with filters: model type (Checkpoint / LoRA), base model (SD 1.5 / SD 2.0)
- Fetch metadata: tags, description, base model version, images, stats
- Download `.safetensors` to local cache
- Rate limiting: token bucket, 2-3 req/sec

**API Authentication:**
- **CivitAI:** API key required for downloads. Stored in `~/.coreml-converter/config.json` or via `CIVITAI_API_KEY` env var. CLI `config set civitai-key <key>` to save. Prompted on first use if missing.
- **HuggingFace:** Token via `huggingface_hub` login (standard `HF_TOKEN` env var or `huggingface-cli login`). Gated models require a token; ungated models work without one. The tool surfaces a clear message when a gated model requires auth.

**Shared abstractions:**

```python
class ModelSource(str, Enum):
    HUGGINGFACE = "huggingface"
    CIVITAI = "civitai"

class BaseArchitecture(str, Enum):
    SD15 = "SD1.5"
    SD20 = "SD2.0"

class ModelType(str, Enum):
    CHECKPOINT = "checkpoint"
    LORA = "lora"

class ModelInfo(BaseModel):  # pydantic.BaseModel
    source: ModelSource
    id: str
    name: str
    base_architecture: BaseArchitecture
    model_type: ModelType
    tags: list[str]
    download_url: str
    metadata: dict

class LoRAEntry(BaseModel):
    model: ModelInfo
    weight: float = 1.0           # 0.0-1.0
    recommended_weight: float | None = None
    weight_source: str | None = None  # "creator" | "community" | "category_default"

class Recipe(BaseModel):
    name: str
    base_model: ModelInfo
    loras: list[LoRAEntry]        # ordered — application order matters
    conversion_config: ConversionConfig

class BuildRecord(BaseModel):
    id: str
    recipe: Recipe
    status: str                   # "pending" | "running" | "completed" | "failed"
    started_at: datetime | None
    completed_at: datetime | None
    result: ConversionResult | None
    error: str | None
    schema_version: int = 1
```

**Registry interface** — synchronous methods. The FastAPI web layer runs I/O-bound calls (search, download) in `ThreadPoolExecutor` via `run_in_executor()`. CPU-bound work (merge, conversion) runs in `ProcessPoolExecutor`.

```python
class Registry:
    def search(query, source, model_type, base_arch) -> list[ModelInfo]
    def get_compatible_loras(base_model: ModelInfo) -> list[ModelInfo]
    def download(model: ModelInfo, dest: Path) -> Path
```

Note: `huggingface_hub.snapshot_download` and `httpx` CivitAI calls are synchronous. For HuggingFace LoRAs (which lack CivitAI metadata), weight guidance falls back to category defaults. If no category can be inferred, default weight is 1.0.

**Local cache:** `~/.coreml-converter/cache/` keyed by source + ID. CLI `cache clear` to manage. HuggingFace downloads are natively resumable via `snapshot_download`. CivitAI downloads use `httpx` with range headers for resume support; partial files are suffixed `.partial` and cleaned up on failure after retries are exhausted.

### Analyzer — Compatibility & Conflict Detection

**Pre-download (metadata-based):**
- Validate base architecture match (LoRA SD1.5 tag must match base)
- Flag missing/ambiguous metadata as "unverified compatibility"
- Check model format is convertible (`.safetensors`, diffusers, `.ckpt`)

**Post-download (weight-based):**
- Validate cross-attention dimensions (768 for SD1.5, 1024 for SD2.0)
- Verify tensor shape compatibility layer-by-layer

**Conflict detection — tag-based:**
- Categorize LoRAs by CivitAI tags: `style`, `character`, `concept`, `clothing`, `pose`, `background`
- Flag multiple LoRAs in same category
- Severity: `info` (same broad category), `warning` (competing tags like "realistic" + "anime")

**Conflict detection — weight overlap analysis:**
- Compare which layers each LoRA modifies most heavily (L2 norm of delta)
- If two LoRAs share >50% weight mass in same layers, flag as "high overlap"
- Fast operation once weights are loaded in memory (loading from disk may add I/O time for large LoRAs)

**LoRA count warnings:**
- 1-3: green
- 4-5: soft warning ("quality may degrade")
- 6+: hard warning ("likely artifacts, proceed at own risk")

**Output:**

```python
class CompatibilityReport(BaseModel):
    is_compatible: bool
    architecture_match: bool
    dimension_check: DimensionResult | None
    conflicts: list[Conflict]
    lora_count_warning: str | None
    overall_risk: str  # "low" | "medium" | "high"
```

### LoRA Weight Guidance

Three sources, layered by priority:

1. **Creator-specified** — parsed from CivitAI model description (e.g., "recommended weight: 0.6-0.8")
2. **Community consensus** — median weight from CivitAI image generation metadata for popular images using the LoRA
3. **Category defaults** — fallback heuristics:
   - Style: 0.6-0.8
   - Character: 0.7-0.9
   - Detail/texture: 0.4-0.6
   - Concept: 0.7-1.0

Builder UI shows: pre-filled slider, recommended range indicator, tooltip with source, gentle nudge if weight exceeds range. Overall recipe panel shows total weight load assessment and dominance tips.

### Merger — LoRA Baking

```python
class Merger:
    def merge(self, recipe: Recipe) -> Path:
        # 1. Load base model into diffusers
        #    - diffusers format: StableDiffusionPipeline.from_pretrained()
        #    - .safetensors/.ckpt: StableDiffusionPipeline.from_single_file()
        # 2. For each LoRA in recipe (ordered):
        #    - pipe.load_lora_weights()
        #    - pipe.fuse_lora(lora_scale=weight)
        # 3. Save merged pipeline to temp dir
        # Return path to merged diffusers pipeline
```

- Application order matters — users can drag-to-reorder in UI
- Default weight 1.0, user-tunable per LoRA
- `.ckpt` support: `.ckpt` files can contain arbitrary pickled code. The tool shows a security warning before loading `.ckpt` files and requires user confirmation. `.safetensors` is recommended and preferred.

### Converter — diffusers to CoreML

Wraps Apple's `python_coreml_stable_diffusion` tooling.

```python
class ConversionConfig(BaseModel):
    compute_units: str       # "cpuAndGPU" | "all"
    attention: str           # "split_einsum" | "original"
    precision: str           # "float16" | "float32"
    include_safety_checker: bool = False  # Most custom models skip it
    output_dir: Path
    model_name: str

class ConversionResult(BaseModel):
    mlpackage_path: Path
    mlmodelc_path: Path
    manifest_path: Path
    conversion_time: float
    model_size_mb: float
```

Converts each component (text_encoder, unet, vae_decoder, optionally safety_checker), compiles to `.mlmodelc`, generates recipe manifest.

**Mapping to Apple's `python_coreml_stable_diffusion` flags:**
- `compute_units` → `--compute-unit` (e.g., `cpu_and_gpu`, `all`)
- `attention` → `--attention-implementation` (e.g., `SPLIT_EINSUM`, `ORIGINAL`)
- `precision: "float16"` → default `coremltools` precision (`ct.precision.FLOAT16`, the default for Apple Silicon — no extra flag needed). `"float32"` disables float16 conversion.
- Each component converted via `--convert-text-encoder`, `--convert-unet`, `--convert-vae-decoder`
- Compilation via `--bundle-resources-for-swift-cli`

**Disk space:** A full build (base + LoRAs + merged + CoreML output) can consume 10-15 GB. The build pipeline runs a pre-flight disk space check and warns if available space is below 20 GB.

### Recipe Manifest (JSON)

The manifest is a serialized subset of `Recipe` + build metadata. It intentionally trims bulky fields (download URLs, full metadata dicts) to keep it portable and human-readable. The `source` + `id` pair is sufficient to re-fetch any model.

```json
{
  "schema_version": 1,
  "name": "my-custom-model",
  "created": "2026-03-24T...",
  "base_model": {
    "source": "civitai",
    "id": "12345",
    "name": "Realistic Vision V5.1",
    "architecture": "SD1.5"
  },
  "loras": [
    {"source": "civitai", "id": "6789", "name": "Detail Tweaker", "weight": 0.8}
  ],
  "conversion": {
    "compute_units": "all",
    "attention": "split_einsum",
    "precision": "float16",
    "include_safety_checker": false
  },
  "tool_version": "0.1.0"
}
```

## CLI Interface

```
coreml-converter search <query>
    --source hf|civitai|all
    --type checkpoint|lora
    --arch sd1.5|sd2.0
    --limit 20

coreml-converter info <source>:<id>

coreml-converter build
    --base <source>:<id>
    --lora <source>:<id>@<weight>     # repeatable
    --name "my-model"
    --compute-units all
    --attention split_einsum
    --output ./output

coreml-converter build --recipe recipe.json

coreml-converter serve
    --port 8420
    --host 127.0.0.1

coreml-converter cache list
coreml-converter cache clear [<source>:<id>]
```

- `build` with no flags enters interactive mode
- `build --recipe` reproduces from manifest
- Progress bars via `rich`
- `serve` binds localhost only by default

## Web UI

**Tech:** FastAPI, Jinja2, htmx, Alpine.js, Pico CSS.

**Pages:**

1. **Search (`/`)** — search bar, source/type toggles, htmx-loaded results with model cards (name, thumbnail, arch badge, stats, tags). Click checkpoint → builder. Click LoRA → detail panel.

2. **Builder (`/build`)** — left: base model card. Right: compatible LoRA search. Center: recipe list with draggable LoRA cards, each with weight slider (pre-filled with recommendation), range indicator, tooltip, remove button. Bottom: live compatibility report (arch match, conflicts, count indicator, risk). Convert button.

3. **Progress (`/build/{job_id}`)** — step progress (Downloading → Validating → Merging → Converting → Compiling). Per-step progress bars. Live updates via SSE. Completion: output links, manifest, "Build again".

4. **History (`/history`)** — previous builds with manifests. Re-run or download outputs.

No auth — localhost, single user.

## Background Jobs & State

**Execution:**
- `concurrent.futures.ProcessPoolExecutor` with single worker
- In-memory job queue, status persisted to `~/.coreml-converter/jobs.json`
- Progress events via callbacks (SSE for web, `rich` for CLI)

**State files (all in `~/.coreml-converter/`):**
- `cache/` — downloaded models
- `builds.json` — all build records (`BuildRecord` list), from pending through completed/failed. Single source of truth for both in-flight and historical builds. `BuildRecord.id` is a UUID4.
- `config.json` — user preferences: default compute units, default attention, output directory, CivitAI API key. Schema defined by a `Config` Pydantic model with defaults.

No database. All JSON files include a `schema_version` field for forward-compatible migrations. Sufficient for single-user local tool.

**Concurrency:** The CLI and web server are not designed for simultaneous use. `builds.json` uses file locking (`fcntl.flock`) as a safety measure, but the expected usage is either CLI or `serve`, not both at once.

**Error handling:**
- Downloads: retry 3x with backoff
- Merge failures: report which LoRA caused shape mismatch
- Conversion failures: surface `coremltools` errors with context
- All errors persisted to job history

## Key Dependencies

- `diffusers` — model loading, LoRA merging
- `coremltools` — CoreML conversion
- `python_coreml_stable_diffusion` (Apple) — SD-specific conversion pipeline
- `huggingface_hub` — HF API client
- `httpx` — CivitAI API client
- `click` — CLI framework
- `rich` — CLI progress/formatting
- `fastapi` + `uvicorn` — web server
- `jinja2` — templates
- `pydantic` — data models
- `torch` + `safetensors` — weight loading

## Scope

**In scope (v1):**
- SD1.5 and SD2.0 checkpoints
- LoRA merging with weight control and guidance
- HuggingFace + CivitAI search and download
- Compatibility validation (metadata + weight-based)
- Conflict detection (tag-based + weight overlap)
- CoreML conversion optimized for Apple Silicon
- CLI + web UI
- Recipe manifests for reproducibility

**Out of scope (future):**
- SDXL support
- Textual TUI
- Hosted/cloud conversion service
- Textual inversions / hypernetworks
- Batch conversion
- Recipe sharing marketplace
