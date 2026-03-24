# CoreML Converter — Design Spec

## Overview

A Python tool (CLI + local web UI) that converts Stable Diffusion 1.5/2.0 checkpoints with merged LoRAs into CoreML models optimized for Apple Silicon. Searches HuggingFace and CivitAI for models, validates compatibility, guides users through building custom recipes, and outputs compiled `.mlmodelc` bundles ready for use with Apple's `ml-stable-diffusion` Swift package.

**Distribution:** Open source, installed via `pip` / Homebrew.
**Target user:** Developers building Swift apps with `ml-stable-diffusion` who want custom models from the broader HF/CivitAI ecosystem.

## Architecture

**Approach: Pipeline Architecture** — four layers with clean interfaces:

1. **Core library** (`coreml_converter.core`) — model registry, compatibility analyzer, LoRA merger, CoreML converter
2. **CLI** (`coreml_converter.cli`) — Click-based, thin wrapper over core
3. **Web UI** (`coreml_converter.web`) — FastAPI + Jinja2 + htmx/Alpine.js
4. **TUI** (`coreml_converter.tui`) — Textual app (stretch goal)

All interfaces share the core library. No duplication of business logic.

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
│       └── tui/                # Stretch goal
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

**Shared abstractions:**

```python
class ModelSource(Enum):
    HUGGINGFACE = "huggingface"
    CIVITAI = "civitai"

class ModelInfo:
    source: ModelSource
    id: str
    name: str
    base_architecture: str  # "SD1.5" | "SD2.0"
    model_type: str         # "checkpoint" | "lora"
    tags: list[str]
    download_url: str
    metadata: dict

class Registry:
    async def search(query, source, model_type, base_arch) -> list[ModelInfo]
    async def get_compatible_loras(base_model: ModelInfo) -> list[ModelInfo]
    async def download(model: ModelInfo, dest: Path) -> Path
```

**Local cache:** `~/.coreml-converter/cache/` keyed by source + ID. CLI `cache clear` to manage.

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
- Sub-second operation, just tensor math

**LoRA count warnings:**
- 1-3: green
- 4-5: soft warning ("quality may degrade")
- 6+: hard warning ("likely artifacts, proceed at own risk")

**Output:**

```python
class CompatibilityReport:
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
        #    - diffusers format: load directly
        #    - .safetensors/.ckpt: convert to diffusers first
        # 2. For each LoRA in recipe (ordered):
        #    - pipe.load_lora_weights()
        #    - pipe.fuse_lora(lora_scale=weight)
        # 3. Save merged pipeline to temp dir
        # Return path to merged diffusers pipeline
```

- Application order matters — users can drag-to-reorder in UI
- Default weight 1.0, user-tunable per LoRA

### Converter — diffusers to CoreML

Wraps Apple's `python_coreml_stable_diffusion` tooling.

```python
class ConversionConfig:
    compute_units: str       # "cpuAndGPU" | "all"
    attention: str           # "split_einsum" | "original"
    precision: str           # "float16" | "float32"
    output_dir: Path
    model_name: str

class ConversionResult:
    mlpackage_path: Path
    mlmodelc_path: Path
    manifest_path: Path
    conversion_time: float
    model_size_mb: float
```

Converts each component (text_encoder, unet, vae_decoder, safety_checker), compiles to `.mlmodelc`, generates recipe manifest.

### Recipe Manifest (JSON)

```json
{
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
    "precision": "float16"
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
- `history.json` — build history
- `jobs.json` — in-flight job state
- `config.json` — user preferences

No database. All JSON. Sufficient for single-user local tool.

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
