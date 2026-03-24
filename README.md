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

```bash
pip install coreml-converter
# For ML dependencies (torch, diffusers, coremltools):
pip install coreml-converter[ml]
```

## Quick Start

```bash
# Search for models
coreml-converter search "realistic vision" --type checkpoint

# Build a model
coreml-converter build --base civitai:4201 --lora civitai:6789@0.7 --name my-model

# Start the web UI
coreml-converter serve
```

## License

MIT
