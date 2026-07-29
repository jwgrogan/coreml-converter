# CoreML Converter

Convert Stable Diffusion 1.5/2.0 checkpoints + LoRAs to CoreML models optimized for Apple Silicon.

## Features

- Search HuggingFace and CivitAI for base models and LoRAs
- Compatibility checking: architecture validation, conflict detection, weight guidance
- Merge multiple LoRAs into a base model with configurable weights
- Convert to CoreML via Apple's official `ml-stable-diffusion` converter, producing the exact `.mlmodelc` layout its Swift runtime (and Fanny Server) require
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
# Apple's Stable Diffusion converter is git-only (NOT on PyPI) and is required
# for the CoreML conversion step:
pip install git+https://github.com/apple/ml-stable-diffusion.git
```

> **Why the extra git install?** `coremltools` is on PyPI, but Apple's
> SD-specific converter (`python_coreml_stable_diffusion` / `torch2coreml`) is
> only distributed from the `apple/ml-stable-diffusion` GitHub repo. This tool
> delegates the actual PyTorch→CoreML conversion to it so the output matches
> Apple's tensor layout (batch-2 sample, rank-4 `encoder_hidden_states`,
> SPLIT_EINSUM attention, UNet chunking). Models converted any other way will
> fail to load in Apple's Swift pipeline. If it's missing, the converter fails
> fast with this install hint.

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
