"""Registry of locally-provided base models and LoRAs.

Models can arrive two ways: copied in through a multipart upload, or
registered by absolute path without copying (the local-first fast path Fanny
uses, since checkpoints run 2-7GB and both processes share a filesystem).

The store is shared by the HTML builder UI and the JSON API so a model
registered through either surface resolves in a build started from the other.
It is in-memory and therefore per-process: refs do not survive a restart,
which is fine because a build request follows registration immediately.
"""
from __future__ import annotations

import shutil
import uuid
from pathlib import Path
from typing import BinaryIO

from coreml_converter.core.models import (
    BaseArchitecture,
    ModelInfo,
    ModelSource,
    ModelType,
)

ALLOWED_EXTENSIONS = {".safetensors", ".ckpt", ".bin"}

_ARCH_MAP = {"SD1.5": BaseArchitecture.SD15, "SD2.0": BaseArchitecture.SD20}

_local_models: dict[str, ModelInfo] = {}


def get(model_ref: str) -> ModelInfo | None:
    """Look up a previously registered local model by its ref."""
    return _local_models.get(model_ref)


def clear() -> None:
    """Drop all registrations (used by tests)."""
    _local_models.clear()


def _validate_extension(filename: str) -> str:
    ext = Path(filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise ValueError(
            f"Unsupported format: {ext or '(none)'}. "
            f"Use {', '.join(sorted(ALLOWED_EXTENSIONS))}"
        )
    return ext


def _register(
    path: Path,
    name: str,
    model_type: str,
    arch: str,
    *,
    uploaded: bool,
    local_id: str,
) -> ModelInfo:
    model = ModelInfo(
        # Source is structurally required but meaningless for local files;
        # the build path keys off metadata["local_path"] and never downloads.
        source=ModelSource.CIVITAI,
        id=local_id,
        name=name,
        base_architecture=_ARCH_MAP.get(arch, BaseArchitecture.SD15),
        model_type=(
            ModelType.CHECKPOINT if model_type == "checkpoint" else ModelType.LORA
        ),
        tags=["uploaded"] if uploaded else ["local"],
        download_url="",
        metadata={"local_path": str(path), "uploaded": uploaded},
    )
    _local_models[local_id] = model
    return model


def register_path(
    path: str | Path,
    model_type: str = "checkpoint",
    arch: str = "SD1.5",
    name: str | None = None,
) -> ModelInfo:
    """Register an existing on-disk checkpoint without copying it.

    `name` overrides the label derived from the filename. Callers that store
    files under generated names — Fanny prefixes each with an id to keep
    same-named imports apart — pass the name the user actually sees, so it is
    that name which ends up in the build manifest.

    Raises ValueError for a disallowed extension and FileNotFoundError if the
    path is not a readable file.
    """
    resolved = Path(path).expanduser()
    _validate_extension(resolved.name)
    # Resolve after the extension check so the error message names the input.
    resolved = resolved.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"No such file: {resolved}")

    local_id = f"local_{uuid.uuid4().hex[:8]}"
    return _register(
        resolved,
        name or resolved.stem,
        model_type,
        arch,
        uploaded=False,
        local_id=local_id,
    )


def store_upload(
    fileobj: BinaryIO,
    filename: str,
    cache_dir: Path,
    model_type: str = "checkpoint",
    arch: str = "SD1.5",
) -> ModelInfo:
    """Copy an uploaded file into the cache and register it."""
    _validate_extension(filename)

    cache_dir.mkdir(parents=True, exist_ok=True)
    upload_id = uuid.uuid4().hex[:8]
    dest = cache_dir / f"{upload_id}_{filename}"

    with open(dest, "wb") as out:
        shutil.copyfileobj(fileobj, out)

    return _register(
        dest,
        Path(filename).stem,
        model_type,
        arch,
        uploaded=True,
        local_id=f"local_{upload_id}",
    )
