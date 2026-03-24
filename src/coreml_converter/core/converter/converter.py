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
        raise RuntimeError(f"Insufficient disk space: {available_gb:.1f} GB available, {required_gb:.1f} GB required")


class Converter:
    def convert(self, merged_model_path: Path, recipe: Recipe,
                progress_callback: Callable[[str, float], None] | None = None) -> ConversionResult:
        config = recipe.conversion_config
        output_dir = config.output_dir / config.model_name
        output_dir.mkdir(parents=True, exist_ok=True)

        def _report(msg, pct):
            if progress_callback:
                progress_callback(msg, pct)

        check_disk_space(config.output_dir)
        _report("Converting to CoreML", 0.1)
        start_time = time.monotonic()

        cmd = [
            "python", "-m", "python_coreml_stable_diffusion.torch2coreml",
            "--model-version", str(merged_model_path),
            "-o", str(output_dir),
            "--convert-unet", "--convert-text-encoder", "--convert-vae-decoder",
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
        mlpackage_path = output_dir / f"{config.model_name}.mlpackage"
        mlmodelc_path = output_dir / f"{config.model_name}.mlmodelc"
        manifest_path = output_dir / "manifest.json"

        _report("Writing manifest", 0.95)
        self._write_manifest(recipe, manifest_path)

        model_size_mb = 0.0
        if mlmodelc_path.exists():
            model_size_mb = sum(f.stat().st_size for f in mlmodelc_path.rglob("*") if f.is_file()) / (1024 ** 2)

        _report("Conversion complete", 1.0)
        return ConversionResult(
            mlpackage_path=mlpackage_path, mlmodelc_path=mlmodelc_path,
            manifest_path=manifest_path, conversion_time=elapsed, model_size_mb=model_size_mb,
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
                {"source": e.model.source.value, "id": e.model.id, "name": e.model.name, "weight": e.weight}
                for e in recipe.loras
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
