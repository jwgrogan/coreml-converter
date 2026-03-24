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

def _is_local_path(ref: str) -> bool:
    """Check if ref is a local file path rather than a source:id reference."""
    p = Path(ref)
    return p.exists() and p.is_file()


def _make_local_model(path: str, model_type: ModelType, arch: BaseArchitecture = BaseArchitecture.SD15) -> ModelInfo:
    """Create a ModelInfo from a local file path."""
    p = Path(path)
    return ModelInfo(
        source=ModelSource.CIVITAI,  # placeholder
        id=f"local_{p.stem}",
        name=p.stem,
        base_architecture=arch,
        model_type=model_type,
        tags=["local"],
        download_url="",
        metadata={"local_path": str(p.resolve()), "uploaded": True},
    )


def _parse_model_ref(ref):
    if ":" not in ref:
        raise click.BadParameter(f"Invalid model ref '{ref}'. Expected format: source:id or a local file path")
    source_str, model_id = ref.split(":", 1)
    source_map = {"hf": ModelSource.HUGGINGFACE, "civitai": ModelSource.CIVITAI}
    source = source_map.get(source_str.lower())
    if source is None:
        raise click.BadParameter(f"Unknown source '{source_str}'. Use 'hf' or 'civitai'.")
    return source, model_id

def _parse_lora_ref(ref):
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
    from coreml_converter.core.ml_check import check_ml_deps
    ok, missing = check_ml_deps()
    if not ok:
        console.print(f"[red]Missing ML dependencies: {', '.join(missing)}[/red]")
        console.print("[yellow]Run: ccml start   (sets up venv with all dependencies)[/yellow]")
        console.print("[yellow]Or:  source .venv/bin/activate && pip install -e '.[ml]'[/yellow]")
        sys.exit(1)

    app_dir = get_app_dir()
    config = load_config(app_dir / "config.json")

    if recipe:
        manifest = json.loads(Path(recipe).read_text())
        console.print(f"[green]Rebuilding from recipe:[/green] {manifest['name']}")
        base_data = manifest["base_model"]
        base_model = ModelInfo(source=ModelSource(base_data["source"]), id=base_data["id"],
            name=base_data["name"], base_architecture=BaseArchitecture(base_data["architecture"]),
            model_type=ModelType.CHECKPOINT, tags=[], download_url="", metadata={})
        lora_entries = []
        for l in manifest.get("loras", []):
            lora_model = ModelInfo(source=ModelSource(l["source"]), id=l["id"], name=l["name"],
                base_architecture=base_model.base_architecture, model_type=ModelType.LORA,
                tags=[], download_url="", metadata={})
            lora_entries.append(LoRAEntry(model=lora_model, weight=l["weight"]))
        conv_data = manifest.get("conversion", {})
        conv_config = ConversionConfig(output_dir=Path(output), model_name=manifest["name"],
            compute_units=conv_data.get("compute_units", "all"),
            attention=conv_data.get("attention", "split_einsum"),
            precision=conv_data.get("precision", "float16"),
            include_safety_checker=conv_data.get("include_safety_checker", False))
        build_recipe = Recipe(name=manifest["name"], base_model=base_model,
            loras=lora_entries, conversion_config=conv_config)
    elif base:
        # Support local file paths: --base /path/to/model.safetensors
        if _is_local_path(base):
            base_model = _make_local_model(base, ModelType.CHECKPOINT)
            console.print(f"[green]Using local base model:[/green] {base}")
        else:
            source, model_id = _parse_model_ref(base)
            from coreml_converter.cli.commands.search import get_registry
            registry = get_registry()
            results = registry.search(model_id, source=source, model_type=ModelType.CHECKPOINT, limit=1)
            if not results:
                console.print(f"[red]Model not found: {base}[/red]")
                sys.exit(1)
            base_model = results[0]

        lora_entries = []
        for lora_ref in lora:
            # Support local paths: --lora /path/to/lora.safetensors@0.7
            weight = 1.0
            ref_part = lora_ref
            if "@" in lora_ref:
                ref_part, weight_str = lora_ref.rsplit("@", 1)
                weight = float(weight_str)

            if _is_local_path(ref_part):
                lora_model = _make_local_model(ref_part, ModelType.LORA, base_model.base_architecture)
                lora_entries.append(LoRAEntry(model=lora_model, weight=weight))
                console.print(f"[green]Using local LoRA:[/green] {ref_part} @ {weight}")
            else:
                if not hasattr(locals(), 'registry'):
                    from coreml_converter.cli.commands.search import get_registry
                    registry = get_registry()
                l_source, l_id, l_weight = _parse_lora_ref(lora_ref)
                lora_results = registry.search(l_id, source=l_source, model_type=ModelType.LORA, limit=1)
                if not lora_results:
                    console.print(f"[red]LoRA not found: {lora_ref}[/red]")
                    sys.exit(1)
                lora_entries.append(LoRAEntry(model=lora_results[0], weight=l_weight))
        model_name = name or f"{base_model.name}-custom"
        conv_config = ConversionConfig(output_dir=Path(output), model_name=model_name,
            compute_units=compute_units, attention=attention)
        build_recipe = Recipe(name=model_name, base_model=base_model,
            loras=lora_entries, conversion_config=conv_config)
    else:
        console.print("[yellow]Interactive build mode not yet implemented. Use --base or --recipe.[/yellow]")
        sys.exit(1)

    from coreml_converter.core.analyzer import check_compatibility, detect_tag_conflicts
    report = check_compatibility(base_model, list(build_recipe.loras))
    if not report.is_compatible:
        console.print("[red]Compatibility check failed:[/red]")
        for c in report.conflicts:
            console.print(f"  - {c.reason}")
        if not click.confirm("Continue anyway?"):
            sys.exit(1)
    if report.lora_count_warning:
        console.print(f"[yellow]Warning: {report.lora_count_warning}[/yellow]")
    tag_conflicts = detect_tag_conflicts(build_recipe.loras)
    for c in tag_conflicts:
        console.print(f"[yellow]Conflict: {c.reason} ({c.severity.value})[/yellow]")

    store = BuildStore(app_dir / "builds.json")
    record = BuildRecord(recipe=build_recipe)
    store.save(record)

    # Pre-flight disk space check
    from coreml_converter.core.converter.converter import check_disk_space
    try:
        check_disk_space(Path(output))
    except RuntimeError as e:
        console.print(f"[red]{e}[/red]")
        if not click.confirm("Continue anyway?"):
            sys.exit(1)

    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console) as progress:
        task = progress.add_task("Starting build...", total=None)
        cache_dir = app_dir / "cache"
        try:
            # Download (skip for local files)
            if not build_recipe.base_model.metadata.get("local_path"):
                progress.update(task, description="Downloading base model...")
                if 'registry' not in dir():
                    from coreml_converter.cli.commands.search import get_registry
                    registry = get_registry()
                base_path = registry.download(build_recipe.base_model, cache_dir)
                build_recipe.base_model.metadata["local_path"] = str(base_path)
            else:
                progress.update(task, description="Using local base model...")

            for entry in build_recipe.loras:
                if not entry.model.metadata.get("local_path"):
                    progress.update(task, description=f"Downloading LoRA: {entry.model.name}...")
                    if 'registry' not in dir():
                        from coreml_converter.cli.commands.search import get_registry
                        registry = get_registry()
                    lora_path = registry.download(entry.model, cache_dir)
                    entry.model.metadata["local_path"] = str(lora_path)
                else:
                    progress.update(task, description=f"Using local LoRA: {entry.model.name}...")

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
            progress.update(task, description="Merging LoRAs into base model...")
            from coreml_converter.core.merger.merger import Merger
            merger = Merger()
            merged_path = merger.merge(build_recipe, cache_dir, Path(output))
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
