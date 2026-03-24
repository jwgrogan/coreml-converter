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
    return Registry(hf_client=HuggingFaceClient(), civitai_client=CivitAIClient(api_key=config.civitai_api_key))

_SOURCE_MAP = {"hf": ModelSource.HUGGINGFACE, "civitai": ModelSource.CIVITAI}
_TYPE_MAP = {"checkpoint": ModelType.CHECKPOINT, "lora": ModelType.LORA}
_ARCH_MAP = {"sd1.5": BaseArchitecture.SD15, "sd2.0": BaseArchitecture.SD20}

@click.command()
@click.argument("query")
@click.option("--source", type=click.Choice(["hf", "civitai", "all"]), default="all")
@click.option("--type", "model_type", type=click.Choice(["checkpoint", "lora"]), default=None)
@click.option("--arch", type=click.Choice(["sd1.5", "sd2.0"]), default=None)
@click.option("--limit", default=20, type=int)
def search(query, source, model_type, arch, limit):
    """Search HuggingFace and CivitAI for models."""
    registry = get_registry()
    results = registry.search(query=query,
        source=_SOURCE_MAP.get(source) if source != "all" else None,
        model_type=_TYPE_MAP.get(model_type) if model_type else None,
        base_arch=_ARCH_MAP.get(arch) if arch else None, limit=limit)
    if not results:
        console.print("[yellow]No results found.[/yellow]")
        return
    print_model_table(results)
