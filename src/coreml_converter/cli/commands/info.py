from __future__ import annotations
import click
from coreml_converter.cli.formatting import console
from coreml_converter.core.models import ModelSource

def _parse_ref(ref):
    if ":" not in ref:
        raise click.BadParameter("Expected format: source:id (e.g., civitai:12345)")
    source_str, model_id = ref.split(":", 1)
    source_map = {"hf": ModelSource.HUGGINGFACE, "civitai": ModelSource.CIVITAI}
    source = source_map.get(source_str.lower())
    if source is None:
        raise click.BadParameter(f"Unknown source '{source_str}'. Use 'hf' or 'civitai'.")
    return source, model_id

@click.command()
@click.argument("model_ref")
def info(model_ref):
    """Show details for a model (e.g., civitai:12345)."""
    from coreml_converter.cli.commands.search import get_registry
    source, model_id = _parse_ref(model_ref)
    registry = get_registry()
    model = registry.get_by_id(source, model_id)
    if not model:
        console.print(f"[red]Model not found: {model_ref}[/red]")
        return
    console.print(f"[bold]{model.name}[/bold]")
    console.print(f"  Source:       {model.source.value}")
    console.print(f"  ID:           {model.id}")
    console.print(f"  Architecture: {model.base_architecture.value}")
    console.print(f"  Type:         {model.model_type.value}")
    console.print(f"  Tags:         {', '.join(model.tags)}")
    if model.metadata.get("download_count"):
        console.print(f"  Downloads:    {model.metadata['download_count']:,}")
    if model.metadata.get("description"):
        console.print(f"  Description:  {model.metadata['description'][:200]}")
