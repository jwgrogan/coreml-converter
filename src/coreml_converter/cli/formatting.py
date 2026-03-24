from __future__ import annotations
from rich.console import Console
from rich.table import Table
from coreml_converter.core.models import ModelInfo

console = Console()

def print_model_table(models: list[ModelInfo]) -> None:
    table = Table(title="Search Results")
    table.add_column("Source", style="cyan")
    table.add_column("ID", style="dim")
    table.add_column("Name", style="bold")
    table.add_column("Arch", style="green")
    table.add_column("Type", style="yellow")
    table.add_column("Tags")
    for m in models:
        table.add_row(m.source.value, m.id, m.name, m.base_architecture.value,
                      m.model_type.value, ", ".join(m.tags[:5]))
    console.print(table)
