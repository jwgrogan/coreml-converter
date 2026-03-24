import click

@click.command()
@click.argument("model_ref")
def info(model_ref):
    """Show details for a model (e.g., civitai:12345)."""
    click.echo(f"Info for {model_ref} — not yet implemented")
