import click

@click.group()
def cache():
    """Manage the model cache."""
    pass

@cache.command("list")
def cache_list():
    """List cached models."""
    click.echo("Cache list — not yet implemented")

@cache.command("clear")
@click.argument("model_ref", required=False)
def cache_clear(model_ref=None):
    """Clear cached models."""
    click.echo("Cache clear — not yet implemented")
