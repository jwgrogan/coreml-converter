import click

@click.command()
@click.option("--base", default=None)
@click.option("--lora", multiple=True)
@click.option("--name", default=None)
@click.option("--recipe", default=None, type=click.Path(exists=True))
@click.option("--compute-units", default="all")
@click.option("--attention", default="split_einsum")
@click.option("--output", default="./output", type=click.Path())
def build(base, lora, name, recipe, compute_units, attention, output):
    """Build a CoreML model from base + LoRAs."""
    click.echo("Build command — not yet implemented")
