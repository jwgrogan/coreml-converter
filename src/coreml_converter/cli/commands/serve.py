import click

@click.command()
@click.option("--port", default=8420, type=int)
@click.option("--host", default="127.0.0.1")
def serve(port, host):
    """Start the web UI."""
    click.echo(f"Starting web UI on {host}:{port} — not yet implemented")
