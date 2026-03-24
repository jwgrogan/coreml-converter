import click
import coreml_converter
from coreml_converter.cli.commands.search import search
from coreml_converter.cli.commands.info import info
from coreml_converter.cli.commands.build import build
from coreml_converter.cli.commands.serve import serve
from coreml_converter.cli.commands.cache import cache

@click.group()
@click.version_option(version=coreml_converter.__version__, prog_name="CoreML Converter")
def cli():
    """CoreML Converter - Convert SD models + LoRAs to CoreML for Apple Silicon."""
    pass

cli.add_command(search)
cli.add_command(info)
cli.add_command(build)
cli.add_command(serve)
cli.add_command(cache)
