from __future__ import annotations

import click
import uvicorn

from coreml_converter.cli.formatting import console
from coreml_converter.core.config import DEFAULT_PORT


@click.command()
@click.option("--port", default=DEFAULT_PORT, type=int, help="Port to listen on")
@click.option("--host", default="127.0.0.1", help="Host to bind to")
def serve(port: int, host: str):
    """Start the web UI."""
    import logging

    # Builds run in a worker thread and log their progress — including every
    # line Apple's converter writes — through the stdlib logger. Without a
    # handler on our own package's logger, a failed conversion reports only
    # "exited with code 1" and the traceback that explains it is discarded.
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s:%(name)s:%(message)s",
    )
    logging.getLogger("coreml_converter").setLevel(logging.INFO)

    from coreml_converter.core.config import get_app_dir, load_config
    from coreml_converter.core.registry import Registry
    from coreml_converter.core.registry.huggingface import HuggingFaceClient
    from coreml_converter.core.registry.civitai import CivitAIClient
    from coreml_converter.core.state import BuildStore
    from coreml_converter.core.favorites import FavoritesStore
    from coreml_converter.web.app import create_app
    from coreml_converter.web.jobs import JobManager

    app_dir = get_app_dir()
    config = load_config(app_dir / "config.json")

    app = create_app()

    app.state.registry = Registry(
        hf_client=HuggingFaceClient(),
        civitai_client=CivitAIClient(api_key=config.civitai_api_key),
    )
    build_store = BuildStore(app_dir / "builds.json")
    app.state.build_store = build_store

    # Builds run in-process, so anything still marked running belongs to a
    # process that no longer exists — otherwise it stays "running" in the
    # user's history forever.
    interrupted = build_store.fail_interrupted()
    if interrupted:
        console.print(f"  Marked {interrupted} interrupted build(s) as failed")
    app.state.job_manager = JobManager(cache_dir=app_dir / "cache", build_store=build_store)
    app.state.favorites = FavoritesStore(app_dir / "favorites.json")

    # Clear scratch dirs stranded by builds that were killed rather than
    # failing cleanly; each can be tens of GB.
    from pathlib import Path
    from coreml_converter.core.converter.converter import sweep_stale_scratch_dirs
    swept = sweep_stale_scratch_dirs(Path(config.output_dir).expanduser())
    if swept:
        console.print(f"  Cleaned {len(swept)} stale build scratch dir(s)")

    # Check ML deps and warn (search/browse works without them, but builds won't)
    from coreml_converter.core.ml_check import check_ml_deps
    ok, missing = check_ml_deps()

    console.print(f"[green bold]CoreML Converter[/green bold] v{__import__('coreml_converter').__version__}")
    console.print(f"  Web UI: http://{host}:{port}")
    if ok:
        console.print(f"  ML deps: [green]all installed[/green] (builds will work)")
    else:
        console.print(f"  ML deps: [yellow]missing {', '.join(missing)}[/yellow]")
        console.print(f"           Search/browse works, but builds require ML deps.")
        console.print(f"           Run: [bold]ccml start[/bold] to set up everything.")
    uvicorn.run(app, host=host, port=port, log_level="info")
