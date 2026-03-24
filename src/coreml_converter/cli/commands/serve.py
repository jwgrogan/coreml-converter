from __future__ import annotations

import click
import uvicorn

from coreml_converter.cli.formatting import console


@click.command()
@click.option("--port", default=8420, type=int, help="Port to listen on")
@click.option("--host", default="127.0.0.1", help="Host to bind to")
def serve(port: int, host: str):
    """Start the web UI."""
    from coreml_converter.core.config import get_app_dir, load_config
    from coreml_converter.core.registry import Registry
    from coreml_converter.core.registry.huggingface import HuggingFaceClient
    from coreml_converter.core.registry.civitai import CivitAIClient
    from coreml_converter.core.state import BuildStore
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
    app.state.job_manager = JobManager(cache_dir=app_dir / "cache", build_store=build_store)

    console.print(f"[green]Starting CoreML Converter web UI[/green]")
    console.print(f"  http://{host}:{port}")
    uvicorn.run(app, host=host, port=port, log_level="info")
