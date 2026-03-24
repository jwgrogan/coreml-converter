from __future__ import annotations
import click
from coreml_converter.cli.formatting import console
from coreml_converter.core.config import get_app_dir, load_config, save_config

@click.group()
def config():
    """Manage configuration."""
    pass

@config.command("set")
@click.argument("key")
@click.argument("value")
def config_set(key, value):
    """Set a config value (e.g., config set civitai-key YOUR_KEY)."""
    app_dir = get_app_dir()
    cfg = load_config(app_dir / "config.json")
    key_map = {"civitai-key": "civitai_api_key", "compute-units": "compute_units",
               "attention": "attention", "output-dir": "output_dir"}
    field = key_map.get(key)
    if field is None:
        console.print(f"[red]Unknown key: {key}. Valid keys: {', '.join(key_map.keys())}[/red]")
        return
    setattr(cfg, field, value)
    save_config(cfg, app_dir / "config.json")
    console.print(f"[green]Set {key} = {value}[/green]")

@config.command("get")
@click.argument("key", required=False)
def config_get(key=None):
    """Show config values."""
    app_dir = get_app_dir()
    cfg = load_config(app_dir / "config.json")
    if key:
        key_map = {"civitai-key": "civitai_api_key", "compute-units": "compute_units",
                   "attention": "attention", "output-dir": "output_dir"}
        field = key_map.get(key)
        if field:
            val = getattr(cfg, field, None)
            if "key" in key and val:
                val = val[:4] + "..." + val[-4:]
            console.print(f"{key} = {val}")
        else:
            console.print(f"[red]Unknown key: {key}[/red]")
    else:
        console.print(f"compute-units = {cfg.compute_units}")
        console.print(f"attention     = {cfg.attention}")
        console.print(f"output-dir    = {cfg.output_dir}")
        key_display = (cfg.civitai_api_key[:4] + "..." + cfg.civitai_api_key[-4:]) if cfg.civitai_api_key else "not set"
        console.print(f"civitai-key   = {key_display}")
