"""ccml start — bootstrap venv, install deps, and launch."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import click

from coreml_converter.cli.formatting import console


def _find_python() -> str | None:
    """Find a suitable Python 3.10-3.12 for the venv."""
    for ver in ["python3.12", "python3.11", "python3.10"]:
        try:
            result = subprocess.run([ver, "--version"], capture_output=True, text=True)
            if result.returncode == 0:
                return ver
        except FileNotFoundError:
            continue
    return None


def _get_project_root() -> Path:
    """Walk up from this file to find pyproject.toml."""
    p = Path(__file__).resolve()
    for parent in [p] + list(p.parents):
        if (parent / "pyproject.toml").exists():
            return parent
    return Path.cwd()


@click.command()
@click.option("--port", default=8420, type=int, help="Port for web UI")
@click.option("--no-serve", is_flag=True, help="Set up venv but don't launch web UI")
def start(port: int, no_serve: bool):
    """Bootstrap environment and launch CoreML Converter."""
    project_root = _get_project_root()
    venv_dir = project_root / ".venv"
    venv_python = venv_dir / "bin" / "python3"
    venv_pip = venv_dir / "bin" / "pip"
    venv_ccml = venv_dir / "bin" / "ccml"

    console.print()
    console.print("[bold green]  ██████╗ ██████╗███╗   ███╗██╗     [/bold green]")
    console.print("[bold green] ██╔════╝██╔════╝████╗ ████║██║     [/bold green]")
    console.print("[bold green] ██║     ██║     ██╔████╔██║██║     [/bold green]")
    console.print("[bold green] ██║     ██║     ██║╚██╔╝██║██║     [/bold green]")
    console.print("[bold green] ╚██████╗╚██████╗██║ ╚═╝ ██║███████╗[/bold green]")
    console.print("[bold green]  ╚═════╝ ╚═════╝╚═╝     ╚═╝╚══════╝[/bold green]")
    console.print()
    console.print("[bold]CoreML Converter[/bold] — SD 1.5/2.0 + LoRAs → CoreML")
    console.print()

    # Step 1: Check/create venv
    if venv_python.exists():
        console.print("[green]✓[/green] Virtual environment found")
    else:
        py = _find_python()
        if not py:
            console.print("[red]✗ No Python 3.10-3.12 found.[/red]")
            console.print("  Install via: brew install python@3.12")
            sys.exit(1)

        console.print(f"[yellow]→[/yellow] Creating virtual environment with {py}...")
        subprocess.run([py, "-m", "venv", str(venv_dir)], check=True)
        console.print("[green]✓[/green] Virtual environment created")

    # Step 2: Install base package
    console.print("[yellow]→[/yellow] Installing base dependencies...")
    subprocess.run(
        [str(venv_pip), "install", "--quiet", "--upgrade", "pip"],
        capture_output=True,
    )
    result = subprocess.run(
        [str(venv_pip), "install", "--quiet", "-e", f"{project_root}[dev]"],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        console.print(f"[red]✗ Base install failed:[/red] {result.stderr[:200]}")
        sys.exit(1)
    console.print("[green]✓[/green] Base dependencies installed")

    # Step 3: Check ML deps
    ml_check = subprocess.run(
        [str(venv_python), "-c", "import torch; import diffusers; import coremltools"],
        capture_output=True,
    )
    if ml_check.returncode == 0:
        console.print("[green]✓[/green] ML dependencies installed (torch, diffusers, coremltools)")
    else:
        console.print("[yellow]→[/yellow] Installing ML dependencies (this may take a few minutes)...")
        # Install individually to avoid the apple git dep build issues
        ml_result = subprocess.run(
            [str(venv_pip), "install", "--quiet",
             "torch>=2.0", "diffusers>=0.25", "transformers>=4.35",
             "safetensors>=0.4", "coremltools>=7.0"],
            capture_output=True, text=True,
        )
        if ml_result.returncode != 0:
            console.print(f"[red]✗ ML install failed:[/red] {ml_result.stderr[:300]}")
            console.print("[yellow]  You can still search/browse without ML deps.[/yellow]")
        else:
            console.print("[green]✓[/green] ML dependencies installed")

        # Apple's ml-stable-diffusion needs --no-build-isolation
        apple_check = subprocess.run(
            [str(venv_python), "-c", "import python_coreml_stable_diffusion"],
            capture_output=True,
        )
        if apple_check.returncode != 0:
            console.print("[yellow]→[/yellow] Installing Apple ml-stable-diffusion...")
            apple_result = subprocess.run(
                [str(venv_pip), "install", "--quiet", "--no-build-isolation", "--no-deps",
                 "git+https://github.com/apple/ml-stable-diffusion.git"],
                capture_output=True, text=True,
            )
            if apple_result.returncode == 0:
                console.print("[green]✓[/green] Apple ml-stable-diffusion installed")
            else:
                console.print("[yellow]⚠ Apple ml-stable-diffusion failed (conversion may not work)[/yellow]")

    # Step 4: Check config
    from coreml_converter.core.config import get_app_dir, load_config
    app_dir = get_app_dir()
    config = load_config(app_dir / "config.json")

    console.print()
    if config.civitai_api_key:
        console.print(f"[green]✓[/green] CivitAI API key: configured")
    else:
        console.print(f"[yellow]⚠[/yellow] CivitAI API key: not set")
        console.print(f"  Run: [bold]ccml config set civitai-key YOUR_KEY[/bold]")

    # Step 5: Launch
    console.print()
    if no_serve:
        console.print("[green]Setup complete![/green] Run commands with:")
        console.print(f"  source {venv_dir}/bin/activate")
        console.print(f"  ccml serve")
        console.print(f"  ccml build --base /path/to/model.safetensors")
    else:
        console.print(f"[green bold]Launching web UI...[/green bold]")
        console.print(f"  http://127.0.0.1:{port}")
        console.print()
        console.print("  [dim]Press Ctrl+C to stop[/dim]")
        console.print()

        # Exec into the venv's ccml serve
        os.execv(
            str(venv_ccml),
            [str(venv_ccml), "serve", "--port", str(port)],
        )
