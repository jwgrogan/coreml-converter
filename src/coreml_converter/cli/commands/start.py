"""ccml start — bootstrap venv, install deps, and launch."""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import click

from coreml_converter.cli.formatting import console
from coreml_converter.core.config import DEFAULT_PORT


SUPPORTED_PYTHONS = ["3.12", "3.11", "3.10"]

# Apple's converter is required for builds but cannot be resolved normally:
# its metadata pins numpy<1.24, which has no wheels for Python 3.11+ and
# cannot be compiled there. We install it without dependencies and satisfy
# them from our own pinned [ml] extra instead.
APPLE_CONVERTER_URL = "git+https://github.com/apple/ml-stable-diffusion.git"


def _uv_python(version: str) -> str | None:
    """Ask uv for an interpreter it manages, if uv is installed."""
    if shutil.which("uv") is None:
        return None
    result = subprocess.run(
        ["uv", "python", "find", version], capture_output=True, text=True
    )
    if result.returncode != 0:
        return None
    path = result.stdout.strip()
    return path if path and Path(path).exists() else None


def _find_python() -> str | None:
    """Find a suitable Python 3.10-3.12 for the venv.

    uv-managed interpreters are checked first: on a uv-only machine (how these
    machines are set up) `python3.12` is often absent from PATH even though
    Python is installed, so probing PATH alone reports "no Python found" on a
    box that has three of them.
    """
    for version in SUPPORTED_PYTHONS:
        found = _uv_python(version)
        if found:
            return found

    for version in SUPPORTED_PYTHONS:
        exe = f"python{version}"
        try:
            result = subprocess.run([exe, "--version"], capture_output=True, text=True)
            if result.returncode == 0:
                return exe
        except FileNotFoundError:
            continue

    # Homebrew installs that never got linked onto PATH.
    for prefix in ("/opt/homebrew/bin", "/usr/local/bin"):
        for version in SUPPORTED_PYTHONS:
            candidate = Path(prefix) / f"python{version}"
            if candidate.exists():
                return str(candidate)

    return None


def _get_project_root() -> Path:
    """Walk up from this file to find pyproject.toml."""
    p = Path(__file__).resolve()
    for parent in [p] + list(p.parents):
        if (parent / "pyproject.toml").exists():
            return parent
    return Path.cwd()


@click.command()
@click.option("--port", default=DEFAULT_PORT, type=int, help="Port for web UI")
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
            console.print("  Install via: uv python install 3.12")
            console.print("  or:          brew install python@3.12")
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

    # Step 3: Check ML deps. The real test is whether Apple's torch2coreml
    # imports — that is what performs the conversion, and it is the piece most
    # likely to be missing or broken by a version drift.
    ml_check = subprocess.run(
        [str(venv_python), "-c",
         "import torch, diffusers, coremltools; "
         "import python_coreml_stable_diffusion.torch2coreml"],
        capture_output=True,
    )
    if ml_check.returncode == 0:
        console.print("[green]✓[/green] ML dependencies installed (torch, diffusers, coremltools, Apple converter)")
    else:
        console.print("[yellow]→[/yellow] Installing ML dependencies (~3GB, this takes several minutes)...")
        ml_result = subprocess.run(
            [str(venv_pip), "install", "--quiet", "-e", f"{project_root}[ml]"],
            capture_output=True, text=True,
        )
        if ml_result.returncode != 0:
            console.print(f"[red]✗ ML install failed:[/red] {ml_result.stderr[:300]}")
            console.print("[yellow]  You can still search/browse without ML deps.[/yellow]")
        else:
            console.print("[green]✓[/green] ML dependencies installed")

            # Apple's converter, without dependency resolution: its metadata
            # pins numpy<1.24, which cannot be built on Python 3.11+. Our [ml]
            # extra already pins the versions it actually needs.
            console.print("[yellow]→[/yellow] Installing Apple ml-stable-diffusion converter...")
            apple_result = subprocess.run(
                [str(venv_pip), "install", "--quiet", "--no-deps", APPLE_CONVERTER_URL],
                capture_output=True, text=True,
            )
            if apple_result.returncode != 0:
                console.print(f"[red]✗ Apple converter install failed:[/red] {apple_result.stderr[:300]}")
                console.print("[yellow]  Builds will not work until this succeeds.[/yellow]")
            else:
                verify = subprocess.run(
                    [str(venv_python), "-c",
                     "import python_coreml_stable_diffusion.torch2coreml"],
                    capture_output=True, text=True,
                )
                if verify.returncode == 0:
                    console.print("[green]✓[/green] Apple converter installed")
                else:
                    console.print(f"[red]✗ Apple converter imports failed:[/red] {verify.stderr[-300:]}")

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
