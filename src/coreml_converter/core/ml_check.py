"""Check availability of ML dependencies."""
from __future__ import annotations


def check_ml_deps() -> tuple[bool, list[str]]:
    """Returns (all_ok, list_of_missing)."""
    missing = []
    for pkg, import_name in [
        ("torch", "torch"),
        ("diffusers", "diffusers"),
        ("transformers", "transformers"),
        ("safetensors", "safetensors"),
        ("coremltools", "coremltools"),
    ]:
        try:
            __import__(import_name)
        except ImportError:
            missing.append(pkg)
    return len(missing) == 0, missing


def require_ml_deps() -> None:
    """Raise with a helpful message if ML deps are missing."""
    ok, missing = check_ml_deps()
    if not ok:
        raise RuntimeError(
            f"Missing ML dependencies: {', '.join(missing)}\n\n"
            f"Run: ccml start   (sets up venv with all dependencies)\n"
            f"Or:  source .venv/bin/activate && pip install -e '.[ml]'"
        )
