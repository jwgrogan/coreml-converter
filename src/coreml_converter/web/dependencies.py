from __future__ import annotations

from pathlib import Path

from fastapi import Request
from fastapi.templating import Jinja2Templates
from starlette.responses import Response

TEMPLATES_DIR = Path(__file__).parent / "templates"
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


def render(request: Request, template: str, context: dict | None = None) -> Response:
    """Render a template with Starlette 1.0+ compatible API."""
    ctx = context or {}
    return templates.TemplateResponse(request, template, ctx)


def get_registry(request: Request):
    return request.app.state.registry


def get_build_store(request: Request):
    return request.app.state.build_store


def get_train_store(request: Request):
    return request.app.state.train_store


def get_train_manager(request: Request):
    return request.app.state.train_manager
