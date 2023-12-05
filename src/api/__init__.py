"""FastAPI endpoints for semantic search."""

from .main import app, create_app
from .routes import router

__all__ = ['app', 'create_app', 'router']
