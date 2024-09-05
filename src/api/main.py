"""
Main FastAPI application for semantic search API.

Provides REST endpoints for BM25, dense, and hybrid search.
"""
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .models import search_engines
from ..config import settings


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize search engines on startup."""
    # Engines will be initialized when documents are loaded
    yield
    # Cleanup if needed
    search_engines.clear()


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="Semantic Search API",
        description="API for comparing BM25, dense vector, and hybrid search methods",
        version="1.0.0",
        lifespan=lifespan
    )

    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Include routes
    from .routes import router
    app.include_router(router, prefix="/api/v1")

    return app


app = create_app()
