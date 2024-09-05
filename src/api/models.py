"""
Pydantic models and shared state for the semantic search API.
"""
from typing import Optional

from fastapi import FastAPI
from pydantic import BaseModel, Field

from ..base import BaseSearchEngine


# Global search engines
search_engines: dict[str, BaseSearchEngine] = {}


class Document(BaseModel):
    """Document model for indexing."""
    id: str
    title: str
    content: str
    metadata: dict = Field(default_factory=dict)


class IndexRequest(BaseModel):
    """Request model for indexing documents."""
    documents: list[Document]
    model_name: str = Field(default="all-MiniLM-L6-v2", description="Embedding model for dense search")
    index_type: str = Field(default="flat", description="FAISS index type")


class SearchRequest(BaseModel):
    """Request model for search."""
    query: str
    k: int = Field(default=10, ge=1, le=100)
    method: str = Field(default="hybrid", description="Search method: bm25, dense, or hybrid")
    fusion_method: Optional[str] = Field(default="rrf", description="Fusion method for hybrid: rrf, weighted, max, combsum")
    bm25_weight: Optional[float] = Field(default=0.5, ge=0.0, le=1.0)
    dense_weight: Optional[float] = Field(default=0.5, ge=0.0, le=1.0)


class SearchResult(BaseModel):
    """Single search result."""
    doc_id: str
    score: float
    title: str
    content: str
    metadata: dict = Field(default_factory=dict)


class SearchResponse(BaseModel):
    """Response model for search."""
    query: str
    results: list[SearchResult]
    latency_ms: float
    method: str
    total_results: int


class IndexStatus(BaseModel):
    """Status of the index."""
    indexed: bool
    num_documents: int
    methods_available: list[str]


class CompareRequest(BaseModel):
    """Request for comparing search methods."""
    query: str
    k: int = Field(default=10, ge=1, le=100)


class CompareResponse(BaseModel):
    """Response with results from all methods."""
    query: str
    bm25: SearchResponse
    dense: SearchResponse
    hybrid: SearchResponse
