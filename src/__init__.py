"""
Semantic Search System - Compare BM25, Dense Embeddings, and Hybrid Search.

This package provides implementations of three search approaches:
- BM25: Lexical matching using term frequency and document length normalization
- Dense: Semantic search using neural embeddings and vector similarity
- Hybrid: Combines both approaches with configurable fusion strategies
"""

from .base import BaseSearchEngine, SearchResult, SearchResponse
from .bm25_search import BM25SearchEngine
from .dense_search import DenseSearchEngine
from .hybrid_search import HybridSearchEngine, FusionConfig

__all__ = [
    'BaseSearchEngine',
    'SearchResult',
    'SearchResponse',
    'BM25SearchEngine',
    'DenseSearchEngine',
    'HybridSearchEngine',
    'FusionConfig'
]

__version__ = '1.0.0'
