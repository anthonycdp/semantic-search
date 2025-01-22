"""Pytest configuration and fixtures."""
import json
import pytest
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.base import BaseSearchEngine
from src.bm25_search import BM25SearchEngine
from src.dense_search import DenseSearchEngine
from src.hybrid_search import HybridSearchEngine, FusionConfig


@pytest.fixture
def sample_documents():
    """Load sample documents for testing."""
    return [
        {"id": "doc1", "title": "Machine Learning Basics", "content": "Machine learning is a subset of artificial intelligence.", "metadata": {}},
        {"id": "doc2", "title": "Deep Learning Guide", "content": "Deep learning uses neural networks with multiple layers.", "metadata": {}},
        {"id": "doc3", "title": "Natural Language Processing", "content": "NLP enables computers to understand human language.", "metadata": {}},
        {"id": "doc4", "title": "Search Engine Optimization", "content": "SEO improves website visibility in search results.", "metadata": {}},
        {"id": "doc5", "title": "Vector Databases", "content": "Vector databases store embeddings for similarity search.", "metadata": {}},
    ]


@pytest.fixture
def bm25_engine(sample_documents):
    """Create and index a BM25 engine."""
    engine = BM25SearchEngine()
    engine.index_documents(sample_documents, show_progress=False)
    return engine


@pytest.fixture
def dense_engine(sample_documents):
    """Create and index a Dense engine."""
    engine = DenseSearchEngine(model_name="all-MiniLM-L6-v2")
    engine.index_documents(sample_documents, batch_size=32, show_progress=False)
    return engine


@pytest.fixture
def hybrid_engine(bm25_engine, dense_engine, sample_documents):
    """Create a Hybrid engine."""
    engine = HybridSearchEngine(
        name="TestHybrid",
        bm25_engine=bm25_engine,
        dense_engine=dense_engine,
        fusion_config=FusionConfig()
    )
    engine.is_indexed = True
    engine.documents = sample_documents
    return engine


@pytest.fixture
def benchmark_queries():
    """Sample queries for benchmarking."""
    return [
        "machine learning",
        "neural networks",
        "search",
        "database",
        "language processing"
    ]


@pytest.fixture
def ground_truth():
    """Ground truth for evaluation."""
    return {
        "machine learning": {"doc1", "doc2"},
        "neural networks": {"doc2"},
        "search": {"doc4", "doc5"},
        "database": {"doc5"},
        "language processing": {"doc3"}
    }
