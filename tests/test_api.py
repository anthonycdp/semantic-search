"""Tests for API endpoints."""
import pytest
from fastapi.testclient import TestClient
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.api.main import app
from src.api.models import search_engines
from src.bm25_search import BM25SearchEngine
from src.dense_search import DenseSearchEngine
from src.hybrid_search import HybridSearchEngine, FusionConfig


@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)


@pytest.fixture
def indexed_engines():
    """Create and store indexed engines."""
    docs = [
        {"id": "doc1", "title": "Machine Learning", "content": "ML is AI subset", "metadata": {}},
        {"id": "doc2", "title": "Deep Learning", "content": "Uses neural networks", "metadata": {}},
    ]

    bm25 = BM25SearchEngine()
    bm25.index_documents(docs, show_progress=False)
    search_engines["bm25"] = bm25

    dense = DenseSearchEngine(model_name="all-MiniLM-L6-v2")
    dense.index_documents(docs, batch_size=32, show_progress=False)
    search_engines["dense"] = dense

    hybrid = HybridSearchEngine(bm25_engine=bm25, dense_engine=dense)
    hybrid.is_indexed = True
    hybrid.documents = docs
    search_engines["hybrid"] = hybrid

    yield

    search_engines.clear()


class TestHealthEndpoint:
    """Test health check endpoint."""

    def test_health_check(self, client):
        """Test health endpoint returns healthy."""
        response = client.get("/api/v1/health")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"


class TestStatusEndpoint:
    """Test status endpoint."""

    def test_status_not_indexed(self, client):
        """Test status when not indexed."""
        search_engines.clear()
        response = client.get("/api/v1/status")

        assert response.status_code == 200
        data = response.json()
        assert data["indexed"] is False
        assert data["num_documents"] == 0


class TestIndexEndpoint:
    """Test document indexing endpoint."""

    def test_index_documents(self, client):
        """Test indexing documents."""
        response = client.post("/api/v1/index", json={
            "documents": [
                {"id": "d1", "title": "Test", "content": "Test content", "metadata": {}}
            ]
        })

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert data["num_documents"] == 1

        # Clean up
        search_engines.clear()

    def test_index_empty_documents(self, client):
        """Test indexing with empty document list."""
        response = client.post("/api/v1/index", json={
            "documents": []
        })

        assert response.status_code == 400


class TestSearchEndpoint:
    """Test search endpoints."""

    def test_search_bm25(self, client, indexed_engines):
        """Test BM25 search."""
        response = client.post("/api/v1/search", json={
            "query": "machine learning",
            "k": 5,
            "method": "bm25"
        })

        assert response.status_code == 200
        data = response.json()
        assert data["query"] == "machine learning"
        assert len(data["results"]) > 0

    def test_search_dense(self, client, indexed_engines):
        """Test dense search."""
        response = client.post("/api/v1/search", json={
            "query": "machine learning",
            "k": 5,
            "method": "dense"
        })

        assert response.status_code == 200
        data = response.json()
        assert data["query"] == "machine learning"

    def test_search_hybrid(self, client, indexed_engines):
        """Test hybrid search."""
        response = client.post("/api/v1/search", json={
            "query": "machine learning",
            "k": 5,
            "method": "hybrid"
        })

        assert response.status_code == 200
        data = response.json()
        assert "Hybrid" in data["method"]

    def test_search_without_index(self, client):
        """Test search without indexing."""
        search_engines.clear()

        response = client.post("/api/v1/search", json={
            "query": "test",
            "k": 5,
            "method": "bm25"
        })

        assert response.status_code == 400


class TestCompareEndpoint:
    """Test compare endpoint."""

    def test_compare_methods(self, client, indexed_engines):
        """Test comparing all methods."""
        response = client.post("/api/v1/compare", json={
            "query": "machine learning",
            "k": 5
        })

        assert response.status_code == 200
        data = response.json()
        assert "bm25" in data
        assert "dense" in data
        assert "hybrid" in data


class TestClearIndex:
    """Test clear index endpoint."""

    def test_clear_index(self, client, indexed_engines):
        """Test clearing the index."""
        response = client.delete("/api/v1/index")

        assert response.status_code == 200
        assert response.json()["status"] == "success"
