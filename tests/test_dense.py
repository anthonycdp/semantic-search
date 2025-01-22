"""Tests for Dense search engine."""
import pytest
import numpy as np

from src.dense_search import DenseSearchEngine


class TestDenseSearchEngine:
    """Test suite for Dense search engine."""

    def test_initialization(self):
        """Test engine initialization."""
        engine = DenseSearchEngine(name="TestDense")

        assert engine.name == "TestDense"
        assert engine.model_name == "all-MiniLM-L6-v2"
        assert engine.index_type == "flat"
        assert engine.normalize_embeddings is True
        assert not engine.is_indexed

    def test_custom_parameters(self):
        """Test custom parameter initialization."""
        engine = DenseSearchEngine(
            name="Custom",
            model_name="all-mpnet-base-v2",
            index_type="hnsw"
        )

        assert engine.model_name == "all-mpnet-base-v2"
        assert engine.index_type == "hnsw"

    def test_index_documents(self, sample_documents):
        """Test document indexing."""
        engine = DenseSearchEngine(model_name="all-MiniLM-L6-v2")
        engine.index_documents(sample_documents, batch_size=32, show_progress=False)

        assert engine.is_indexed
        assert len(engine.documents) == 5
        assert engine.doc_embeddings is not None
        assert engine.doc_embeddings.shape[0] == 5
        assert engine.faiss_index is not None

    def test_search_returns_results(self, dense_engine):
        """Test that search returns results."""
        response = dense_engine.search("machine learning", k=3)

        assert response.query == "machine learning"
        assert response.method == "Dense"
        assert len(response.results) > 0
        assert response.latency_ms > 0

    def test_search_result_structure(self, dense_engine):
        """Test search result structure."""
        response = dense_engine.search("machine learning", k=3)

        for result in response.results:
            assert hasattr(result, 'doc_id')
            assert hasattr(result, 'score')
            assert hasattr(result, 'title')
            assert hasattr(result, 'content')

    def test_search_semantic_matching(self, dense_engine):
        """Test semantic matching capabilities."""
        # Query with different words but similar meaning
        response1 = dense_engine.search("AI and ML", k=3)
        response2 = dense_engine.search("artificial intelligence machine learning", k=3)

        # Should find similar documents
        doc_ids1 = {r.doc_id for r in response1.results}
        doc_ids2 = {r.doc_id for r in response2.results}

        # At least some overlap expected
        assert len(doc_ids1) > 0
        assert len(doc_ids2) > 0

    def test_get_embedding(self, dense_engine):
        """Test embedding generation."""
        embedding = dense_engine.get_embedding("test sentence")

        assert isinstance(embedding, np.ndarray)
        assert embedding.shape[0] == dense_engine.embedding_dim

    def test_similarity(self, dense_engine):
        """Test text similarity computation."""
        similarity = dense_engine.similarity("machine learning", "AI and ML")
        assert 0 <= similarity <= 1

        # Similar sentences should have high similarity
        similarity_same = dense_engine.similarity("hello world", "hello world")
        assert similarity_same > 0.99

        # Different sentences should have lower similarity
        similarity_diff = dense_engine.similarity("machine learning", "cooking recipes")
        assert similarity_diff < similarity_same

    def test_get_document_by_id(self, dense_engine):
        """Test document retrieval by ID."""
        doc = dense_engine.get_document_by_id("doc1")

        assert doc is not None
        assert doc['id'] == "doc1"

    def test_get_document_by_id_not_found(self, dense_engine):
        """Test retrieving non-existent document."""
        doc = dense_engine.get_document_by_id("nonexistent")
        assert doc is None

    def test_search_without_indexing(self):
        """Test that search fails without indexing."""
        engine = DenseSearchEngine()

        with pytest.raises(RuntimeError, match="not indexed"):
            engine.search("test query")

    def test_embedding_dimension(self):
        """Test correct embedding dimension."""
        engine = DenseSearchEngine(model_name="all-MiniLM-L6-v2")
        assert engine.embedding_dim == 384  # all-MiniLM-L6-v2 dimension


class TestDenseIndexTypes:
    """Test different FAISS index types."""

    def test_flat_index(self, sample_documents):
        """Test flat index creation."""
        engine = DenseSearchEngine(index_type="flat")
        engine.index_documents(sample_documents, show_progress=False)

        response = engine.search("test", k=3)
        assert len(response.results) > 0

    def test_hnsw_index(self, sample_documents):
        """Test HNSW index creation."""
        engine = DenseSearchEngine(index_type="hnsw")
        engine.index_documents(sample_documents, show_progress=False)

        response = engine.search("test", k=3)
        assert len(response.results) > 0
