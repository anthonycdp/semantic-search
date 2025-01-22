"""Tests for Hybrid search engine."""
import pytest

from src.hybrid_search import HybridSearchEngine, FusionConfig


class TestHybridSearchEngine:
    """Test suite for Hybrid search engine."""

    def test_initialization(self, bm25_engine, dense_engine):
        """Test engine initialization."""
        engine = HybridSearchEngine(
            bm25_engine=bm25_engine,
            dense_engine=dense_engine
        )

        assert engine.name == "Hybrid"
        assert engine.bm25_engine is not None
        assert engine.dense_engine is not None
        assert engine.fusion_config is not None

    def test_custom_fusion_config(self, bm25_engine, dense_engine):
        """Test custom fusion configuration."""
        config = FusionConfig(
            method="weighted",
            bm25_weight=0.7,
            dense_weight=0.3
        )

        engine = HybridSearchEngine(
            bm25_engine=bm25_engine,
            dense_engine=dense_engine,
            fusion_config=config
        )

        assert engine.fusion_config.method == "weighted"
        assert engine.fusion_config.bm25_weight == 0.7
        assert engine.fusion_config.dense_weight == 0.3

    def test_search_returns_results(self, hybrid_engine):
        """Test that search returns results."""
        response = hybrid_engine.search("machine learning", k=3)

        assert response.query == "machine learning"
        assert "Hybrid" in response.method
        assert len(response.results) > 0
        assert response.latency_ms > 0

    def test_search_result_structure(self, hybrid_engine):
        """Test search result structure."""
        response = hybrid_engine.search("machine learning", k=3)

        for result in response.results:
            assert hasattr(result, 'doc_id')
            assert hasattr(result, 'score')
            assert hasattr(result, 'title')
            assert hasattr(result, 'content')
            assert 'fusion_method' in result.metadata

    def test_rrf_fusion(self, hybrid_engine):
        """Test RRF fusion method."""
        hybrid_engine.set_fusion_method("rrf")
        response = hybrid_engine.search("machine learning", k=5)

        assert len(response.results) > 0
        assert "rrf" in response.method

    def test_weighted_fusion(self, hybrid_engine):
        """Test weighted fusion method."""
        hybrid_engine.set_fusion_method("weighted")
        hybrid_engine.set_fusion_weights(0.7, 0.3)
        response = hybrid_engine.search("machine learning", k=5)

        assert len(response.results) > 0
        assert hybrid_engine.fusion_config.bm25_weight == 0.7
        assert hybrid_engine.fusion_config.dense_weight == 0.3

    def test_max_fusion(self, hybrid_engine):
        """Test max fusion method."""
        hybrid_engine.set_fusion_method("max")
        response = hybrid_engine.search("machine learning", k=5)

        assert len(response.results) > 0

    def test_combsum_fusion(self, hybrid_engine):
        """Test combsum fusion method."""
        hybrid_engine.set_fusion_method("combsum")
        response = hybrid_engine.search("machine learning", k=5)

        assert len(response.results) > 0

    def test_set_fusion_method_invalid(self, hybrid_engine):
        """Test setting invalid fusion method."""
        with pytest.raises(ValueError, match="Unknown fusion method"):
            hybrid_engine.set_fusion_method("invalid")

    def test_get_document_by_id(self, hybrid_engine):
        """Test document retrieval by ID."""
        doc = hybrid_engine.get_document_by_id("doc1")

        assert doc is not None
        assert doc['id'] == "doc1"

    def test_search_with_breakdown(self, hybrid_engine):
        """Test search with detailed breakdown."""
        result = hybrid_engine.search_with_breakdown("machine learning", k=3)

        assert 'hybrid' in result
        assert 'bm25' in result
        assert 'dense' in result
        assert 'fusion_config' in result

        # Check that all have results
        assert len(result['hybrid']['results']) > 0
        assert len(result['bm25']['results']) > 0
        assert len(result['dense']['results']) > 0

    def test_search_without_indexing(self):
        """Test that search fails without indexing."""
        engine = HybridSearchEngine()

        with pytest.raises(RuntimeError, match="not indexed"):
            engine.search("test query")

    def test_hybrid_combines_both_methods(self, hybrid_engine):
        """Test that hybrid combines BM25 and Dense results."""
        # Query that might get different results from each method
        response = hybrid_engine.search("AI neural network deep learning", k=5)

        # Should have results from the combination
        assert len(response.results) > 0

        # Check scores are fusion scores
        for result in response.results:
            assert result.score >= 0
            assert 'fusion_score' in result.metadata


class TestFusionConfig:
    """Test FusionConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = FusionConfig()

        assert config.method == "rrf"
        assert config.bm25_weight == 0.5
        assert config.dense_weight == 0.5
        assert config.rrf_k == 60

    def test_custom_config(self):
        """Test custom configuration values."""
        config = FusionConfig(
            method="weighted",
            bm25_weight=0.8,
            dense_weight=0.2,
            rrf_k=100
        )

        assert config.method == "weighted"
        assert config.bm25_weight == 0.8
        assert config.dense_weight == 0.2
        assert config.rrf_k == 100
