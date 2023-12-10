"""Tests for BM25 search engine."""
import pytest

from src.bm25_search import BM25SearchEngine


class TestBM25SearchEngine:
    """Test suite for BM25 search engine."""

    def test_initialization(self):
        """Test engine initialization."""
        engine = BM25SearchEngine(name="TestBM25")
        assert engine.name == "TestBM25"
        assert engine.k1 == 1.5
        assert engine.b == 0.75
        assert not engine.is_indexed

    def test_custom_parameters(self):
        """Test custom parameter initialization."""
        engine = BM25SearchEngine(name="Custom", k1=2.0, b=0.5)
        assert engine.k1 == 2.0
        assert engine.b == 0.5

    def test_index_documents(self, sample_documents):
        """Test document indexing."""
        engine = BM25SearchEngine()
        engine.index_documents(sample_documents, show_progress=False)

        assert engine.is_indexed
        assert len(engine.documents) == 5
        assert len(engine.tokenized_corpus) == 5

    def test_search_returns_results(self, bm25_engine):
        """Test that search returns results."""
        response = bm25_engine.search("machine learning", k=3)

        assert response.query == "machine learning"
        assert response.method == "BM25"
        assert len(response.results) > 0
        assert response.latency_ms > 0

    def test_search_result_structure(self, bm25_engine):
        """Test search result structure."""
        response = bm25_engine.search("machine learning", k=3)

        for result in response.results:
            assert hasattr(result, 'doc_id')
            assert hasattr(result, 'score')
            assert hasattr(result, 'title')
            assert hasattr(result, 'content')
            assert result.score >= 0

    def test_search_no_match(self, bm25_engine):
        """Test search with no matches."""
        response = bm25_engine.search("xyzzy123nonexistent", k=3)

        # Should return empty or very low scores
        assert isinstance(response.results, list)

    def test_get_document_by_id(self, bm25_engine):
        """Test document retrieval by ID."""
        doc = bm25_engine.get_document_by_id("doc1")

        assert doc is not None
        assert doc['id'] == "doc1"
        assert "Machine Learning" in doc['title']

    def test_get_document_by_id_not_found(self, bm25_engine):
        """Test retrieving non-existent document."""
        doc = bm25_engine.get_document_by_id("nonexistent")
        assert doc is None

    def test_search_without_indexing(self):
        """Test that search fails without indexing."""
        engine = BM25SearchEngine()

        with pytest.raises(RuntimeError, match="not indexed"):
            engine.search("test query")

    def test_explain_score(self, bm25_engine):
        """Test score explanation."""
        explanation = bm25_engine.explain_score("machine learning", "doc1")

        assert explanation['doc_id'] == "doc1"
        assert 'final_score' in explanation
        assert 'term_analysis' in explanation

    def test_tokenization(self):
        """Test tokenization method."""
        engine = BM25SearchEngine()

        tokens = engine._tokenize("Hello, World! This is a Test.")
        assert "hello" in tokens
        assert "world" in tokens
        assert "test" in tokens
        assert "," not in tokens
        assert "!" not in tokens

    def test_preprocess_documents(self, sample_documents):
        """Test document preprocessing."""
        engine = BM25SearchEngine()
        processed = engine._preprocess_documents(sample_documents)

        assert len(processed) == len(sample_documents)
        # Title should appear twice (weighted)
        assert processed[0].count("Machine") >= 2
