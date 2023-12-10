"""Tests for evaluation metrics."""
import pytest

from src.evaluation.metrics import (
    compute_precision_at_k,
    compute_recall_at_k,
    compute_f1_at_k,
    compute_average_precision,
    compute_mrr,
    compute_ndcg_at_k,
    compute_precision_recall,
    evaluate_retrieval
)


class TestPrecisionAtK:
    """Test Precision@K metric."""

    def test_perfect_precision(self):
        """Test perfect precision when all retrieved are relevant."""
        retrieved = ["d1", "d2", "d3"]
        relevant = {"d1", "d2", "d3", "d4"}

        precision = compute_precision_at_k(retrieved, relevant, k=3)
        assert precision == 1.0

    def test_partial_precision(self):
        """Test partial precision."""
        retrieved = ["d1", "d5", "d3"]
        relevant = {"d1", "d2", "d3"}

        precision = compute_precision_at_k(retrieved, relevant, k=3)
        assert precision == pytest.approx(2/3)

    def test_zero_precision(self):
        """Test zero precision when no retrieved are relevant."""
        retrieved = ["d5", "d6", "d7"]
        relevant = {"d1", "d2", "d3"}

        precision = compute_precision_at_k(retrieved, relevant, k=3)
        assert precision == 0.0

    def test_k_greater_than_retrieved(self):
        """Test when k is greater than retrieved list."""
        retrieved = ["d1", "d2"]
        relevant = {"d1", "d2", "d3"}

        precision = compute_precision_at_k(retrieved, relevant, k=5)
        assert precision == pytest.approx(2/5)


class TestRecallAtK:
    """Test Recall@K metric."""

    def test_perfect_recall(self):
        """Test perfect recall when all relevant are retrieved."""
        retrieved = ["d1", "d2", "d3"]
        relevant = {"d1", "d2"}

        recall = compute_recall_at_k(retrieved, relevant, k=3)
        assert recall == 1.0

    def test_partial_recall(self):
        """Test partial recall."""
        retrieved = ["d1", "d5"]
        relevant = {"d1", "d2", "d3"}

        recall = compute_recall_at_k(retrieved, relevant, k=2)
        assert recall == pytest.approx(1/3)

    def test_zero_recall(self):
        """Test zero recall when no relevant are retrieved."""
        retrieved = ["d5", "d6"]
        relevant = {"d1", "d2", "d3"}

        recall = compute_recall_at_k(retrieved, relevant, k=2)
        assert recall == 0.0

    def test_empty_relevant(self):
        """Test with empty relevant set."""
        retrieved = ["d1", "d2"]
        relevant = set()

        recall = compute_recall_at_k(retrieved, relevant, k=2)
        assert recall == 0.0


class TestF1AtK:
    """Test F1@K metric."""

    def test_perfect_f1(self):
        """Test perfect F1."""
        retrieved = ["d1", "d2", "d3"]
        relevant = {"d1", "d2", "d3"}

        f1 = compute_f1_at_k(retrieved, relevant, k=3)
        assert f1 == 1.0

    def test_balanced_f1(self):
        """Test F1 with balanced precision and recall."""
        retrieved = ["d1", "d2"]
        relevant = {"d1", "d2", "d3", "d4"}

        # Precision = 2/2 = 1.0
        # Recall = 2/4 = 0.5
        # F1 = 2 * (1.0 * 0.5) / (1.0 + 0.5) = 1.0 / 1.5 ≈ 0.667
        f1 = compute_f1_at_k(retrieved, relevant, k=2)
        assert f1 == pytest.approx(2/3, rel=0.01)


class TestAveragePrecision:
    """Test Average Precision metric."""

    def test_perfect_ap(self):
        """Test perfect Average Precision."""
        retrieved = ["d1", "d2", "d3"]
        relevant = {"d1", "d2", "d3"}

        ap = compute_average_precision(retrieved, relevant)
        assert ap == 1.0

    def test_ap_with_irrelevant(self):
        """Test AP with some irrelevant results."""
        retrieved = ["d1", "d5", "d2", "d6", "d3"]
        relevant = {"d1", "d2", "d3"}

        # P@1 = 1/1, P@3 = 2/3, P@5 = 3/5
        # AP = (1 + 2/3 + 3/5) / 3
        ap = compute_average_precision(retrieved, relevant)
        assert ap == pytest.approx((1 + 2/3 + 3/5) / 3)

    def test_ap_no_relevant_retrieved(self):
        """Test AP when no relevant docs are retrieved."""
        retrieved = ["d5", "d6", "d7"]
        relevant = {"d1", "d2", "d3"}

        ap = compute_average_precision(retrieved, relevant)
        assert ap == 0.0


class TestMRR:
    """Test Mean Reciprocal Rank metric."""

    def test_mrr_first_position(self):
        """Test MRR when relevant doc is first."""
        retrieved = ["d1", "d2", "d3"]
        relevant = {"d1"}

        mrr = compute_mrr(retrieved, relevant)
        assert mrr == 1.0

    def test_mrr_second_position(self):
        """Test MRR when relevant doc is second."""
        retrieved = ["d5", "d1", "d3"]
        relevant = {"d1"}

        mrr = compute_mrr(retrieved, relevant)
        assert mrr == 0.5

    def test_mrr_third_position(self):
        """Test MRR when relevant doc is third."""
        retrieved = ["d5", "d6", "d1"]
        relevant = {"d1"}

        mrr = compute_mrr(retrieved, relevant)
        assert mrr == pytest.approx(1/3)

    def test_mrr_not_found(self):
        """Test MRR when relevant doc is not in results."""
        retrieved = ["d5", "d6", "d7"]
        relevant = {"d1"}

        mrr = compute_mrr(retrieved, relevant)
        assert mrr == 0.0


class TestNDCG:
    """Test NDCG@K metric."""

    def test_perfect_ndcg(self):
        """Test perfect NDCG with ideal ranking."""
        retrieved = ["d1", "d2", "d3"]
        relevant = {"d1", "d2", "d3"}

        ndcg = compute_ndcg_at_k(retrieved, relevant, k=3)
        assert ndcg == pytest.approx(1.0)

    def test_ndcg_with_relevance_scores(self):
        """Test NDCG with graded relevance scores."""
        retrieved = ["d1", "d2", "d3"]
        relevant = {"d1", "d2", "d3"}
        rel_scores = {"d1": 3.0, "d2": 2.0, "d3": 1.0}

        ndcg = compute_ndcg_at_k(retrieved, relevant, k=3, relevance_scores=rel_scores)
        assert 0 < ndcg <= 1


class TestEvaluateRetrieval:
    """Test comprehensive evaluation function."""

    def test_evaluate_retrieval_basic(self):
        """Test basic evaluation."""
        search_results = {
            "q1": ["d1", "d2", "d3"],
            "q2": ["d4", "d5", "d6"]
        }
        ground_truth = {
            "q1": {"d1", "d2"},
            "q2": {"d4"}
        }

        metrics = evaluate_retrieval(search_results, ground_truth)

        assert 'num_queries' in metrics
        assert 'map' in metrics
        assert 'mrr' in metrics
        assert metrics['num_queries'] == 2

    def test_evaluate_retrieval_with_k_values(self):
        """Test evaluation with custom K values."""
        search_results = {
            "q1": ["d1", "d2", "d3", "d4", "d5"]
        }
        ground_truth = {
            "q1": {"d1", "d3"}
        }

        metrics = evaluate_retrieval(
            search_results,
            ground_truth,
            k_values=[1, 3, 5]
        )

        assert 'precision@1' in metrics
        assert 'recall@1' in metrics
        assert 'precision@3' in metrics
        assert 'precision@5' in metrics
