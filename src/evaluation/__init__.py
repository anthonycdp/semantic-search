"""Evaluation framework for comparing search methods."""

from .metrics import evaluate_retrieval, compute_mrr, compute_precision_recall
from .metrics import compute_ndcg_at_k as compute_ndcg
from .benchmark import BenchmarkRunner, BenchmarkResult

__all__ = [
    'evaluate_retrieval',
    'compute_mrr',
    'compute_ndcg',
    'compute_precision_recall',
    'BenchmarkRunner',
    'BenchmarkResult'
]
