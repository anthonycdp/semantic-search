"""
Evaluation metrics for search systems.

This module provides standard IR (Information Retrieval) metrics
for evaluating and comparing different search methods.
"""
from typing import List, Dict, Set, Tuple
from collections import Counter


def compute_precision_at_k(
    retrieved_ids: List[str],
    relevant_ids: Set[str],
    k: int
) -> float:
    """
    Compute Precision@K.

    Precision@K = (# of relevant docs in top K) / K

    Args:
        retrieved_ids: List of retrieved document IDs in ranked order
        relevant_ids: Set of relevant document IDs (ground truth)
        k: Number of top results to consider

    Returns:
        Precision@K score
    """
    if k == 0:
        return 0.0

    top_k = retrieved_ids[:k]
    relevant_retrieved = sum(1 for doc_id in top_k if doc_id in relevant_ids)

    return relevant_retrieved / k


def compute_recall_at_k(
    retrieved_ids: List[str],
    relevant_ids: Set[str],
    k: int
) -> float:
    """
    Compute Recall@K.

    Recall@K = (# of relevant docs in top K) / (total # of relevant docs)

    Args:
        retrieved_ids: List of retrieved document IDs in ranked order
        relevant_ids: Set of relevant document IDs (ground truth)
        k: Number of top results to consider

    Returns:
        Recall@K score
    """
    if len(relevant_ids) == 0:
        return 0.0

    top_k = retrieved_ids[:k]
    relevant_retrieved = sum(1 for doc_id in top_k if doc_id in relevant_ids)

    return relevant_retrieved / len(relevant_ids)


def compute_f1_at_k(
    retrieved_ids: List[str],
    relevant_ids: Set[str],
    k: int
) -> float:
    """
    Compute F1@K (harmonic mean of precision and recall).

    Args:
        retrieved_ids: List of retrieved document IDs in ranked order
        relevant_ids: Set of relevant document IDs (ground truth)
        k: Number of top results to consider

    Returns:
        F1@K score
    """
    precision = compute_precision_at_k(retrieved_ids, relevant_ids, k)
    recall = compute_recall_at_k(retrieved_ids, relevant_ids, k)

    if precision + recall == 0:
        return 0.0

    return 2 * (precision * recall) / (precision + recall)


def compute_average_precision(
    retrieved_ids: List[str],
    relevant_ids: Set[str]
) -> float:
    """
    Compute Average Precision (AP).

    AP = (1/R) * sum(P(k) * rel(k))

    where R is the number of relevant documents, P(k) is precision at k,
    and rel(k) is 1 if document at rank k is relevant, 0 otherwise.

    Args:
        retrieved_ids: List of retrieved document IDs in ranked order
        relevant_ids: Set of relevant document IDs (ground truth)

    Returns:
        Average Precision score
    """
    if len(relevant_ids) == 0:
        return 0.0

    precisions = []
    num_relevant_found = 0

    for i, doc_id in enumerate(retrieved_ids):
        if doc_id in relevant_ids:
            num_relevant_found += 1
            precision_at_i = num_relevant_found / (i + 1)
            precisions.append(precision_at_i)

    if len(precisions) == 0:
        return 0.0

    return sum(precisions) / len(relevant_ids)


def compute_mrr(
    retrieved_ids: List[str],
    relevant_ids: Set[str]
) -> float:
    """
    Compute Mean Reciprocal Rank (MRR).

    MRR = 1 / rank of first relevant document

    Args:
        retrieved_ids: List of retrieved document IDs in ranked order
        relevant_ids: Set of relevant document IDs (ground truth)

    Returns:
        Reciprocal Rank score (use mean across queries for MRR)
    """
    for i, doc_id in enumerate(retrieved_ids):
        if doc_id in relevant_ids:
            return 1.0 / (i + 1)

    return 0.0


def compute_dcg_at_k(
    retrieved_ids: List[str],
    relevant_ids: Set[str],
    k: int,
    relevance_scores: Dict[str, float] = None
) -> float:
    """
    Compute Discounted Cumulative Gain at K (DCG@K).

    DCG@K = sum(rel(i) / log2(i + 1)) for i in 1..K

    Args:
        retrieved_ids: List of retrieved document IDs in ranked order
        relevant_ids: Set of relevant document IDs (ground truth)
        k: Number of top results to consider
        relevance_scores: Optional dict of doc_id -> relevance score (binary if None)

    Returns:
        DCG@K score
    """
    import math

    dcg = 0.0

    for i, doc_id in enumerate(retrieved_ids[:k]):
        if relevance_scores:
            rel = relevance_scores.get(doc_id, 0.0)
        else:
            rel = 1.0 if doc_id in relevant_ids else 0.0

        # Position is 1-indexed for DCG formula: rel / log2(position + 1)
        # i=0 (position 1): rel / log2(2) = rel / 1.0 = rel
        # i=1 (position 2): rel / log2(3), etc.
        position = i + 1
        dcg += rel / math.log2(position + 1)

    return dcg


def compute_ndcg_at_k(
    retrieved_ids: List[str],
    relevant_ids: Set[str],
    k: int,
    relevance_scores: Dict[str, float] = None
) -> float:
    """
    Compute Normalized Discounted Cumulative Gain at K (NDCG@K).

    NDCG@K = DCG@K / IDCG@K

    where IDCG is the ideal DCG (with perfect ranking).

    Args:
        retrieved_ids: List of retrieved document IDs in ranked order
        relevant_ids: Set of relevant document IDs (ground truth)
        k: Number of top results to consider
        relevance_scores: Optional dict of doc_id -> relevance score (binary if None)

    Returns:
        NDCG@K score (0 to 1)
    """
    dcg = compute_dcg_at_k(retrieved_ids, relevant_ids, k, relevance_scores)

    # Compute ideal DCG (perfect ranking)
    if relevance_scores:
        # Sort by relevance scores
        ideal_order = sorted(
            relevance_scores.keys(),
            key=lambda x: relevance_scores[x],
            reverse=True
        )[:k]
    else:
        ideal_order = list(relevant_ids)[:k]

    idcg = compute_dcg_at_k(ideal_order, relevant_ids, k, relevance_scores)

    if idcg == 0:
        return 0.0

    return dcg / idcg


def compute_precision_recall(
    retrieved_ids: List[str],
    relevant_ids: Set[str],
    k_values: List[int] = None
) -> Dict[str, float]:
    """
    Compute precision and recall at multiple K values.

    Args:
        retrieved_ids: List of retrieved document IDs in ranked order
        relevant_ids: Set of relevant document IDs (ground truth)
        k_values: List of K values to compute (default: [1, 5, 10, 20])

    Returns:
        Dictionary with precision@K and recall@K for each K
    """
    if k_values is None:
        k_values = [1, 5, 10, 20]

    results = {}

    for k in k_values:
        results[f'precision@{k}'] = compute_precision_at_k(retrieved_ids, relevant_ids, k)
        results[f'recall@{k}'] = compute_recall_at_k(retrieved_ids, relevant_ids, k)

    return results


def evaluate_retrieval(
    search_results: Dict[str, List[str]],
    ground_truth: Dict[str, Set[str]],
    k_values: List[int] = None,
    relevance_scores: Dict[str, Dict[str, float]] = None
) -> Dict[str, float]:
    """
    Evaluate search results against ground truth.

    Computes multiple metrics averaged across all queries.

    Args:
        search_results: Dict mapping query_id -> list of retrieved doc_ids
        ground_truth: Dict mapping query_id -> set of relevant doc_ids
        k_values: List of K values for P@K and R@K
        relevance_scores: Optional nested dict: query_id -> doc_id -> score

    Returns:
        Dictionary with averaged metrics
    """
    if k_values is None:
        k_values = [1, 5, 10, 20]

    metrics = {
        'num_queries': len(search_results),
        'map': 0.0,  # Mean Average Precision
        'mrr': 0.0,  # Mean Reciprocal Rank
    }

    # Add placeholders for P@K, R@K, F1@K, NDCG@K
    for k in k_values:
        metrics[f'precision@{k}'] = 0.0
        metrics[f'recall@{k}'] = 0.0
        metrics[f'f1@{k}'] = 0.0
        metrics[f'ndcg@{k}'] = 0.0

    num_queries = 0

    for query_id, retrieved in search_results.items():
        if query_id not in ground_truth:
            continue

        relevant = ground_truth[query_id]
        query_rel_scores = relevance_scores.get(query_id, {}) if relevance_scores else None

        # Compute metrics for this query
        metrics['map'] += compute_average_precision(retrieved, relevant)
        metrics['mrr'] += compute_mrr(retrieved, relevant)

        for k in k_values:
            metrics[f'precision@{k}'] += compute_precision_at_k(retrieved, relevant, k)
            metrics[f'recall@{k}'] += compute_recall_at_k(retrieved, relevant, k)
            metrics[f'f1@{k}'] += compute_f1_at_k(retrieved, relevant, k)
            metrics[f'ndcg@{k}'] += compute_ndcg_at_k(retrieved, relevant, k, query_rel_scores)

        num_queries += 1

    # Average all metrics
    if num_queries > 0:
        for key in metrics:
            if key != 'num_queries':
                metrics[key] /= num_queries

    return metrics
