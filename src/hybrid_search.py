"""
Hybrid Search Engine Implementation.

Hybrid search combines lexical (BM25) and semantic (dense) search approaches
to get the best of both worlds. It can capture both exact keyword matches
and semantic relationships.

Key characteristics:
- Combines lexical and semantic matching
- More robust across different query types
- Better recall and precision
- Configurable fusion strategies
"""
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from collections import defaultdict
import time

from .base import BaseSearchEngine, SearchResult, SearchResponse
from .bm25_search import BM25SearchEngine
from .dense_search import DenseSearchEngine


@dataclass
class FusionConfig:
    """Configuration for score fusion."""
    method: str = "rrf"  # 'rrf', 'weighted', 'max', 'combsum'
    bm25_weight: float = 0.5
    dense_weight: float = 0.5
    rrf_k: int = 60  # RRF constant


class HybridSearchEngine(BaseSearchEngine):
    """
    Hybrid search engine combining BM25 and dense vector search.

    Supports multiple fusion strategies for combining scores from
    different search methods.
    """

    VALID_FUSION_METHODS = {'rrf', 'weighted', 'max', 'combsum'}

    def __init__(
        self,
        name: str = "Hybrid",
        bm25_engine: BM25SearchEngine = None,
        dense_engine: DenseSearchEngine = None,
        fusion_config: FusionConfig = None
    ):
        """
        Initialize hybrid search engine.

        Args:
            name: Name identifier for this engine
            bm25_engine: BM25 search engine instance
            dense_engine: Dense search engine instance
            fusion_config: Configuration for score fusion
        """
        super().__init__(name)

        self.bm25_engine = bm25_engine or BM25SearchEngine()
        self.dense_engine = dense_engine or DenseSearchEngine()
        self.fusion_config = fusion_config or FusionConfig()

        # Fusion method dispatcher
        self._fusion_methods = {
            'rrf': self._reciprocal_rank_fusion,
            'weighted': self._weighted_fusion,
            'max': self._max_fusion,
            'combsum': self._combsum_fusion,
        }

    def index_documents(
        self,
        documents: List[Dict[str, Any]],
        batch_size: int = 32,
        show_progress: bool = True
    ) -> None:
        """
        Index documents in both BM25 and dense engines.

        Args:
            documents: List of document dictionaries
            batch_size: Batch size for dense encoding
            show_progress: Whether to show progress bars
        """
        self.documents = documents

        if show_progress:
            print("Indexing documents in BM25 engine...")
        self.bm25_engine.index_documents(documents, show_progress=show_progress)

        if show_progress:
            print("\nIndexing documents in Dense engine...")
        self.dense_engine.index_documents(
            documents,
            batch_size=batch_size,
            show_progress=show_progress
        )

        self.is_indexed = True

        if show_progress:
            print("\nHybrid indexing complete!")

    def _normalize_scores(
        self,
        results: List[SearchResult],
        method: str = "minmax"
    ) -> List[Tuple[str, float]]:
        """
        Normalize scores to [0, 1] range.

        Args:
            results: List of search results
            method: Normalization method ('minmax', 'sum', 'none')

        Returns:
            List of (doc_id, normalized_score) tuples
        """
        if not results:
            return []

        scores = [r.score for r in results]

        if method == "minmax":
            min_score = min(scores)
            max_score = max(scores)
            if max_score - min_score > 0:
                normalized = [(r.doc_id, (r.score - min_score) / (max_score - min_score))
                            for r in results]
            else:
                normalized = [(r.doc_id, 1.0) for r in results]

        elif method == "sum":
            total = sum(scores)
            if total > 0:
                normalized = [(r.doc_id, r.score / total) for r in results]
            else:
                normalized = [(r.doc_id, 1.0 / len(results)) for r in results]

        else:  # 'none'
            normalized = [(r.doc_id, r.score) for r in results]

        return normalized

    def _reciprocal_rank_fusion(
        self,
        bm25_results: List[SearchResult],
        dense_results: List[SearchResult],
        k: int = 60
    ) -> List[Tuple[str, float]]:
        """
        Combine results using Reciprocal Rank Fusion (RRF).

        RRF score = sum(1 / (k + rank)) for each result list

        Args:
            bm25_results: Results from BM25 search
            dense_results: Results from dense search
            k: RRF constant (default: 60)

        Returns:
            List of (doc_id, rrf_score) tuples sorted by score
        """
        rrf_scores = defaultdict(float)

        # Add BM25 ranks
        for rank, result in enumerate(bm25_results, 1):
            rrf_scores[result.doc_id] += 1 / (k + rank)

        # Add dense ranks
        for rank, result in enumerate(dense_results, 1):
            rrf_scores[result.doc_id] += 1 / (k + rank)

        # Sort by RRF score
        sorted_results = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

        return sorted_results

    def _weighted_fusion(
        self,
        bm25_results: List[SearchResult],
        dense_results: List[SearchResult],
        bm25_weight: float = 0.5,
        dense_weight: float = 0.5
    ) -> List[Tuple[str, float]]:
        """
        Combine results using weighted score fusion.

        Final score = w1 * normalized_bm25 + w2 * normalized_dense

        Args:
            bm25_results: Results from BM25 search
            dense_results: Results from dense search
            bm25_weight: Weight for BM25 scores
            dense_weight: Weight for dense scores

        Returns:
            List of (doc_id, weighted_score) tuples sorted by score
        """
        # Normalize scores
        bm25_normalized = dict(self._normalize_scores(bm25_results, method="minmax"))
        dense_normalized = dict(self._normalize_scores(dense_results, method="minmax"))

        # Combine scores
        combined_scores = defaultdict(float)
        all_doc_ids = set(bm25_normalized.keys()) | set(dense_normalized.keys())

        for doc_id in all_doc_ids:
            bm25_score = bm25_normalized.get(doc_id, 0.0)
            dense_score = dense_normalized.get(doc_id, 0.0)
            combined_scores[doc_id] = (bm25_weight * bm25_score +
                                       dense_weight * dense_score)

        # Sort by combined score
        sorted_results = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)

        return sorted_results

    def _max_fusion(
        self,
        bm25_results: List[SearchResult],
        dense_results: List[SearchResult]
    ) -> List[Tuple[str, float]]:
        """
        Combine results using max score fusion.

        Final score = max(normalized_bm25, normalized_dense)

        Args:
            bm25_results: Results from BM25 search
            dense_results: Results from dense search

        Returns:
            List of (doc_id, max_score) tuples sorted by score
        """
        # Normalize scores
        bm25_normalized = dict(self._normalize_scores(bm25_results, method="minmax"))
        dense_normalized = dict(self._normalize_scores(dense_results, method="minmax"))

        # Combine using max
        combined_scores = {}
        all_doc_ids = set(bm25_normalized.keys()) | set(dense_normalized.keys())

        for doc_id in all_doc_ids:
            bm25_score = bm25_normalized.get(doc_id, 0.0)
            dense_score = dense_normalized.get(doc_id, 0.0)
            combined_scores[doc_id] = max(bm25_score, dense_score)

        # Sort by combined score
        sorted_results = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)

        return sorted_results

    def _combsum_fusion(
        self,
        bm25_results: List[SearchResult],
        dense_results: List[SearchResult]
    ) -> List[Tuple[str, float]]:
        """
        Combine results using CombSUM fusion.

        Final score = sum(normalized_bm25, normalized_dense)

        Args:
            bm25_results: Results from BM25 search
            dense_results: Results from dense search

        Returns:
            List of (doc_id, sum_score) tuples sorted by score
        """
        # Normalize scores using sum normalization
        bm25_normalized = dict(self._normalize_scores(bm25_results, method="sum"))
        dense_normalized = dict(self._normalize_scores(dense_results, method="sum"))

        # Combine using sum
        combined_scores = defaultdict(float)
        all_doc_ids = set(bm25_normalized.keys()) | set(dense_normalized.keys())

        for doc_id in all_doc_ids:
            combined_scores[doc_id] = (bm25_normalized.get(doc_id, 0.0) +
                                       dense_normalized.get(doc_id, 0.0))

        # Sort by combined score
        sorted_results = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)

        return sorted_results

    def _fuse_results(
        self,
        bm25_response: SearchResponse,
        dense_response: SearchResponse,
        k: int
    ) -> List[Tuple[str, float]]:
        """
        Fuse results from BM25 and dense search.

        Args:
            bm25_response: BM25 search response
            dense_response: Dense search response
            k: Number of results to return

        Returns:
            List of (doc_id, fused_score) tuples
        """
        method = self.fusion_config.method

        if method not in self._fusion_methods:
            raise ValueError(f"Unknown fusion method: {method}")

        fusion_func = self._fusion_methods[method]

        # Prepare arguments based on method
        if method == 'rrf':
            fused = fusion_func(
                bm25_response.results,
                dense_response.results,
                k=self.fusion_config.rrf_k
            )
        elif method == 'weighted':
            fused = fusion_func(
                bm25_response.results,
                dense_response.results,
                bm25_weight=self.fusion_config.bm25_weight,
                dense_weight=self.fusion_config.dense_weight
            )
        else:
            fused = fusion_func(
                bm25_response.results,
                dense_response.results
            )

        return fused[:k]

    def search(
        self,
        query: str,
        k: int = 10,
        retrieval_k: int = 100
    ) -> SearchResponse:
        """
        Search using hybrid approach.

        Args:
            query: Search query string
            k: Number of final results to return
            retrieval_k: Number of results to retrieve from each engine

        Returns:
            SearchResponse with fused results
        """
        if not self.is_indexed:
            raise RuntimeError("Documents not indexed. Call index_documents() first.")

        start_time = time.perf_counter()

        bm25_response = self.bm25_engine.search(query, k=retrieval_k)
        dense_response = self.dense_engine.search(query, k=retrieval_k)

        fused_results = self._fuse_results(bm25_response, dense_response, k)
        results = self._build_search_results(fused_results)

        end_time = time.perf_counter()
        latency_ms = (end_time - start_time) * 1000

        return SearchResponse(
            query=query,
            results=results,
            latency_ms=latency_ms,
            method=f"{self.name} ({self.fusion_config.method})",
            total_results=len(results)
        )

    def get_document_by_id(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a document by its ID."""
        return self.bm25_engine.get_document_by_id(doc_id)

    def search_with_breakdown(
        self,
        query: str,
        k: int = 10,
        retrieval_k: int = 100
    ) -> Dict[str, Any]:
        """
        Search with detailed breakdown of individual engine results.

        Args:
            query: Search query string
            k: Number of final results to return
            retrieval_k: Number of results to retrieve from each engine

        Returns:
            Dictionary with hybrid results and individual engine results
        """
        if not self.is_indexed:
            raise RuntimeError("Documents not indexed. Call index_documents() first.")

        start_time = time.perf_counter()

        bm25_response = self.bm25_engine.search(query, k=retrieval_k)
        dense_response = self.dense_engine.search(query, k=retrieval_k)

        fused_results = self._fuse_results(bm25_response, dense_response, k)

        # Build hybrid response from already computed results
        results = self._build_search_results(fused_results)

        end_time = time.perf_counter()
        latency_ms = (end_time - start_time) * 1000

        hybrid_response = SearchResponse(
            query=query,
            results=results,
            latency_ms=latency_ms,
            method=f"{self.name} ({self.fusion_config.method})",
            total_results=len(results)
        )

        return {
            'hybrid': hybrid_response.to_dict(),
            'bm25': bm25_response.to_dict(),
            'dense': dense_response.to_dict(),
            'fusion_config': {
                'method': self.fusion_config.method,
                'bm25_weight': self.fusion_config.bm25_weight,
                'dense_weight': self.fusion_config.dense_weight,
                'rrf_k': self.fusion_config.rrf_k
            }
        }

    def _build_search_results(
        self,
        fused_results: List[Tuple[str, float]]
    ) -> List[SearchResult]:
        """Build SearchResult list from fused results."""
        results = []
        for doc_id, score in fused_results:
            doc = self.get_document_by_id(doc_id)
            if doc:
                results.append(SearchResult(
                    doc_id=doc_id,
                    score=score,
                    title=doc.get('title', ''),
                    content=doc.get('content', ''),
                    metadata={
                        'fusion_score': score,
                        'fusion_method': self.fusion_config.method
                    }
                ))
        return results

    def set_fusion_weights(self, bm25_weight: float, dense_weight: float) -> None:
        """
        Update fusion weights.

        Args:
            bm25_weight: Weight for BM25 scores
            dense_weight: Weight for dense scores
        """
        self.fusion_config.bm25_weight = bm25_weight
        self.fusion_config.dense_weight = dense_weight

    def set_fusion_method(self, method: str) -> None:
        """
        Set the fusion method.

        Args:
            method: Fusion method ('rrf', 'weighted', 'max', 'combsum')
        """
        if method not in self.VALID_FUSION_METHODS:
            raise ValueError(f"Unknown fusion method: {method}")
        self.fusion_config.method = method
