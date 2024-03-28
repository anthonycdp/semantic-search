"""
Base classes and interfaces for search systems.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
import time


@dataclass
class SearchResult:
    """Represents a single search result."""
    doc_id: str
    score: float
    title: str
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'doc_id': self.doc_id,
            'score': self.score,
            'title': self.title,
            'content': self.content,
            'metadata': self.metadata
        }


@dataclass
class SearchResponse:
    """Represents a complete search response."""
    query: str
    results: List[SearchResult]
    latency_ms: float
    method: str
    total_results: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            'query': self.query,
            'results': [r.to_dict() for r in self.results],
            'latency_ms': self.latency_ms,
            'method': self.method,
            'total_results': self.total_results
        }


class BaseSearchEngine(ABC):
    """Abstract base class for all search engines."""

    def __init__(self, name: str):
        self.name = name
        self.documents: List[Dict[str, Any]] = []
        self.is_indexed = False

    @abstractmethod
    def index_documents(self, documents: List[Dict[str, Any]]) -> None:
        """
        Index a list of documents.

        Args:
            documents: List of document dictionaries with 'id', 'title', 'content' keys
        """
        pass

    @abstractmethod
    def search(self, query: str, k: int = 10) -> SearchResponse:
        """
        Search for documents matching the query.

        Args:
            query: Search query string
            k: Number of results to return

        Returns:
            SearchResponse object with results and metadata
        """
        pass

    @abstractmethod
    def get_document_by_id(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a document by its ID."""
        pass

    def benchmark_search(self, queries: List[str], k: int = 10) -> Dict[str, Any]:
        """
        Benchmark search performance across multiple queries.

        Args:
            queries: List of query strings
            k: Number of results per query

        Returns:
            Dictionary with benchmark statistics
        """
        latencies = []
        results_per_query = []

        for query in queries:
            start_time = time.perf_counter()
            response = self.search(query, k)
            end_time = time.perf_counter()

            latency = (end_time - start_time) * 1000  # Convert to ms
            latencies.append(latency)
            results_per_query.append(len(response.results))

        return {
            'method': self.name,
            'num_queries': len(queries),
            'avg_latency_ms': sum(latencies) / len(latencies),
            'min_latency_ms': min(latencies),
            'max_latency_ms': max(latencies),
            'p50_latency_ms': sorted(latencies)[len(latencies) // 2],
            'p95_latency_ms': sorted(latencies)[int(len(latencies) * 0.95)] if len(latencies) > 20 else max(latencies),
            'total_time_ms': sum(latencies)
        }
