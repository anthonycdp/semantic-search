"""
Benchmark runner for comparing search methods.

Provides tools for running performance and accuracy benchmarks
across different search engines.
"""
import time
import json
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass, asdict
from pathlib import Path

from ..base import BaseSearchEngine


@dataclass
class BenchmarkResult:
    """Container for benchmark results."""
    method: str
    num_queries: int
    avg_latency_ms: float
    min_latency_ms: float
    max_latency_ms: float
    p50_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    total_time_ms: float
    queries_per_second: float

    # Accuracy metrics
    map_score: Optional[float] = None
    mrr_score: Optional[float] = None
    precision_at_10: Optional[float] = None
    recall_at_10: Optional[float] = None
    ndcg_at_10: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)


class BenchmarkRunner:
    """
    Runner for comparing search methods.

    Provides comprehensive benchmarking including latency metrics,
    throughput, and accuracy evaluation.
    """

    def __init__(
        self,
        queries: List[str],
        ground_truth: Dict[str, List[str]] = None,
        k: int = 10
    ):
        """
        Initialize benchmark runner.

        Args:
            queries: List of query strings to benchmark
            ground_truth: Optional dict mapping query -> list of relevant doc_ids
            k: Number of results to retrieve per query
        """
        self.queries = queries
        self.ground_truth = ground_truth or {}
        self.k = k

    def _compute_percentile(self, values: List[float], percentile: float) -> float:
        """Compute percentile of a list of values."""
        if not values:
            return 0.0
        sorted_values = sorted(values)
        index = int(len(sorted_values) * percentile / 100)
        return sorted_values[min(index, len(sorted_values) - 1)]

    def run_latency_benchmark(
        self,
        engine: BaseSearchEngine,
        warmup_queries: int = 5
    ) -> BenchmarkResult:
        """
        Run latency benchmark on a search engine.

        Args:
            engine: Search engine to benchmark
            warmup_queries: Number of warmup queries before timing

        Returns:
            BenchmarkResult with latency metrics
        """
        if not engine.is_indexed:
            raise RuntimeError("Engine must be indexed before benchmarking")

        # Warmup
        warmup = self.queries[:warmup_queries] if len(self.queries) >= warmup_queries else self.queries
        for query in warmup:
            engine.search(query, k=self.k)

        # Actual benchmark
        latencies = []

        for query in self.queries:
            start = time.perf_counter()
            engine.search(query, k=self.k)
            end = time.perf_counter()

            latencies.append((end - start) * 1000)  # Convert to ms

        # Compute statistics
        total_time = sum(latencies)
        avg_latency = total_time / len(latencies)

        return BenchmarkResult(
            method=engine.name,
            num_queries=len(self.queries),
            avg_latency_ms=avg_latency,
            min_latency_ms=min(latencies),
            max_latency_ms=max(latencies),
            p50_latency_ms=self._compute_percentile(latencies, 50),
            p95_latency_ms=self._compute_percentile(latencies, 95),
            p99_latency_ms=self._compute_percentile(latencies, 99),
            total_time_ms=total_time,
            queries_per_second=1000 / avg_latency
        )

    def run_accuracy_benchmark(
        self,
        engine: BaseSearchEngine
    ) -> BenchmarkResult:
        """
        Run accuracy benchmark comparing results to ground truth.

        Args:
            engine: Search engine to evaluate

        Returns:
            BenchmarkResult with accuracy metrics
        """
        from .metrics import evaluate_retrieval

        if not engine.is_indexed:
            raise RuntimeError("Engine must be indexed before benchmarking")

        # Collect search results
        search_results = {}
        latencies = []

        for query in self.queries:
            start = time.perf_counter()
            response = engine.search(query, k=self.k)
            end = time.perf_counter()

            latencies.append((end - start) * 1000)

            # Extract doc_ids
            search_results[query] = [r.doc_id for r in response.results]

        # Convert ground truth to expected format
        gt_sets = {q: set(docs) for q, docs in self.ground_truth.items()}

        # Compute accuracy metrics
        eval_results = evaluate_retrieval(search_results, gt_sets, k_values=[10])

        total_time = sum(latencies)
        avg_latency = total_time / len(latencies)

        return BenchmarkResult(
            method=engine.name,
            num_queries=len(self.queries),
            avg_latency_ms=avg_latency,
            min_latency_ms=min(latencies),
            max_latency_ms=max(latencies),
            p50_latency_ms=self._compute_percentile(latencies, 50),
            p95_latency_ms=self._compute_percentile(latencies, 95),
            p99_latency_ms=self._compute_percentile(latencies, 99),
            total_time_ms=total_time,
            queries_per_second=1000 / avg_latency,
            map_score=eval_results.get('map'),
            mrr_score=eval_results.get('mrr'),
            precision_at_10=eval_results.get('precision@10'),
            recall_at_10=eval_results.get('recall@10'),
            ndcg_at_10=eval_results.get('ndcg@10')
        )

    def run_full_benchmark(
        self,
        engines: Dict[str, BaseSearchEngine],
        include_accuracy: bool = True
    ) -> Dict[str, BenchmarkResult]:
        """
        Run full benchmark across multiple engines.

        Args:
            engines: Dict mapping engine name -> engine instance
            include_accuracy: Whether to compute accuracy metrics

        Returns:
            Dict mapping engine name -> BenchmarkResult
        """
        results = {}

        for name, engine in engines.items():
            print(f"\nBenchmarking {name}...")

            if include_accuracy and self.ground_truth:
                result = self.run_accuracy_benchmark(engine)
            else:
                result = self.run_latency_benchmark(engine)

            results[name] = result
            print(f"  Avg latency: {result.avg_latency_ms:.2f}ms")
            if result.map_score is not None:
                print(f"  MAP: {result.map_score:.4f}")
                print(f"  NDCG@10: {result.ndcg_at_10:.4f}")

        return results

    def compare_results(
        self,
        results: Dict[str, BenchmarkResult]
    ) -> str:
        """
        Generate a comparison table of benchmark results.

        Args:
            results: Dict of benchmark results

        Returns:
            Formatted string table
        """
        lines = []
        lines.append("=" * 100)
        lines.append("BENCHMARK COMPARISON")
        lines.append("=" * 100)
        lines.append(f"{'Method':<20} {'Avg Lat':<12} {'P95 Lat':<12} {'QPS':<12} {'MAP':<10} {'NDCG@10':<10}")
        lines.append("-" * 100)

        for name, result in results.items():
            lat = f"{result.avg_latency_ms:.2f}ms"
            p95 = f"{result.p95_latency_ms:.2f}ms"
            qps = f"{result.queries_per_second:.1f}"
            map_s = f"{result.map_score:.4f}" if result.map_score else "N/A"
            ndcg = f"{result.ndcg_at_10:.4f}" if result.ndcg_at_10 else "N/A"

            lines.append(f"{name:<20} {lat:<12} {p95:<12} {qps:<12} {map_s:<10} {ndcg:<10}")

        lines.append("=" * 100)

        return "\n".join(lines)

    def save_results(
        self,
        results: Dict[str, BenchmarkResult],
        output_path: str
    ) -> None:
        """
        Save benchmark results to JSON file.

        Args:
            results: Benchmark results
            output_path: Path to output JSON file
        """
        output = {
            'benchmark_config': {
                'num_queries': len(self.queries),
                'k': self.k,
                'has_ground_truth': bool(self.ground_truth)
            },
            'results': {name: r.to_dict() for name, r in results.items()}
        }

        with open(output_path, 'w') as f:
            json.dump(output, f, indent=2)

        print(f"Results saved to {output_path}")
