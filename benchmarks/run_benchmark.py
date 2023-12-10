"""
Performance benchmarks for semantic search methods.

Run this script to benchmark BM25, Dense, and Hybrid search methods
on a sample dataset.
"""
import json
import time
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.bm25_search import BM25SearchEngine
from src.dense_search import DenseSearchEngine
from src.hybrid_search import HybridSearchEngine, FusionConfig
from src.evaluation.benchmark import BenchmarkRunner


def load_sample_data():
    """Load sample documents."""
    data_path = Path(__file__).parent.parent / "data" / "sample_documents.json"

    with open(data_path, 'r') as f:
        documents = json.load(f)

    return documents


def get_benchmark_queries():
    """Get benchmark queries with expected results."""
    queries = [
        "machine learning basics",
        "semantic search using embeddings",
        "how does BM25 ranking work",
        "vector database for similarity search",
        "transformer neural network architecture",
        "hybrid search combining multiple methods",
        "NLP text preprocessing",
        "document chunking strategies",
        "RAG retrieval augmented generation",
        "cosine similarity for comparing text",
    ]

    # Ground truth: relevant doc IDs for each query
    ground_truth = {
        "machine learning basics": ["doc001", "doc002"],
        "semantic search using embeddings": ["doc008", "doc006", "doc011"],
        "how does BM25 ranking work": ["doc005", "doc004"],
        "vector database for similarity search": ["doc006", "doc012"],
        "transformer neural network architecture": ["doc007", "doc026"],
        "hybrid search combining multiple methods": ["doc009", "doc017"],
        "NLP text preprocessing": ["doc003", "doc025"],
        "document chunking strategies": ["doc018"],
        "RAG retrieval augmented generation": ["doc019"],
        "cosine similarity for comparing text": ["doc029", "doc006"],
    }

    return queries, ground_truth


def run_indexing_benchmark(documents):
    """Benchmark document indexing time."""
    print("\n" + "="*60)
    print("INDEXING BENCHMARK")
    print("="*60)

    results = {}

    # BM25 indexing
    print("\nIndexing with BM25...")
    bm25 = BM25SearchEngine(name="BM25")
    start = time.perf_counter()
    bm25.index_documents(documents, show_progress=False)
    bm25_time = (time.perf_counter() - start) * 1000
    results["BM25"] = {"indexing_time_ms": bm25_time}
    print(f"  BM25 indexing time: {bm25_time:.2f}ms")

    # Dense indexing
    print("\nIndexing with Dense (all-MiniLM-L6-v2)...")
    dense = DenseSearchEngine(name="Dense", model_name="all-MiniLM-L6-v2")
    start = time.perf_counter()
    dense.index_documents(documents, batch_size=32, show_progress=False)
    dense_time = (time.perf_counter() - start) * 1000
    results["Dense"] = {"indexing_time_ms": dense_time}
    print(f"  Dense indexing time: {dense_time:.2f}ms")

    # Hybrid (uses already-indexed engines)
    hybrid = HybridSearchEngine(
        name="Hybrid",
        bm25_engine=bm25,
        dense_engine=dense
    )
    hybrid.is_indexed = True
    hybrid.documents = documents
    results["Hybrid"] = {"indexing_time_ms": bm25_time + dense_time}

    return results, bm25, dense, hybrid


def run_search_benchmark(engines, queries, ground_truth):
    """Benchmark search performance."""
    print("\n" + "="*60)
    print("SEARCH BENCHMARK")
    print("="*60)

    runner = BenchmarkRunner(queries=queries, ground_truth=ground_truth, k=10)
    results = runner.run_full_benchmark(engines, include_accuracy=True)

    # Print comparison table
    print(runner.compare_results(results))

    return results


def run_fusion_method_comparison(hybrid_engine, queries, ground_truth):
    """Compare different fusion methods."""
    print("\n" + "="*60)
    print("FUSION METHOD COMPARISON")
    print("="*60)

    fusion_methods = ["rrf", "weighted", "max", "combsum"]
    results = {}

    for method in fusion_methods:
        print(f"\nTesting {method} fusion...")
        hybrid_engine.set_fusion_method(method)

        latencies = []
        for query in queries:
            start = time.perf_counter()
            hybrid_engine.search(query, k=10)
            latencies.append((time.perf_counter() - start) * 1000)

        avg_latency = sum(latencies) / len(latencies)
        results[method] = {"avg_latency_ms": avg_latency}
        print(f"  Average latency: {avg_latency:.2f}ms")

    print("\nFusion Method Results:")
    print(f"{'Method':<15} {'Avg Latency':<15}")
    print("-" * 30)
    for method, data in results.items():
        print(f"{method:<15} {data['avg_latency_ms']:.2f}ms")

    return results


def main():
    """Run all benchmarks."""
    print("="*60)
    print("SEMANTIC SEARCH BENCHMARK SUITE")
    print("="*60)

    # Load data
    print("\nLoading sample documents...")
    documents = load_sample_data()
    print(f"Loaded {len(documents)} documents")

    # Load queries
    queries, ground_truth = get_benchmark_queries()
    print(f"Loaded {len(queries)} benchmark queries")

    # Run indexing benchmark
    indexing_results, bm25, dense, hybrid = run_indexing_benchmark(documents)

    # Define engines for search benchmark
    engines = {
        "BM25": bm25,
        "Dense": dense,
        "Hybrid": hybrid
    }

    # Run search benchmark
    search_results = run_search_benchmark(engines, queries, ground_truth)

    # Run fusion comparison
    fusion_results = run_fusion_method_comparison(hybrid, queries, ground_truth)

    # Summary
    print("\n" + "="*60)
    print("BENCHMARK SUMMARY")
    print("="*60)

    print("\nIndexing Performance:")
    for method, data in indexing_results.items():
        print(f"  {method}: {data['indexing_time_ms']:.2f}ms")

    print("\nSearch Performance (avg latency):")
    for method, result in search_results.items():
        print(f"  {method}: {result.avg_latency_ms:.2f}ms")
        if result.map_score:
            print(f"    MAP: {result.map_score:.4f}")
            print(f"    NDCG@10: {result.ndcg_at_10:.4f}")

    print("\n✅ Benchmarks complete!")

    # Save results
    output_path = Path(__file__).parent / "benchmark_results.json"
    output = {
        "indexing": indexing_results,
        "search": {k: v.to_dict() for k, v in search_results.items()},
        "fusion_methods": fusion_results
    }

    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
