# Semantic Search: BM25 vs Embeddings vs Hybrid

A comprehensive semantic search system that implements and compares three search approaches:

- **BM25**: Lexical matching using term frequency and document length normalization
- **Dense (Embeddings)**: Semantic search using neural network embeddings and vector similarity
- **Hybrid**: Combined approach using multiple fusion strategies

## Features

- 📊 Complete implementation of BM25, Dense, and Hybrid search
- 🔄 Multiple fusion strategies: RRF, Weighted, Max, CombSUM
- 📈 Evaluation framework with standard IR metrics (MAP, MRR, NDCG, Precision/Recall)
- 🚀 FastAPI REST API endpoints
- 🐳 Docker support for easy deployment
- 🧪 Comprehensive test suite
- 📋 Performance benchmarks

## Quick Start

### Installation

```bash
# Clone the repository
git clone <repo-url>
cd semantic-search

# Create virtual environment
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows

# Install dependencies
pip install -e ".[dev]"
```

### Basic Usage

```python
from src.bm25_search import BM25SearchEngine
from src.dense_search import DenseSearchEngine
from src.hybrid_search import HybridSearchEngine, FusionConfig

# Sample documents
documents = [
    {"id": "1", "title": "Machine Learning", "content": "ML is a subset of AI..."},
    {"id": "2", "title": "Deep Learning", "content": "Neural networks with layers..."},
    # ... more documents
]

# BM25 Search
bm25 = BM25SearchEngine()
bm25.index_documents(documents)
results = bm25.search("artificial intelligence", k=5)

# Dense Search
dense = DenseSearchEngine(model_name="all-MiniLM-L6-v2")
dense.index_documents(documents)
results = dense.search("AI and machine learning", k=5)

# Hybrid Search
hybrid = HybridSearchEngine(
    bm25_engine=bm25,
    dense_engine=dense,
    fusion_config=FusionConfig(method="rrf")
)
results = hybrid.search("neural networks", k=5)
```

### Running the API

```bash
# Start the API server
uvicorn src.api.main:app --reload

# Or using Docker
docker-compose up
```

API will be available at `http://localhost:8000` with interactive docs at `http://localhost:8000/docs`.

## Architecture

```
semantic-search/
├── src/
│   ├── __init__.py           # Package exports
│   ├── config.py             # Configuration settings
│   ├── base.py               # Base classes and interfaces
│   ├── bm25_search.py        # BM25 implementation
│   ├── dense_search.py       # Dense vector search
│   ├── hybrid_search.py      # Hybrid search with fusion
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── metrics.py        # IR evaluation metrics
│   │   └── benchmark.py      # Benchmark runner
│   └── api/
│       ├── __init__.py
│       ├── main.py           # FastAPI application
│       ├── routes.py         # API endpoints
│       └── models.py         # Pydantic request/response models
├── tests/
│   ├── __init__.py
│   ├── conftest.py           # Pytest fixtures
│   ├── test_bm25.py
│   ├── test_dense.py
│   ├── test_hybrid.py
│   ├── test_evaluation.py
│   └── test_api.py
├── benchmarks/
│   ├── __init__.py
│   └── run_benchmark.py      # Performance benchmarks
├── data/
│   └── sample_documents.json # Sample dataset
├── Dockerfile
├── docker-compose.yml
├── pyproject.toml
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

## Search Methods Explained

### 1. BM25 (Lexical Search)

BM25 (Best Matching 25) is a probabilistic ranking function that estimates document relevance:

**Formula:**
```
score(D, Q) = Σ IDF(qi) · (f(qi, D) · (k1 + 1)) / (f(qi, D) + k1 · (1 - b + b · |D|/avgdl))
```

**Characteristics:**
- Fast and efficient
- Exact keyword matching
- Good for known-item searches
- Handles document length bias

**Best for:**
- Keyword-heavy queries
- Technical searches
- When precision on exact terms matters

### 2. Dense Vector Search (Semantic)

Uses neural network embeddings to represent documents and queries in a shared semantic space:

**Process:**
1. Encode documents using a transformer model (e.g., SentenceTransformer)
2. Store embeddings in a vector index (FAISS)
3. Encode query and find nearest neighbors

**Characteristics:**
- Captures semantic meaning
- Handles synonyms and paraphrases
- Better for natural language queries
- Requires more computational resources

**Best for:**
- Conceptual searches
- Natural language queries
- Cross-lingual retrieval

### 3. Hybrid Search

Combines lexical and semantic approaches:

**Fusion Methods:**

| Method | Description |
|--------|-------------|
| **RRF** | Reciprocal Rank Fusion: `1/(k + rank)` |
| **Weighted** | Linear combination of normalized scores |
| **Max** | Takes maximum of normalized scores |
| **CombSUM** | Sum of normalized scores |

**Characteristics:**
- Best of both worlds
- More robust across query types
- Better recall and precision
- Slightly higher latency

## Evaluation Metrics

The evaluation framework implements standard IR metrics:

| Metric | Description | Range |
|--------|-------------|-------|
| **Precision@K** | Fraction of relevant docs in top K | 0-1 |
| **Recall@K** | Fraction of relevant docs retrieved | 0-1 |
| **F1@K** | Harmonic mean of precision and recall | 0-1 |
| **MAP** | Mean Average Precision across queries | 0-1 |
| **MRR** | Mean Reciprocal Rank of first relevant | 0-1 |
| **NDCG@K** | Normalized Discounted Cumulative Gain | 0-1 |

### Running Evaluation

```python
from src.evaluation.metrics import evaluate_retrieval

search_results = {
    "query1": ["doc1", "doc2", "doc3"],
    "query2": ["doc4", "doc5"]
}

ground_truth = {
    "query1": {"doc1", "doc3"},
    "query2": {"doc4"}
}

metrics = evaluate_retrieval(search_results, ground_truth)
print(f"MAP: {metrics['map']:.4f}")
print(f"NDCG@10: {metrics['ndcg@10']:.4f}")
```

## API Reference

### Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/v1/health` | Health check |
| GET | `/api/v1/status` | Index status |
| POST | `/api/v1/index` | Index documents |
| POST | `/api/v1/search` | Search with specified method |
| POST | `/api/v1/search/bm25` | BM25 search |
| POST | `/api/v1/search/dense` | Dense search |
| POST | `/api/v1/search/hybrid` | Hybrid search |
| POST | `/api/v1/compare` | Compare all methods |
| DELETE | `/api/v1/index` | Clear index |

### Example Requests

**Index Documents:**
```bash
curl -X POST http://localhost:8000/api/v1/index \
  -H "Content-Type: application/json" \
  -d '{
    "documents": [
      {"id": "1", "title": "ML Guide", "content": "Machine learning basics...", "metadata": {}}
    ]
  }'
```

**Search:**
```bash
curl -X POST http://localhost:8000/api/v1/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "machine learning",
    "k": 10,
    "method": "hybrid",
    "fusion_method": "rrf"
  }'
```

**Compare Methods:**
```bash
curl -X POST http://localhost:8000/api/v1/compare \
  -H "Content-Type: application/json" \
  -d '{"query": "neural networks", "k": 10}'
```

## Running Benchmarks

```bash
# Run full benchmark suite
python benchmarks/run_benchmark.py
```

Expected output:
```
============================================================
BENCHMARK COMPARISON
============================================================
Method               Avg Lat       P95 Lat       QPS          MAP        NDCG@10
------------------------------------------------------------
BM25                 0.15ms       0.32ms       6666.7      0.7500     0.8200
Dense                2.45ms       3.12ms       408.2       0.8200     0.8800
Hybrid (rrf)         2.62ms       3.45ms       381.7       0.8800     0.9200
============================================================
```

## Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test file
pytest tests/test_hybrid.py -v
```

## Configuration

Environment variables (`.env` file):

```env
# API Settings
API_HOST=0.0.0.0
API_PORT=8000
DEBUG=false

# Model Settings
EMBEDDING_MODEL=all-MiniLM-L6-v2
EMBEDDING_DIMENSION=384

# Search Settings
DEFAULT_TOP_K=10
BM25_K1=1.5
BM25_B=0.75
HYBRID_ALPHA=0.5
```

## Docker

```bash
# Build and run
docker-compose up

# Run benchmarks in Docker
docker-compose --profile benchmark up benchmark
```

## Performance Considerations

| Factor | BM25 | Dense | Hybrid |
|--------|------|-------|--------|
| Indexing Speed | Fast | Slow | Slow |
| Query Latency | ~0.1ms | ~2-5ms | ~3-6ms |
| Memory Usage | Low | Medium | Medium |
| Accuracy (Keyword) | High | Medium | High |
| Accuracy (Semantic) | Low | High | High |

## Dependencies

- **FastAPI**: Web framework for APIs
- **SentenceTransformers**: Neural embedding models
- **FAISS**: Efficient similarity search and vector indexing
- **rank-bm25**: BM25 implementation
- **NumPy/PyTorch**: Numerical computing

## License

MIT License

## References

- [BM25 Paper](https://doi.org/10.1561/1500000019)
- [Sentence Transformers](https://www.sbert.net/)
- [FAISS Library](https://github.com/facebookresearch/faiss)
- [Reciprocal Rank Fusion](https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf)
