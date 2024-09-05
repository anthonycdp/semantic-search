"""
API routes for semantic search endpoints.
"""
from typing import Optional

from fastapi import APIRouter, HTTPException, BackgroundTasks

from .models import (
    search_engines, Document, IndexRequest, SearchRequest,
    SearchResult, SearchResponse, IndexStatus, CompareRequest, CompareResponse
)
from ..bm25_search import BM25SearchEngine
from ..dense_search import DenseSearchEngine
from ..hybrid_search import HybridSearchEngine, FusionConfig


router = APIRouter()


@router.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}


@router.get("/status", response_model=IndexStatus)
async def get_status():
    """Get current index status."""
    indexed = "bm25" in search_engines and search_engines["bm25"].is_indexed
    num_docs = len(search_engines.get("bm25", {}).documents) if indexed else 0

    return IndexStatus(
        indexed=indexed,
        num_documents=num_docs,
        methods_available=list(search_engines.keys())
    )


@router.post("/index")
async def index_documents(request: IndexRequest, background_tasks: BackgroundTasks):
    """
    Index documents for search.

    This creates BM25, dense, and hybrid search indexes.
    """
    # Convert Pydantic models to dicts
    docs = [doc.model_dump() for doc in request.documents]

    if len(docs) == 0:
        raise HTTPException(status_code=400, detail="No documents provided")

    # Create BM25 engine
    bm25_engine = BM25SearchEngine(name="BM25")
    bm25_engine.index_documents(docs, show_progress=False)
    search_engines["bm25"] = bm25_engine

    # Create dense engine
    dense_engine = DenseSearchEngine(
        name="Dense",
        model_name=request.model_name,
        index_type=request.index_type
    )
    dense_engine.index_documents(docs, batch_size=32, show_progress=False)
    search_engines["dense"] = dense_engine

    # Create hybrid engine
    fusion_config = FusionConfig()
    hybrid_engine = HybridSearchEngine(
        name="Hybrid",
        bm25_engine=bm25_engine,
        dense_engine=dense_engine,
        fusion_config=fusion_config
    )
    hybrid_engine.is_indexed = True  # Already indexed via sub-engines
    hybrid_engine.documents = docs
    search_engines["hybrid"] = hybrid_engine

    return {
        "status": "success",
        "num_documents": len(docs),
        "methods": ["bm25", "dense", "hybrid"]
    }


@router.post("/search", response_model=SearchResponse)
async def search(request: SearchRequest):
    """
    Search documents using specified method.

    Supported methods: bm25, dense, hybrid
    """
    method = request.method.lower()

    if method not in search_engines:
        raise HTTPException(
            status_code=400,
            detail=f"Method '{method}' not available. Initialize with /index first."
        )

    engine = search_engines[method]

    if not engine.is_indexed:
        raise HTTPException(
            status_code=400,
            detail="Index not initialized. Call /index first."
        )

    # Configure hybrid search if needed
    if method == "hybrid" and request.fusion_method:
        engine.set_fusion_method(request.fusion_method)
        if request.bm25_weight and request.dense_weight:
            engine.set_fusion_weights(request.bm25_weight, request.dense_weight)

    # Perform search
    response = engine.search(request.query, k=request.k)

    return SearchResponse(
        query=response.query,
        results=[
            SearchResult(
                doc_id=r.doc_id,
                score=r.score,
                title=r.title,
                content=r.content,
                metadata=r.metadata or {}
            )
            for r in response.results
        ],
        latency_ms=response.latency_ms,
        method=response.method,
        total_results=response.total_results
    )


@router.post("/search/bm25", response_model=SearchResponse)
async def search_bm25(query: str, k: int = 10):
    """Search using BM25 method."""
    return await search(SearchRequest(query=query, k=k, method="bm25"))


@router.post("/search/dense", response_model=SearchResponse)
async def search_dense(query: str, k: int = 10):
    """Search using dense embeddings."""
    return await search(SearchRequest(query=query, k=k, method="dense"))


@router.post("/search/hybrid", response_model=SearchResponse)
async def search_hybrid(
    query: str,
    k: int = 10,
    fusion: str = "rrf",
    bm25_weight: float = 0.5,
    dense_weight: float = 0.5
):
    """Search using hybrid method."""
    return await search(SearchRequest(
        query=query,
        k=k,
        method="hybrid",
        fusion_method=fusion,
        bm25_weight=bm25_weight,
        dense_weight=dense_weight
    ))


@router.post("/compare", response_model=CompareResponse)
async def compare_methods(request: CompareRequest):
    """
    Compare all search methods side by side.

    Returns results from BM25, Dense, and Hybrid search for comparison.
    """
    if "hybrid" not in search_engines:
        raise HTTPException(
            status_code=400,
            detail="Index not initialized. Call /index first."
        )

    # Get results from all methods
    bm25_response = search_engines["bm25"].search(request.query, k=request.k)
    dense_response = search_engines["dense"].search(request.query, k=request.k)
    hybrid_response = search_engines["hybrid"].search(request.query, k=request.k)

    def to_api_response(response):
        return SearchResponse(
            query=response.query,
            results=[
                SearchResult(
                    doc_id=r.doc_id,
                    score=r.score,
                    title=r.title,
                    content=r.content,
                    metadata=r.metadata or {}
                )
                for r in response.results
            ],
            latency_ms=response.latency_ms,
            method=response.method,
            total_results=response.total_results
        )

    return CompareResponse(
        query=request.query,
        bm25=to_api_response(bm25_response),
        dense=to_api_response(dense_response),
        hybrid=to_api_response(hybrid_response)
    )


@router.delete("/index")
async def clear_index():
    """Clear all indexed documents."""
    global search_engines
    search_engines.clear()

    return {"status": "success", "message": "Index cleared"}
