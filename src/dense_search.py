"""
Dense Vector Search Engine Implementation.

Dense search uses neural network embeddings to represent documents and queries
as dense vectors in a high-dimensional space. Semantic similarity is computed
using vector distance metrics (typically cosine or L2 distance).

Key characteristics:
- Semantic understanding (handles synonyms, paraphrases)
- Captures conceptual relationships
- Better for natural language queries
- Requires more computational resources
"""
import numpy as np
from typing import List, Dict, Any, Optional, Union
import time

import faiss
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from .base import BaseSearchEngine, SearchResult, SearchResponse


class DenseSearchEngine(BaseSearchEngine):
    """
    Dense vector search engine using SentenceTransformers and FAISS.

    Uses pre-trained transformer models to encode documents and queries
    into dense vectors, then performs efficient similarity search using FAISS.
    """

    def __init__(
        self,
        name: str = "Dense",
        model_name: str = "all-MiniLM-L6-v2",
        index_type: str = "flat",
        normalize_embeddings: bool = True,
        device: str = None
    ):
        """
        Initialize dense search engine.

        Args:
            name: Name identifier for this engine
            model_name: SentenceTransformer model name
                - 'all-MiniLM-L6-v2': Fast, good quality (384 dims)
                - 'all-mpnet-base-v2': Best quality, slower (768 dims)
                - 'multi-qa-MiniLM-L6-cos-v1': Optimized for QA
            index_type: FAISS index type ('flat', 'ivf', 'hnsw')
            normalize_embeddings: Whether to L2-normalize embeddings
            device: Device to use ('cuda', 'cpu', or None for auto)
        """
        super().__init__(name)
        self.model_name = model_name
        self.index_type = index_type
        self.normalize_embeddings = normalize_embeddings

        # Load model
        print(f"Loading SentenceTransformer model: {model_name}")
        self.model = SentenceTransformer(model_name, device=device)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()

        # FAISS index
        self.faiss_index: Optional[faiss.Index] = None
        self.doc_embeddings: Optional[np.ndarray] = None
        self._doc_index: Dict[str, int] = {}  # doc_id -> position mapping

    def _build_doc_index(self) -> None:
        """Build document ID to index mapping for O(1) lookups."""
        self._doc_index = {
            doc.get('id', str(i)): i
            for i, doc in enumerate(self.documents)
        }

    def _encode_texts(
        self,
        texts: List[str],
        batch_size: int = 32,
        show_progress: bool = True
    ) -> np.ndarray:
        """
        Encode texts into dense vectors.

        Args:
            texts: List of texts to encode
            batch_size: Batch size for encoding
            show_progress: Whether to show progress bar

        Returns:
            Numpy array of embeddings (n_texts, embedding_dim)
        """
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            normalize_embeddings=self.normalize_embeddings,
            convert_to_numpy=True
        )
        return embeddings

    def _create_index(self, embeddings: np.ndarray) -> faiss.Index:
        """
        Create FAISS index based on configured type.

        Args:
            embeddings: Document embeddings

        Returns:
            FAISS index
        """
        n_vectors, dim = embeddings.shape

        if self.index_type == "flat":
            # Simple flat index - exact search, good for small datasets
            if self.normalize_embeddings:
                index = faiss.IndexFlatIP(dim)  # Inner product for normalized vectors
            else:
                index = faiss.IndexFlatL2(dim)
            index.add(embeddings.astype('float32'))

        elif self.index_type == "ivf":
            # IVF index - approximate search, faster for large datasets
            nlist = min(int(np.sqrt(n_vectors)), 1000)  # Number of clusters
            quantizer = faiss.IndexFlatL2(dim)
            index = faiss.IndexIVFFlat(quantizer, dim, nlist)

            # Train on the embeddings
            print(f"Training IVF index with {nlist} clusters...")
            index.train(embeddings.astype('float32'))
            index.add(embeddings.astype('float32'))

        elif self.index_type == "hnsw":
            # HNSW index - hierarchical navigable small world, very fast
            M = 32  # Number of connections per node
            index = faiss.IndexHNSWFlat(dim, M)
            index.add(embeddings.astype('float32'))

        else:
            raise ValueError(f"Unknown index type: {self.index_type}")

        return index

    def index_documents(
        self,
        documents: List[Dict[str, Any]],
        batch_size: int = 32,
        show_progress: bool = True
    ) -> None:
        """
        Index documents by encoding them into dense vectors.

        Args:
            documents: List of document dictionaries
            batch_size: Batch size for encoding
            show_progress: Whether to show progress bars
        """
        self.documents = documents

        # Prepare texts for encoding
        texts = []
        for doc in documents:
            # Combine title and content
            combined = f"{doc.get('title', '')} {doc.get('content', '')}"
            texts.append(combined)

        # Encode documents
        if show_progress:
            print(f"Encoding {len(documents)} documents with {self.model_name}...")

        self.doc_embeddings = self._encode_texts(
            texts,
            batch_size=batch_size,
            show_progress=show_progress
        )

        # Create FAISS index
        if show_progress:
            print(f"Creating FAISS {self.index_type} index...")

        self.faiss_index = self._create_index(self.doc_embeddings)

        # Build document index for fast lookups
        self._build_doc_index()

        self.is_indexed = True

        if show_progress:
            print(f"Indexed {len(documents)} documents successfully.")
            print(f"Embedding dimension: {self.embedding_dim}")

    def search(
        self,
        query: str,
        k: int = 10,
        nprobe: int = 10
    ) -> SearchResponse:
        """
        Search for documents using semantic similarity.

        Args:
            query: Search query string
            k: Number of results to return
            nprobe: Number of clusters to probe (for IVF index)

        Returns:
            SearchResponse with ranked results
        """
        if not self.is_indexed:
            raise RuntimeError("Documents not indexed. Call index_documents() first.")

        start_time = time.perf_counter()

        # Encode query
        query_embedding = self._encode_texts([query], show_progress=False)

        # Search FAISS index
        if self.index_type == "ivf":
            self.faiss_index.nprobe = nprobe

        scores, indices = self.faiss_index.search(query_embedding.astype('float32'), k)

        # Build results
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0:  # FAISS returns -1 for empty results
                doc = self.documents[idx]

                # Convert distance to similarity score
                if self.normalize_embeddings:
                    similarity = float(score)  # Inner product for normalized vectors
                else:
                    similarity = float(1 / (1 + score))  # Convert L2 distance to similarity

                results.append(SearchResult(
                    doc_id=doc.get('id', str(idx)),
                    score=similarity,
                    title=doc.get('title', ''),
                    content=doc.get('content', ''),
                    metadata=doc.get('metadata', {})
                ))

        end_time = time.perf_counter()
        latency_ms = (end_time - start_time) * 1000

        return SearchResponse(
            query=query,
            results=results,
            latency_ms=latency_ms,
            method=self.name,
            total_results=len(results)
        )

    def get_document_by_id(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a document by its ID."""
        idx = self._doc_index.get(doc_id)
        return self.documents[idx] if idx is not None else None

    def get_embedding(self, text: str) -> np.ndarray:
        """
        Get embedding for a single text.

        Args:
            text: Input text

        Returns:
            Numpy array of shape (embedding_dim,)
        """
        return self._encode_texts([text], show_progress=False)[0]

    def similarity(self, text1: str, text2: str) -> float:
        """
        Compute similarity between two texts.

        Args:
            text1: First text
            text2: Second text

        Returns:
            Similarity score (cosine similarity if normalized)
        """
        embeddings = self._encode_texts([text1, text2], show_progress=False)

        if self.normalize_embeddings:
            # Dot product of normalized vectors = cosine similarity
            similarity = float(np.dot(embeddings[0], embeddings[1]))
        else:
            # Cosine similarity
            similarity = float(np.dot(embeddings[0], embeddings[1]) /
                              (np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1])))

        return similarity

    def save_index(self, path: str) -> None:
        """Save FAISS index to disk."""
        if not self.is_indexed:
            raise RuntimeError("No index to save. Call index_documents() first.")
        faiss.write_index(self.faiss_index, path)

    def load_index(self, path: str) -> None:
        """Load FAISS index from disk."""
        self.faiss_index = faiss.read_index(path)
