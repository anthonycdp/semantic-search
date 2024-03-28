"""
BM25 Search Engine Implementation.

BM25 (Best Matching 25) is a probabilistic ranking function that estimates
the relevance of documents to a given search query. It's an improvement
over TF-IDF that accounts for document length normalization.

Key characteristics:
- Lexical matching (exact word matches)
- Fast and efficient
- Good for keyword-based queries
- Handles document length bias
"""
import re
from typing import List, Dict, Any, Optional
import time

from rank_bm25 import BM25Okapi
from tqdm import tqdm

from .base import BaseSearchEngine, SearchResult, SearchResponse


class BM25SearchEngine(BaseSearchEngine):
    """
    BM25-based search engine using rank_bm25 library.

    This implementation uses BM25Okapi which is optimized for general use cases
    and provides good default parameters.
    """

    def __init__(
        self,
        name: str = "BM25",
        tokenizer: str = "simple",
        k1: float = 1.5,
        b: float = 0.75
    ):
        """
        Initialize BM25 search engine.

        Args:
            name: Name identifier for this engine
            tokenizer: Tokenization method ('simple', 'word', 'nltk')
            k1: Term frequency saturation parameter (default: 1.5)
            b: Document length normalization parameter (default: 0.75)
        """
        super().__init__(name)
        self.tokenizer = tokenizer
        self.k1 = k1
        self.b = b
        self.bm25_index: Optional[BM25Okapi] = None
        self.tokenized_corpus: List[List[str]] = []
        self._doc_index: Dict[str, int] = {}  # doc_id -> position mapping

    def _build_doc_index(self) -> None:
        """Build document ID to index mapping for O(1) lookups."""
        self._doc_index = {
            doc.get('id', str(i)): i
            for i, doc in enumerate(self.documents)
        }

    def _tokenize(self, text: str) -> List[str]:
        """
        Tokenize text into a list of tokens.

        Args:
            text: Input text to tokenize

        Returns:
            List of lowercase tokens
        """
        # Simple tokenization: lowercase, split on non-alphanumeric
        text = text.lower()
        # Keep only alphanumeric characters and spaces
        text = re.sub(r'[^a-z0-9\s]', ' ', text)
        # Split and filter empty strings
        tokens = [token for token in text.split() if token]
        return tokens

    def _preprocess_documents(self, documents: List[Dict[str, Any]]) -> List[str]:
        """
        Preprocess documents for indexing.

        Combines title and content for better matching.

        Args:
            documents: List of document dictionaries

        Returns:
            List of combined text strings
        """
        processed = []
        for doc in documents:
            # Combine title and content with weight on title
            combined = f"{doc.get('title', '')} {doc.get('title', '')} {doc.get('content', '')}"
            processed.append(combined)
        return processed

    def index_documents(self, documents: List[Dict[str, Any]], show_progress: bool = True) -> None:
        """
        Index documents using BM25.

        Args:
            documents: List of document dictionaries with 'id', 'title', 'content' keys
            show_progress: Whether to show progress bar during indexing
        """
        self.documents = documents

        # Preprocess and tokenize
        processed_docs = self._preprocess_documents(documents)

        if show_progress:
            print(f"Tokenizing {len(documents)} documents...")
            self.tokenized_corpus = [
                self._tokenize(doc) for doc in tqdm(processed_docs)
            ]
        else:
            self.tokenized_corpus = [self._tokenize(doc) for doc in processed_docs]

        # Build BM25 index
        if show_progress:
            print("Building BM25 index...")

        self.bm25_index = BM25Okapi(self.tokenized_corpus)

        # Override default parameters if needed
        self.bm25_index.k1 = self.k1
        self.bm25_index.b = self.b

        # Build document index for fast lookups
        self._build_doc_index()

        self.is_indexed = True

        if show_progress:
            print(f"Indexed {len(documents)} documents successfully.")

    def search(self, query: str, k: int = 10) -> SearchResponse:
        """
        Search for documents using BM25 scoring.

        Args:
            query: Search query string
            k: Number of results to return

        Returns:
            SearchResponse with ranked results
        """
        if not self.is_indexed:
            raise RuntimeError("Documents not indexed. Call index_documents() first.")

        start_time = time.perf_counter()

        # Tokenize query
        query_tokens = self._tokenize(query)

        # Get BM25 scores
        scores = self.bm25_index.get_scores(query_tokens)

        # Get top-k indices
        top_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True
        )[:k]

        # Build results
        results = []
        for idx in top_indices:
            if scores[idx] != 0:  # Include documents with non-zero scores
                doc = self.documents[idx]
                results.append(SearchResult(
                    doc_id=doc.get('id', str(idx)),
                    score=float(scores[idx]),
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
        """
        Retrieve a document by its ID.

        Args:
            doc_id: Document identifier

        Returns:
            Document dictionary or None if not found
        """
        idx = self._doc_index.get(doc_id)
        return self.documents[idx] if idx is not None else None

    def explain_score(self, query: str, doc_id: str) -> Dict[str, Any]:
        """
        Explain the BM25 score for a specific document-query pair.

        Args:
            query: Search query
            doc_id: Document ID

        Returns:
            Dictionary with score explanation
        """
        if not self.is_indexed:
            raise RuntimeError("Documents not indexed. Call index_documents() first.")

        doc_idx = self._doc_index.get(doc_id)
        if doc_idx is None:
            return {'error': f'Document {doc_id} not found'}

        query_tokens = self._tokenize(query)
        doc_tokens = self.tokenized_corpus[doc_idx]

        # Calculate individual term scores
        term_scores = {}
        for token in set(query_tokens):
            if token in doc_tokens:
                tf = doc_tokens.count(token)
                term_scores[token] = {
                    'term_frequency': tf,
                    'in_document': True
                }
            else:
                term_scores[token] = {
                    'term_frequency': 0,
                    'in_document': False
                }

        scores = self.bm25_index.get_scores(query_tokens)
        final_score = scores[doc_idx]

        return {
            'query': query,
            'doc_id': doc_id,
            'final_score': float(final_score),
            'query_tokens': query_tokens,
            'doc_tokens_sample': doc_tokens[:20],
            'term_analysis': term_scores
        }
