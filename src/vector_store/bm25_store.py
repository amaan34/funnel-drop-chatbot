import logging
import pickle
import re
from typing import Dict, List

from rank_bm25 import BM25Okapi

logger = logging.getLogger(__name__)


class BM25KeywordStore:
    """Keyword search using BM25 for exact term matches."""

    def __init__(self) -> None:
        self.bm25: BM25Okapi | None = None
        self.chunks: List[Dict] = []
        self.tokenized_corpus: List[List[str]] = []

    def tokenize(self, text: str) -> List[str]:
        """Lowercase tokenize while preserving all-caps codes."""
        words = text.split()
        tokens: List[str] = []
        for word in words:
            if word.isupper() and len(word) > 1:
                tokens.append(word)
            cleaned = re.sub(r"[^\w\s]", " ", word.lower())
            tokens.extend(cleaned.split())
        return [t for t in tokens if t]

    def build_index(self, chunks: List[Dict]) -> None:
        """Build the BM25 index from chunks."""
        self.chunks = chunks
        corpus_texts = [chunk["content"] for chunk in chunks]
        self.tokenized_corpus = [self.tokenize(text) for text in corpus_texts]
        self.bm25 = BM25Okapi(self.tokenized_corpus)
        logger.info("BM25 index built with %s documents", len(chunks))

    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """Search using BM25."""
        if not self.bm25:
            raise ValueError("BM25 index not built. Call build_index() first.")
        query_tokens = self.tokenize(query)
        scores = self.bm25.get_scores(query_tokens)
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

        results: List[Dict] = []
        for idx in top_indices:
            if scores[idx] > 0:
                results.append(
                    {
                        "chunk_id": self.chunks[idx]["chunk_id"],
                        "content": self.chunks[idx]["content"],
                        "metadata": self.chunks[idx].get("metadata", {}),
                        "bm25_score": float(scores[idx]),
                    }
                )
        return results

    def save_index(self, filepath: str = "data/bm25_index.pkl") -> None:
        """Persist the BM25 index."""
        with open(filepath, "wb") as f:
            pickle.dump(
                {
                    "bm25": self.bm25,
                    "chunks": self.chunks,
                    "tokenized_corpus": self.tokenized_corpus,
                },
                f,
            )
        logger.info("BM25 index saved to %s", filepath)

    def load_index(self, filepath: str = "data/bm25_index.pkl") -> None:
        """Load a persisted BM25 index."""
        with open(filepath, "rb") as f:
            data = pickle.load(f)
        self.bm25 = data["bm25"]
        self.chunks = data["chunks"]
        self.tokenized_corpus = data["tokenized_corpus"]
        logger.info("BM25 index loaded from %s", filepath)