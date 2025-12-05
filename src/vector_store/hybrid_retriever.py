import logging
import re
from typing import Dict, List, Optional

from .bm25_store import BM25KeywordStore
from .chroma_store import ChromaVectorStore
from .reranker import CrossEncoderReranker

logger = logging.getLogger(__name__)


class HybridRetriever:
    """Combine semantic (Chroma) and keyword (BM25) search."""

    def __init__(
        self,
        chroma_store: ChromaVectorStore,
        bm25_store: BM25KeywordStore,
        embedding_generator,
        reranker: Optional[CrossEncoderReranker] = None,
    ) -> None:
        self.chroma_store = chroma_store
        self.bm25_store = bm25_store
        self.embedding_generator = embedding_generator
        self.reranker = reranker

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        semantic_weight: float = 0.7,
        keyword_weight: float = 0.3,
        metadata_filter: Optional[Dict] = None,
        strategy: str = "hybrid",
        apply_rerank: bool = False,
        rerank_top_k: int = 10,
    ) -> List[Dict]:
        if strategy == "semantic_only":
            results = self._semantic_search(query, top_k, metadata_filter)
        elif strategy == "keyword_only":
            results = self._keyword_search(query, top_k)
        elif strategy == "adaptive":
            results = self._adaptive_search(query, top_k, metadata_filter)
        else:
            results = self._hybrid_search(
                query, top_k, semantic_weight, keyword_weight, metadata_filter
            )

        if apply_rerank and self.reranker and results:
            rerank_limit = min(max(rerank_top_k, top_k), len(results))
            reranked = self.reranker.rerank(query, results, top_k=rerank_limit)
            return reranked[:top_k]

        return results[:top_k]

    def _semantic_search(
        self, query: str, top_k: int, metadata_filter: Optional[Dict]
    ) -> List[Dict]:
        results = self.chroma_store.query_by_text(
            query_text=query,
            embedding_generator=self.embedding_generator,
            n_results=top_k,
            where=metadata_filter,
        )
        for result in results:
            result["retrieval_method"] = "semantic"
        return results

    def _keyword_search(self, query: str, top_k: int) -> List[Dict]:
        results = self.bm25_store.search(query, top_k)
        if results:
            max_score = max(r["bm25_score"] for r in results)
            if max_score > 0:
                for result in results:
                    result["similarity_score"] = result["bm25_score"] / max_score
        for result in results:
            result["retrieval_method"] = "keyword"
        return results

    def _hybrid_search(
        self,
        query: str,
        top_k: int,
        semantic_weight: float,
        keyword_weight: float,
        metadata_filter: Optional[Dict],
    ) -> List[Dict]:
        semantic_results = self._semantic_search(query, top_k * 2, metadata_filter)
        keyword_results = self._keyword_search(query, top_k * 2)

        semantic_ranks = {r["chunk_id"]: i for i, r in enumerate(semantic_results)}
        keyword_ranks = {r["chunk_id"]: i for i, r in enumerate(keyword_results)}
        all_chunk_ids = set(semantic_ranks.keys()) | set(keyword_ranks.keys())

        k = 60
        rrf_scores = {}
        for chunk_id in all_chunk_ids:
            score = 0.0
            if chunk_id in semantic_ranks:
                score += semantic_weight / (k + semantic_ranks[chunk_id] + 1)
            if chunk_id in keyword_ranks:
                score += keyword_weight / (k + keyword_ranks[chunk_id] + 1)
            rrf_scores[chunk_id] = score

        ranked_chunk_ids = sorted(rrf_scores.keys(), key=lambda cid: rrf_scores[cid], reverse=True)[
            :top_k
        ]

        results: List[Dict] = []
        for chunk_id in ranked_chunk_ids:
            chunk_data = next((r for r in semantic_results if r["chunk_id"] == chunk_id), None)
            if not chunk_data:
                chunk_data = next((r for r in keyword_results if r["chunk_id"] == chunk_id), None)
            if chunk_data:
                chunk_data["rrf_score"] = rrf_scores[chunk_id]
                chunk_data["retrieval_method"] = "hybrid"
                chunk_data["in_semantic"] = chunk_id in semantic_ranks
                chunk_data["in_keyword"] = chunk_id in keyword_ranks
                results.append(chunk_data)
        return results

    def _adaptive_search(
        self, query: str, top_k: int, metadata_filter: Optional[Dict]
    ) -> List[Dict]:
        query_lower = query.lower()
        has_error_codes = bool(re.search(r"\b[A-Z][A-Z_]{2,}\b", query))
        technical_terms = ["ekyc", "vkyc", "otp", "pan", "aadhaar", "upi"]
        has_technical_terms = any(term in query_lower for term in technical_terms)
        question_words = ["why", "how", "what", "when", "where", "explain", "tell"]
        is_question = any(word in query_lower for word in question_words)

        if has_error_codes or has_technical_terms:
            logger.info("[Adaptive] Using keyword-heavy strategy")
            return self._hybrid_search(
                query,
                top_k,
                semantic_weight=0.3,
                keyword_weight=0.7,
                metadata_filter=metadata_filter,
            )
        if is_question:
            logger.info("[Adaptive] Using semantic-heavy strategy")
            return self._hybrid_search(
                query,
                top_k,
                semantic_weight=0.8,
                keyword_weight=0.2,
                metadata_filter=metadata_filter,
            )
        logger.info("[Adaptive] Using balanced strategy")
        return self._hybrid_search(
            query,
            top_k,
            semantic_weight=0.5,
            keyword_weight=0.5,
            metadata_filter=metadata_filter,
        )