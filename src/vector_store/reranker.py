import logging
from typing import Dict, List, Optional

from sentence_transformers import CrossEncoder

logger = logging.getLogger(__name__)


class CrossEncoderReranker:
    """
    Lightweight cross-encoder reranker using sentence-transformers.
    Scores each (query, chunk_content) pair and returns ranked chunks.
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        device: Optional[str] = None,
    ) -> None:
        self.model_name = model_name
        self.model = CrossEncoder(model_name, device=device)
        logger.info("Initialized cross-encoder reranker: %s", model_name)

    def rerank(self, query: str, results: List[Dict], top_k: int = 5) -> List[Dict]:
        if not results:
            return []

        pairs = [(query, r.get("content", "")) for r in results]
        scores = self.model.predict(pairs)

        reranked: List[Dict] = []
        for result, score in zip(results, scores):
            enriched = dict(result)
            enriched["rerank_score"] = float(score)
            reranked.append(enriched)

        reranked.sort(key=lambda r: r.get("rerank_score", 0.0), reverse=True)
        return reranked[:top_k]

