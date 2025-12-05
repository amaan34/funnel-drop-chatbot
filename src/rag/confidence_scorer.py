from __future__ import annotations

import statistics
from typing import Dict, List, Tuple


class ConfidenceScorer:
    """
    Computes retrieval confidence to gate low-certainty responses.
    """

    def __init__(self, threshold: float = 0.6) -> None:
        self.threshold = threshold

    def score(self, results: List[Dict]) -> float:
        if not results:
            return 0.0

        similarity_scores = [
            r.get("similarity_score")
            for r in results
            if r.get("similarity_score") is not None
        ]
        rerank_scores = [
            r.get("rerank_score")
            for r in results
            if r.get("rerank_score") is not None
        ]

        avg_similarity = statistics.fmean(similarity_scores) if similarity_scores else 0.0
        max_similarity = max(similarity_scores) if similarity_scores else 0.0
        avg_rerank = statistics.fmean(rerank_scores) if rerank_scores else 1.0

        confidence = avg_similarity * max_similarity * avg_rerank
        return float(confidence)

    def is_low_confidence(self, confidence: float) -> bool:
        return confidence < self.threshold


def validate_citations(results: List[Dict]) -> Tuple[bool, List[str]]:
    """
    Basic citation validation to reduce hallucinations.
    Ensures each result includes a chunk_id and non-empty content.
    """
    issues: List[str] = []
    for idx, result in enumerate(results):
        if not result.get("chunk_id"):
            issues.append(f"Result {idx} missing chunk_id")
        if not result.get("content"):
            issues.append(f"Result {idx} missing content")
    return (len(issues) == 0, issues)

