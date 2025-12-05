from typing import Dict, List


class Evaluator:
    def evaluate_rag_relevance(self, test_cases: List[Dict]) -> Dict:
        """
        Expected test_case fields: {query, expected_chunk_ids: []}
        Returns recall@5 over cases.
        """
        hits = 0
        total = 0
        for case in test_cases:
            retrieved = case.get("retrieved_ids", [])
            expected = set(case.get("expected_chunk_ids", []))
            overlap = expected.intersection(set(retrieved[:5]))
            if expected:
                hits += 1 if overlap else 0
                total += 1
        recall_at_5 = hits / total if total else 0.0
        return {"recall_at_5": recall_at_5, "cases": total}

    def evaluate_hallucination(self, responses: List[Dict]) -> float:
        """
        Check if every citation id exists in provided source list.
        Each response should carry citations and a source_ids list.
        """
        violations = 0
        total = 0
        for resp in responses:
            citations = [c.get("chunk_id") for c in resp.get("citations", []) if c.get("chunk_id")]
            source_ids = set(resp.get("source_ids", []))
            if citations:
                total += 1
                if not all(cid in source_ids for cid in citations):
                    violations += 1
        return (violations / total) if total else 0.0

    def evaluate_json_correctness(self, responses: List[Dict]) -> float:
        required_fields = {
            "predicted_drop_reason",
            "explanation",
            "steps_to_fix",
            "confidence_score",
            "citations",
        }
        total = len(responses)
        if total == 0:
            return 0.0
        good = sum(1 for r in responses if required_fields.issubset(r.keys()))
        return good / total

    def evaluate_nudge_helpfulness(self, nudges: List[Dict]) -> Dict:
        """
        Placeholder for LLM-as-judge. Expects nudges with a `llm_score` already attached.
        """
        scores = [n.get("llm_score") for n in nudges if n.get("llm_score") is not None]
        if not scores:
            return {"avg_score": 0.0, "count": 0}
        avg_score = sum(scores) / len(scores)
        return {"avg_score": avg_score, "count": len(scores)}

    def measure_performance(self, responses: List[Dict]) -> Dict:
        latencies = [r.get("latency_ms") for r in responses if r.get("latency_ms") is not None]
        token_costs = [r.get("token_cost") for r in responses if r.get("token_cost") is not None]
        return {
            "avg_latency_ms": (sum(latencies) / len(latencies)) if latencies else 0.0,
            "avg_token_cost": (sum(token_costs) / len(token_costs)) if token_costs else 0.0,
        }

