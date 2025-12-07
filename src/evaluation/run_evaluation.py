import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from src.evaluation.llm_judge import LLMJudge
from src.evaluation.metrics import Evaluator
from src.llm.llm_client import LLMClient


def _maybe_llm_client() -> Optional[LLMClient]:
    try:
        return LLMClient()
    except Exception:
        return None


def _sample_inputs() -> Dict:
    """
    Lightweight synthetic inputs so the script can run without live traffic.
    """
    test_cases = [
        {"query": "Why did VKYC fail?", "expected_chunk_ids": ["faq_1"], "retrieved_ids": ["faq_1", "faq_2"]},
        {"query": "OTP not received", "expected_chunk_ids": ["otp_troubleshoot"], "retrieved_ids": ["otp_troubleshoot"]},
    ]

    responses = [
        {
            "predicted_drop_reason": "OCR quality issue",
            "explanation": "Blurry image",
            "steps_to_fix": ["Retake in good light"],
            "confidence_score": 0.82,
            "citations": [{"chunk_id": "faq_1"}],
            "source_ids": ["faq_1", "faq_2"],
            "latency_ms": 900,
            "token_cost": 0.004,
        },
        {
            "predicted_drop_reason": "Network delay",
            "explanation": "OTP lag",
            "steps_to_fix": ["Retry with stable network"],
            "confidence_score": 0.7,
            "citations": [{"chunk_id": "otp_troubleshoot"}],
            "source_ids": ["otp_troubleshoot"],
            "latency_ms": 750,
            "token_cost": 0.003,
        },
    ]

    nudges = [
        {"nudge_text": "Please retry VKYC in better lighting.", "nudge_type": "explanatory"},
        {"nudge_text": "Tap to retry OTP now.", "nudge_type": "cta_focused"},
    ]

    return {"test_cases": test_cases, "responses": responses, "nudges": nudges}


def run_evaluation(inputs: Optional[Dict] = None, output_path: str = "data/eval_report.json") -> Dict:
    evaluator = Evaluator()
    data = inputs or _sample_inputs()
    test_cases: List[Dict] = data["test_cases"]
    responses: List[Dict] = data["responses"]
    nudges: List[Dict] = data["nudges"]

    recall = evaluator.evaluate_rag_relevance(test_cases)
    hallucination_rate = evaluator.evaluate_hallucination(responses)
    json_correctness = evaluator.evaluate_json_correctness(responses)
    perf = evaluator.measure_performance(responses)

    llm_client = _maybe_llm_client()
    nudge_scores: Dict[str, float | int] = {"avg_score": 0.0, "count": 0}
    if llm_client:
        judge = LLMJudge(llm_client)
        scored = judge.score_nudges(nudges)
        nudge_scores = evaluator.evaluate_nudge_helpfulness(scored)

    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "recall_at_5": recall.get("recall_at_5"),
        "hallucination_rate": hallucination_rate,
        "json_correctness": json_correctness,
        "avg_latency_ms": perf.get("avg_latency_ms"),
        "avg_token_cost": perf.get("avg_token_cost"),
        "nudge_helpfulness": nudge_scores,
        "cases_evaluated": {
            "rag_cases": recall.get("cases"),
            "responses": len(responses),
            "nudges": len(nudges),
        },
        "llm_judge_used": bool(llm_client),
    }

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    return report


if __name__ == "__main__":  # pragma: no cover - manual execution
    result = run_evaluation()
    print(json.dumps(result, indent=2))

