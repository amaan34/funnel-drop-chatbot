import json
from pathlib import Path

from src.evaluation.metrics import Evaluator


def load_test_cases(path: str = "tests/test_cases.json"):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    test_cases = load_test_cases()
    evaluator = Evaluator()

    # Placeholder: hook orchestrator here to fill retrieved_ids/source_ids/responses
    # For now, we demonstrate metric calculation with empty responses.
    relevance = evaluator.evaluate_rag_relevance(test_cases)
    hallucination_rate = evaluator.evaluate_hallucination([])
    json_correctness = evaluator.evaluate_json_correctness([])

    report = {
        "relevance": relevance,
        "hallucination_rate": hallucination_rate,
        "json_correctness": json_correctness,
    }
    Path("docs").mkdir(exist_ok=True)
    Path("docs/evaluation_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("Evaluation report written to docs/evaluation_report.json")


if __name__ == "__main__":
    main()

