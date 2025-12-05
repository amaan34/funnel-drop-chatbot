import json
from pathlib import Path
from typing import Dict, List

from src.llm.llm_client import LLMClient

EXAMPLES = [
    {
        "user_state": {"stage_dropped": "eKYC", "error_codes": ["eKYC_FAIL"], "device_type": "Android"},
        "query": "Why was my eKYC rejected?",
    },
    {
        "user_state": {"stage_dropped": "OTP", "error_codes": ["OTP_TIMEOUT"], "device_type": "iOS"},
        "query": "I never got the OTP. What happened?",
    },
]


def build_prompt(user_state: Dict, query: str) -> str:
    return f"""
Generate one JSON training example for the funnel drop chatbot.
Fields: predicted_drop_reason, explanation, steps_to_fix (list), confidence_score (0-1), citations (list).
User state: {user_state}
Query: {query}
Return JSON only.
"""


def generate_examples(num_examples: int = 5, output_path: str = "data/fine_tuning/training_dataset.jsonl") -> None:
    llm = LLMClient()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    rows: List[str] = []
    seeds = EXAMPLES * max(1, num_examples // len(EXAMPLES))
    for idx, example in enumerate(seeds[:num_examples]):
        prompt = build_prompt(example["user_state"], example["query"])
        result = llm.chat(
            [
                {"role": "system", "content": "You are a precise data generator. Return JSON only."},
                {"role": "user", "content": prompt},
            ]
        )
        messages = [
            {"role": "system", "content": "You are a credit card onboarding support assistant."},
            {
                "role": "user",
                "content": f"User state: {example['user_state']}\nQuery: {example['query']}",
            },
            {"role": "assistant", "content": result},
        ]
        rows.append(json.dumps({"messages": messages}))

    Path(output_path).write_text("\n".join(rows), encoding="utf-8")
    print(f"Wrote {len(rows)} examples to {output_path}")


if __name__ == "__main__":
    generate_examples(num_examples=10)

