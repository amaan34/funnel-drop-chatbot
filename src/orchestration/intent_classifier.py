from typing import Dict, List, Optional

from src.llm.llm_client import LLMClient

INTENTS = [
    "question_about_drop",
    "request_for_help",
    "general_inquiry",
    "complaint",
]


FEW_SHOTS = """
Classify the user's intent. Return JSON with fields: intent, confidence (0-1).

Examples:
User: Why was my VKYC rejected?
Intent: question_about_drop

User: Please help me finish my KYC.
Intent: request_for_help

User: What documents do I need for VKYC?
Intent: general_inquiry

User: This process is frustrating and keeps failing.
Intent: complaint
"""


class IntentClassifier:
    def __init__(self, llm_client: LLMClient) -> None:
        self.llm = llm_client

    def classify(self, query: str) -> Dict:
        prompt = f"{FEW_SHOTS}\n\nUser: {query}\nRespond with JSON."
        raw = self.llm.chat(
            [
                {"role": "system", "content": "Classify intent; only return JSON."},
                {"role": "user", "content": prompt},
            ]
        )
        intent = None
        confidence = 0.6
        try:
            import json

            parsed = json.loads(raw)
            intent = parsed.get("intent")
            confidence = float(parsed.get("confidence", confidence))
        except Exception:
            # fallback heuristic
            text = query.lower()
            if "why" in text or "reason" in text:
                intent = "question_about_drop"
            elif "help" in text or "how do i" in text:
                intent = "request_for_help"
            elif "frustrated" in text or "angry" in text:
                intent = "complaint"
            else:
                intent = "general_inquiry"

        if intent not in INTENTS:
            intent = "general_inquiry"
        return {"intent": intent, "confidence": confidence}

