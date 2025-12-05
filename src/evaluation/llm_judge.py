from typing import Dict, List

from src.llm.llm_client import LLMClient

JUDGE_PROMPT = """
Rate the nudge on helpfulness, clarity, empathy, and compliance (1-5 each).
Return JSON with fields: helpfulness, clarity, empathy, compliance, overall (average).
Text:
{nudge}
"""


class LLMJudge:
    def __init__(self, llm_client: LLMClient) -> None:
        self.llm = llm_client

    def score_nudges(self, nudges: List[Dict]) -> List[Dict]:
        scored: List[Dict] = []
        for nudge in nudges:
            prompt = JUDGE_PROMPT.format(nudge=nudge.get("nudge_text", ""))
            raw = self.llm.chat(
                [
                    {"role": "system", "content": "You are a strict evaluator. Return JSON only."},
                    {"role": "user", "content": prompt},
                ]
            )
            try:
                import json

                scores = json.loads(raw)
                nudge["llm_score"] = scores.get("overall")
                nudge["llm_score_breakdown"] = scores
            except Exception:
                nudge["llm_score"] = None
            scored.append(nudge)
        return scored

