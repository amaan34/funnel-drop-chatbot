import re
from typing import Dict, Tuple

from src.llm.llm_client import LLMClient

FORBIDDEN_PATTERNS = [
    r"\bguarantee\b",
    r"\bfinancial advice\b",
    r"\bmedical\b",
    r"\bblame\b",
    # Over-promising or misleading certainty
    r"\b100%\s*(success|guarantee)?\b",
    r"\balways works\b",
    r"\bnever fails?\b",
    # Discriminatory or harmful language
    r"\bracist\b",
    r"\bhate speech\b",
    r"\bdiscriminat(e|ion)\b",
    # Sensitive data leakage
    r"\baccount number\b",
    r"\bcard number\b",
    r"\b\d{12,19}\b",  # potential payment/ID numbers
]


class ComplianceValidator:
    """
    Lightweight compliance checks. Optionally uses LLM for deeper validation.
    """

    def __init__(self, llm_client: LLMClient | None = None) -> None:
        self.llm_client = llm_client

    def check_text(self, text: str) -> Tuple[bool, Dict]:
        issues: list[str] = []
        for pattern in FORBIDDEN_PATTERNS:
            if re.search(pattern, text, flags=re.IGNORECASE):
                issues.append(f"Contains forbidden pattern: {pattern}")

        if self.llm_client:
            try:
                verdict = self.llm_client.chat(
                    [
                        {"role": "system", "content": "Check text for compliance. Reply 'PASS' or 'FAIL' with reasons."},
                        {"role": "user", "content": text},
                    ]
                )
                if verdict and "FAIL" in verdict.upper():
                    issues.append("LLM flagged potential compliance risk")
            except Exception:
                # If LLM check fails, fall back to regex results
                pass

        return (len(issues) == 0, {"issues": issues})

