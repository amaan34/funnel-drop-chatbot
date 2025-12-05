import json
import logging
from typing import Dict, List, Optional

from src.llm.llm_client import LLMClient
from src.reasoning.reasoning_prompts import build_drop_off_prompt

logger = logging.getLogger(__name__)


RULE_BASED_HINTS = {
    ("VKYC", "OCR_FAIL"): "OCR quality issue during video KYC (blurry document/lighting).",
    ("VKYC", "POOR_LIGHT"): "Low light causing verification failure during VKYC.",
    ("OTP", "OTP_TIMEOUT"): "OTP not received or expired before entry.",
    ("OTP", "OTP_INVALID"): "Incorrect OTP entered multiple times.",
    ("eKYC", "AADHAAR_MISMATCH"): "Aadhaar details mismatch during eKYC.",
}


class DropOffReasoner:
    """
    Combines rule-based hints with LLM chain-of-thought reasoning to explain drop-offs.
    """

    def __init__(self, llm_client: Optional[LLMClient] = None) -> None:
        self.llm_client = llm_client

    def _rule_based_reason(self, user_state: Dict) -> Optional[str]:
        stage = user_state.get("stage_dropped")
        error_codes = user_state.get("error_codes") or []
        for code in error_codes:
            reason = RULE_BASED_HINTS.get((stage, code))
            if reason:
                return reason
        return None

    def _context_to_snippets(self, retrieved_context: List[Dict], limit: int = 3) -> List[str]:
        return [c.get("content", "") for c in retrieved_context[:limit] if c.get("content")]

    def analyze(self, user_state: Dict, retrieved_context: List[Dict]) -> Dict:
        rule_reason = self._rule_based_reason(user_state)
        context_snippets = self._context_to_snippets(retrieved_context)

        llm_output = None
        if self.llm_client:
            prompt = build_drop_off_prompt(user_state, context_snippets)
            try:
                raw = self.llm_client.chat(
                    [
                        {"role": "system", "content": "You are a precise analyst. Always return JSON."},
                        {"role": "user", "content": prompt},
                    ]
                )
                llm_output = json.loads(raw)
            except Exception as exc:  # noqa: BLE001
                logger.warning("LLM reasoning failed, falling back to rules. Error: %s", exc)

        primary_reason = (
            llm_output.get("primary_reason")
            if isinstance(llm_output, dict) and llm_output.get("primary_reason")
            else rule_reason
            or "Unable to determine primary reason"
        )

        secondary_reasons = (
            llm_output.get("secondary_reasons")
            if isinstance(llm_output, dict) and llm_output.get("secondary_reasons")
            else []
        )

        confidence = (
            float(llm_output.get("confidence"))
            if isinstance(llm_output, dict) and llm_output.get("confidence") is not None
            else (0.78 if rule_reason else 0.6)
        )

        reasoning_chain = (
            llm_output.get("reasoning_chain")
            if isinstance(llm_output, dict) and llm_output.get("reasoning_chain")
            else "Rule-based heuristic applied" if rule_reason else "No rule match; default reasoning"
        )

        return {
            "primary_reason": primary_reason,
            "secondary_reasons": secondary_reasons,
            "confidence": confidence,
            "reasoning_chain": reasoning_chain,
        }

