import json
import logging
from datetime import datetime, timezone
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

# Device- and time-aware heuristics for more granular explanations.
DEVICE_HINTS = {
    ("VKYC", "android"): "Camera permissions or device-specific camera quality issues on Android during VKYC.",
    ("VKYC", "ios"): "Camera access or focus issues on iOS during VKYC.",
    ("OTP", "android"): "Android SMS interception disabled or network jitter affecting OTP delivery.",
    ("OTP", "ios"): "iOS SMS filtering or delayed push/SMS delivery affecting OTP.",
    ("EKYC", "android"): "On Android eKYC, camera/flash permissions or unstable network can block OCR.",
    ("EKYC", "ios"): "On iOS eKYC, ensure camera permissions and stable lighting for OCR.",
}

TIME_BUCKET_HINTS = {
    "late_night": "Late-night hours can have weaker network quality causing verification to fail.",
    "peak_evening": "Peak evening traffic may slow down OTP/video verification responses.",
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

    def _device_based_reason(self, user_state: Dict) -> Optional[str]:
        stage = (user_state.get("stage_dropped") or "").upper()
        device = (user_state.get("device_type") or user_state.get("device") or "").lower()
        if not stage or not device:
            return None
        return DEVICE_HINTS.get((stage, device))

    def _time_bucket(self, timestamp_str: Optional[str]) -> Optional[str]:
        if not timestamp_str:
            return None
        try:
            ts = (
                datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
                if isinstance(timestamp_str, str)
                else None
            )
            if not ts:
                return None
            ts = ts.astimezone(timezone.utc)
            hour = ts.hour
            if 0 <= hour < 6:
                return "late_night"
            if 18 <= hour <= 23:
                return "peak_evening"
        except Exception:  # noqa: BLE001
            return None
        return None

    def _time_based_reason(self, user_state: Dict) -> Optional[str]:
        bucket = self._time_bucket(user_state.get("timestamp"))
        if not bucket:
            return None
        return TIME_BUCKET_HINTS.get(bucket)

    def _context_to_snippets(self, retrieved_context: List[Dict], limit: int = 3) -> List[str]:
        return [c.get("content", "") for c in retrieved_context[:limit] if c.get("content")]

    def analyze(self, user_state: Dict, retrieved_context: List[Dict]) -> Dict:
        rule_reason = self._rule_based_reason(user_state)
        device_reason = self._device_based_reason(user_state)
        time_reason = self._time_based_reason(user_state)
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

        primary_reason = None
        if isinstance(llm_output, dict) and llm_output.get("primary_reason"):
            primary_reason = llm_output.get("primary_reason")
        else:
            primary_reason = rule_reason or device_reason or time_reason or "Unable to determine primary reason"

        secondary_reasons = (
            llm_output.get("secondary_reasons")
            if isinstance(llm_output, dict) and llm_output.get("secondary_reasons")
            else [r for r in [rule_reason, device_reason, time_reason] if r and r != primary_reason]
        )

        confidence = (
            float(llm_output.get("confidence"))
            if isinstance(llm_output, dict) and llm_output.get("confidence") is not None
            else (0.78 if rule_reason else 0.7 if device_reason else 0.65 if time_reason else 0.6)
        )

        reasoning_chain = (
            llm_output.get("reasoning_chain")
            if isinstance(llm_output, dict) and llm_output.get("reasoning_chain")
            else (
                "Rule-based heuristic applied"
                if rule_reason
                else "Device/time heuristic applied"
                if device_reason or time_reason
                else "No rule match; default reasoning"
            )
        )

        return {
            "primary_reason": primary_reason,
            "secondary_reasons": secondary_reasons,
            "confidence": confidence,
            "reasoning_chain": reasoning_chain,
        }

