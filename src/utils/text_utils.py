import re
from typing import List, Optional


def initial_clean(text: str) -> str:
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"\n\d+\n", "\n", text)
    text = text.replace("—", "-")
    return text.strip()


def normalize_stage_name(stage: str) -> str:
    stage_map = {
        "ekyc": "eKYC",
        "e-kyc": "eKYC",
        "kyc": "eKYC",
        "vkyc": "VKYC",
        "video kyc": "VKYC",
        "v-kyc": "VKYC",
        "otp": "OTP",
        "activation": "OTP",
        "liveliness": "Liveliness",
        "liveness": "Liveliness",
        "additional details": "Additional_Details",
        "additional_details": "Additional_Details",
    }
    return stage_map.get(stage.lower(), stage)


def get_stage_priority(stage: str) -> int:
    ordered = [
        "eKYC",
        "Liveliness",
        "Additional_Details",
        "VKYC",
        "OTP",
        "General",
    ]
    try:
        return ordered.index(normalize_stage_name(stage))
    except ValueError:
        return len(ordered)


def extract_error_codes(text: str) -> List[str]:
    pattern = r"\b[A-Z][A-Z_]{2,}\b"
    codes = re.findall(pattern, text)
    common_words = {"AND", "THE", "FOR", "CARD", "KYC", "OTP", "PAN", "UPI", "CSB", "INR"}
    return [code for code in codes if code not in common_words]


def extract_timeline_mentions(text: str) -> Optional[List[str]]:
    timeline_patterns = [
        r"\d+\s*(?:day|days)",
        r"\d+\s*(?:hour|hours)",
        r"\d+\s*(?:month|months)",
        r"\d+\s*(?:AM|PM|am|pm)",
        r"within\s+\d+",
        r"before\s+\d+",
        r"after\s+\d+",
    ]
    timelines: List[str] = []
    for pattern in timeline_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        timelines.extend(matches)
    return timelines if timelines else None


def detect_tone(text: str) -> str:
    text_lower = text.lower()
    if any(word in text_lower for word in ["congratulations", "pleased", "glad", "happy"]):
        return "reassuring"
    if any(word in text_lower for word in ["please", "must", "required", "ensure"]):
        return "procedural"
    if any(word in text_lower for word in ["however", "because", "reason", "due to"]):
        return "explanatory"
    return "neutral"


def has_actionable_steps(text: str) -> bool:
    action_indicators = [
        "step",
        "click",
        "open",
        "enter",
        "upload",
        "complete",
        "ensure",
        "verify",
        "provide",
        "contact",
        "retry",
    ]
    text_lower = text.lower()
    return any(indicator in text_lower for indicator in action_indicators)


def estimate_tokens(text: str) -> int:
    return max(len(text) // 4, 1)

