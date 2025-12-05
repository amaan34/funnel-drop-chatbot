from typing import Dict, List


def build_drop_off_prompt(user_state: Dict, context_snippets: List[str]) -> str:
    context_text = "\n---\n".join(context_snippets)
    stage = user_state.get("stage_dropped") or user_state.get("stage") or "unknown"
    device = user_state.get("device_type") or user_state.get("device") or "unknown"
    timestamp = user_state.get("timestamp") or "unknown"

    return f"""
You are a credit card onboarding support assistant.
Analyze why the user dropped off.
Explicitly use device type and time-of-day effects (network quality / SMS delivery windows).

User State:
{user_state}

Key Signals:
- Stage: {stage}
- Device: {device}
- Timestamp: {timestamp}

Retrieved Context:
{context_text}

Requirements:
- Think step-by-step.
- Identify primary reason and 2-3 secondary reasons.
- Consider device-specific issues (Android/iOS) and time-based network issues when relevant.
- Output must be a single JSON object only (no prose, no markdown, no code fences).
- JSON keys (all required):
  - primary_reason: concise string
  - secondary_reasons: array of 2-3 concise strings
  - confidence: number between 0 and 1
  - reasoning_chain: short step-by-step string
- Respond with only the JSON object and nothing else.
"""

