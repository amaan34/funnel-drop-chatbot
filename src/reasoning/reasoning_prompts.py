from typing import Dict, List


def build_drop_off_prompt(user_state: Dict, context_snippets: List[str]) -> str:
    context_text = "\n---\n".join(context_snippets)
    return f"""
You are a credit card onboarding support assistant.
Analyze why the user dropped off.

User State:
{user_state}

Retrieved Context:
{context_text}

Requirements:
- Think step-by-step.
- Identify primary reason and 2-3 secondary reasons.
- Return JSON with fields: primary_reason, secondary_reasons, confidence (0-1), reasoning_chain.
"""

