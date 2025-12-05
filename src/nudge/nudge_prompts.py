def build_explanatory_prompt(drop_reason: str, stage: str) -> str:
    return f"""
Create a clear, educational explanation about: {drop_reason}
User stage: {stage}
Tone: helpful, non-blaming, concise.
Respond in English.
"""


def build_cta_prompt(stage: str) -> str:
    return f"""
Write a short action-oriented message (1-2 sentences) to help the user complete {stage}.
Include one specific next step. Tone: direct and supportive. Respond in English.
"""


def build_empathetic_prompt(drop_reason: str, stage: str) -> str:
    return f"""
Write a reassuring message about: {drop_reason}
Stage: {stage}
Tone: empathetic, trust-building, and concise. Avoid blame.
Respond in English.
"""


def build_translation_prompt(text: str, target_language: str = "Hindi") -> str:
    return f"Translate the following text to {target_language} while keeping the meaning and tone:\n\n{text}"

