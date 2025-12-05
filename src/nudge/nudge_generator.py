from typing import Dict, List, Optional

from src.llm.llm_client import LLMClient
from src.nudge.compliance_validator import ComplianceValidator
from src.nudge import nudge_prompts as prompts


class NudgeGenerator:
    """
    Generates nudges in three styles (explanatory, CTA-focused, empathetic) with bilingual output.
    """

    def __init__(
        self,
        llm_client: LLMClient,
        compliance_validator: Optional[ComplianceValidator] = None,
    ) -> None:
        self.llm = llm_client
        self.validator = compliance_validator or ComplianceValidator(llm_client)

    def _derive_stage(self, user_state: Dict) -> str:
        """
        Resolve the most relevant stage information from user_state, handling
        both flat and nested representations.
        """
        if not isinstance(user_state, dict):
            return "onboarding"

        for key in ("stage_dropped", "stage", "current_stage"):
            stage = user_state.get(key)
            if stage:
                return str(stage)

        for nested_key in ("resume_info", "funnel_state", "metadata"):
            nested = user_state.get(nested_key)
            if isinstance(nested, dict):
                for key in ("stage_dropped", "stage", "current_stage", "last_completed_stage"):
                    stage = nested.get(key)
                    if stage:
                        return str(stage)

        return "onboarding"

    def _generate_text(self, prompt: str) -> str:
        return self.llm.chat(
            [
                {"role": "system", "content": "You are a concise onboarding support assistant."},
                {"role": "user", "content": prompt},
            ]
        )

    def _translate(self, text: str, target_language: str) -> str:
        if target_language.lower() == "english":
            return text
        translation_prompt = prompts.build_translation_prompt(text, target_language=target_language.capitalize())
        return self._generate_text(translation_prompt)

    def generate(self, drop_reason: str, user_state: Dict, nudge_type: str, language: str = "english") -> Dict:
        stage = self._derive_stage(user_state)

        if nudge_type == "explanatory":
            prompt = prompts.build_explanatory_prompt(drop_reason, stage)
        elif nudge_type == "cta_focused":
            prompt = prompts.build_cta_prompt(stage)
        elif nudge_type == "empathetic":
            prompt = prompts.build_empathetic_prompt(drop_reason, stage)
        elif nudge_type == "compliance_safe":
            prompt = prompts.build_compliance_safe_prompt(drop_reason, stage)
        else:
            raise ValueError(f"Unsupported nudge_type: {nudge_type}")

        english_text = self._generate_text(prompt)
        target_text = self._translate(english_text, target_language=language)

        is_compliant, details = self.validator.check_text(target_text)

        return {
            "nudge_text": target_text,
            "language": language,
            "nudge_type": nudge_type,
            "compliance_check_passed": is_compliant,
            "compliance_details": details,
            "english_reference": english_text,
        }

    def generate_all(self, drop_reason: str, user_state: Dict, language: str = "english") -> Dict[str, Dict]:
        outputs: Dict[str, Dict] = {}
        for nudge_type in ["explanatory", "cta_focused", "empathetic", "compliance_safe"]:
            outputs[nudge_type] = self.generate(drop_reason, user_state, nudge_type, language)
        return outputs

