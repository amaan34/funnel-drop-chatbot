from typing import Dict

from src.nudge.compliance_validator import ComplianceValidator


class GuardrailValidator:
    """
    Applies simple guardrails to the final response using compliance checks.
    """

    def __init__(self, compliance_validator: ComplianceValidator) -> None:
        self.compliance_validator = compliance_validator

    def validate(self, response: Dict) -> Dict:
        conversational = response.get("conversational_message", "")
        ok, details = self.compliance_validator.check_text(conversational)
        response["compliance"] = {"passed": ok, "details": details}
        return response

