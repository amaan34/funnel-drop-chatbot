import os
import pytest

from src.nudge.compliance_validator import ComplianceValidator
from src.orchestration.response_synthesizer import ResponseSynthesizer


requires_api_key = pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY"), reason="OPENAI_API_KEY not set"
)


@requires_api_key
def test_synthesizer_returns_citations():
    synthesizer = ResponseSynthesizer()
    reasoning = {"primary_reason": "Test", "reasoning_chain": "Because"}
    nudges = {}
    context = [
        {"chunk_id": "c1", "metadata": {"section": "faq"}, "content": "Step content"},
        {"chunk_id": "c2", "metadata": {"section": "faq"}, "content": "Another step"},
    ]
    result = synthesizer.synthesize(reasoning, nudges, context, confidence=0.9)
    assert result["citations"][0]["chunk_id"] == "c1"


def test_guardrail_flags_forbidden_patterns():
    validator = ComplianceValidator()
    passed, details = validator.check_text("This has 100% success and an account number 123456789012.")
    assert passed is False
    assert details["issues"]
    assert any("100%" in issue or "account number" in issue for issue in details["issues"])


def test_guardrail_allows_clean_text():
    validator = ComplianceValidator()
    passed, details = validator.check_text("Please retry the step with better lighting and a stable network.")
    assert passed is True
    assert details["issues"] == []

