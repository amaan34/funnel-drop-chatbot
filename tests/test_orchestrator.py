import os
import pytest

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

