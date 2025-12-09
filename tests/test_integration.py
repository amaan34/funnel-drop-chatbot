from src.orchestration.response_synthesizer import ResponseSynthesizer


def test_response_synthesizer_low_confidence():
    synthesizer = ResponseSynthesizer()
    reasoning = {"primary_reason": "Test reason", "reasoning_chain": "Chain"}
    nudges = {}
    context = [{"chunk_id": "c1", "metadata": {"section": "faq"}, "content": "step 1"}]
    result = synthesizer.synthesize(reasoning, nudges, context, confidence=0.2, low_confidence=True)
    assert result["low_confidence"] is True
    assert "not fully sure" in result["conversational_message"].lower()

