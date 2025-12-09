from typing import Dict

from src.nudge.compliance_validator import ComplianceValidator
from src.nudge.nudge_generator import NudgeGenerator
from src.orchestration.chatbot_orchestrator import ChatbotOrchestrator
from src.orchestration.response_synthesizer import ResponseSynthesizer
from src.reasoning.drop_off_reasoner import DropOffReasoner


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


class FakeLLMClient:
    def __init__(self) -> None:
        self.calls: list[Dict] = []

    def chat(self, messages, temperature=None):
        self.calls.append({"messages": messages, "temperature": temperature})
        content = messages[-1]["content"] if messages else ""
        if "Classify the user's intent" in content or "Classify intent" in content:
            return '{"intent": "request_for_help", "confidence": 0.9}'
        if "Translate the following text" in content:
            return "Translated text"
        return (
            '{"primary_reason": "Mock reason", "secondary_reasons": ["rule"], '
            '"confidence": 0.85, "reasoning_chain": "Mock chain"}'
        )


def test_orchestrator_process_happy_path():
    class StubRetriever:
        def retrieve(
            self, query, top_k=5, metadata_filter=None, strategy=None, apply_rerank=None, rerank_top_k=None
        ):
            return [
                {
                    "chunk_id": "c1",
                    "content": "Step 1: follow the instructions carefully.",
                    "metadata": {"section": "faq"},
                    "similarity_score": 0.9,
                    "rerank_score": 0.85,
                }
            ]

    class StubReranker:
        pass

    orchestrator = ChatbotOrchestrator(
        retriever=StubRetriever(),
        reranker=StubReranker(),
        llm_client=FakeLLMClient(),
    )

    captured_state: Dict = {}

    def fake_classify(q: str) -> Dict:
        captured_state["classified_query"] = q
        return {"intent": "request_for_help", "confidence": 0.91}

    def fake_analyze(state: Dict, results):
        captured_state["passed_intent"] = state.get("intent")
        return {
            "primary_reason": "Network issue",
            "secondary_reasons": ["Network jitter"],
            "confidence": 0.88,
            "reasoning_chain": "Mock reasoning chain",
        }

    orchestrator.intent_classifier.classify = fake_classify
    orchestrator.drop_off_reasoner.analyze = fake_analyze

    orchestrator.nudge_generator.generate_all = lambda reason, user_state, language="english": {
        "cta_focused": {
            "nudge_text": "Please retry the step.",
            "language": language,
            "nudge_type": "cta_focused",
            "compliance_check_passed": True,
            "compliance_details": {"issues": []},
            "english_reference": "Please retry the step.",
        }
    }

    user_state = {"stage_dropped": "OTP", "language": "english"}
    response = orchestrator.process(user_state, "Help me finish KYC")

    assert captured_state["classified_query"] == "Help me finish KYC"
    assert captured_state["passed_intent"]["intent"] == "request_for_help"
    assert response["intent"]["confidence"] == 0.91
    assert response["nudge_messages"]["cta_focused"]["nudge_type"] == "cta_focused"
    assert response["citation_check"][0] is True


def test_drop_off_reasoner_sanitizes_fenced_json():
    class FencedLLM:
        def chat(self, messages, temperature=None):
            return """Here is the analysis:
```json
{
  "primary_reason": "OCR blur",
  "secondary_reasons": ["Low light"],
  "confidence": 0.77,
  "reasoning_chain": "Camera blur detected in dim lighting."
}
```
Extra commentary after JSON."""

    reasoner = DropOffReasoner(FencedLLM())
    result = reasoner.analyze(
        {"stage_dropped": "VKYC", "device_type": "android", "timestamp": "2024-01-01T10:00:00Z"},
        [{"content": "Step 1: hold the device steady."}],
    )

    assert result["primary_reason"] == "OCR blur"
    assert result["secondary_reasons"] == ["Low light"]
    assert result["confidence"] == 0.77
    assert "Camera blur detected" in result["reasoning_chain"]


def test_nudge_generator_derives_stage_from_nested_state():
    captured_prompt: Dict = {}

    class PromptCapturingNudge(NudgeGenerator):
        def _generate_text(self, prompt: str) -> str:
            captured_prompt["prompt"] = prompt
            return "Nudge text"

        def _translate(self, text: str, target_language: str) -> str:
            return text

    generator = PromptCapturingNudge(llm_client=FakeLLMClient(), compliance_validator=ComplianceValidator())
    user_state = {"resume_info": {"stage": "OTP"}}

    result = generator.generate("Need OTP", user_state, nudge_type="cta_focused", language="english")

    assert "OTP" in captured_prompt["prompt"]
    assert result["nudge_type"] == "cta_focused"


def test_analyze_reason_uses_retrieval_and_confidence():
    class StubRetriever:
        def retrieve(self, query, top_k=5, metadata_filter=None, strategy=None, apply_rerank=None, rerank_top_k=None):
            return [
                {
                    "chunk_id": "ctx1",
                    "content": "Step 1: retry with better lighting.",
                    "metadata": {"section": "faq", "stage": "VKYC"},
                    "similarity_score": 0.9,
                    "rerank_score": 0.8,
                }
            ]

    class StubReranker:
        pass

    orchestrator = ChatbotOrchestrator(
        retriever=StubRetriever(),
        reranker=StubReranker(),
        llm_client=FakeLLMClient(),
    )
    # Avoid LLM-based compliance calls
    orchestrator.guardrail_validator.compliance_validator.llm_client = None

    def fake_reasoner(state: Dict, ctx):
        return {
            "primary_reason": "Mock reason",
            "secondary_reasons": ["Mock secondary"],
            "confidence": 0.88,
            "reasoning_chain": "Used stub context",
        }

    orchestrator.drop_off_reasoner.analyze = fake_reasoner

    user_state = {"stage_dropped": "VKYC", "error_codes": ["OCR_FAIL"], "device_type": "Android"}
    result = orchestrator.analyze_reason(user_state)

    assert result["predicted_drop_reason"] == "Mock reason"
    assert result["citations"][0]["chunk_id"] == "ctx1"
    assert result["confidence_score"] > 0


def test_generate_nudges_returns_citations_and_nudges():
    class StubRetriever:
        def retrieve(self, query, top_k=5, metadata_filter=None, strategy=None, apply_rerank=None, rerank_top_k=None):
            return [
                {
                    "chunk_id": "ctx2",
                    "content": "Step 1: check OTP network.",
                    "metadata": {"section": "faq", "stage": "OTP"},
                    "similarity_score": 0.8,
                    "rerank_score": 0.7,
                }
            ]

    class StubReranker:
        pass

    orchestrator = ChatbotOrchestrator(
        retriever=StubRetriever(),
        reranker=StubReranker(),
        llm_client=FakeLLMClient(),
    )
    orchestrator.guardrail_validator.compliance_validator.llm_client = None

    orchestrator.drop_off_reasoner.analyze = lambda state, ctx: {
        "primary_reason": "Network delay",
        "secondary_reasons": ["SMS lag"],
        "confidence": 0.8,
        "reasoning_chain": "Stubbed reasoning",
    }
    orchestrator.nudge_generator.generate_all = lambda reason, user_state, language="english": {
        "cta_focused": {"nudge_text": "Retry OTP", "language": language, "nudge_type": "cta_focused"}
    }

    user_state = {"stage_dropped": "OTP", "language": "english"}
    result = orchestrator.generate_nudges(user_state)

    assert result["nudge_messages"]["cta_focused"]["nudge_text"] == "Retry OTP"
    assert result["citations"][0]["chunk_id"] == "ctx2"
    assert result["confidence_score"] > 0

