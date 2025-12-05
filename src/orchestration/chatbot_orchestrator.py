from typing import Dict, List, Optional

from src.llm.llm_client import LLMClient
from src.nudge.compliance_validator import ComplianceValidator
from src.nudge.nudge_generator import NudgeGenerator
from src.orchestration.guardrail_validator import GuardrailValidator
from src.orchestration.intent_classifier import IntentClassifier
from src.orchestration.response_synthesizer import ResponseSynthesizer
from src.rag.confidence_scorer import ConfidenceScorer, validate_citations
from src.reasoning.drop_off_reasoner import DropOffReasoner
from src.vector_store.hybrid_retriever import HybridRetriever
from src.vector_store.reranker import CrossEncoderReranker


class ChatbotOrchestrator:
    """
    Orchestrates the full flow: intent → retrieve → rerank → reason → nudge → synthesize → guardrail.
    """

    def __init__(
        self,
        retriever: HybridRetriever,
        reranker: CrossEncoderReranker,
        llm_client: Optional[LLMClient] = None,
    ) -> None:
        self.llm = llm_client or LLMClient()
        self.retriever = retriever
        self.reranker = reranker
        self.confidence_scorer = ConfidenceScorer()
        self.intent_classifier = IntentClassifier(self.llm)
        self.drop_off_reasoner = DropOffReasoner(self.llm)
        compliance_validator = ComplianceValidator(self.llm)
        self.nudge_generator = NudgeGenerator(self.llm, compliance_validator)
        self.response_synthesizer = ResponseSynthesizer()
        self.guardrail_validator = GuardrailValidator(compliance_validator)

    def _fallback_query(self, user_state: Dict) -> str:
        """
        Build a retrieval query when the user did not provide one.
        Uses stage, error codes, and device to pull focused troubleshooting chunks.
        """
        stage = user_state.get("stage_dropped") or user_state.get("stage") or "General"
        errors = ", ".join(user_state.get("error_codes") or []) or "no specific error code"
        device = user_state.get("device_type") or user_state.get("device") or "unspecified device"
        return (
            f"Drop off troubleshooting for stage {stage}. Error codes: {errors}. "
            f"Device: {device}. Return causes, steps, and nudges."
        )

    def process(self, user_state: Dict, query: str) -> Dict:
        working_state = dict(user_state) if user_state else {}

        # 1. Classify intent
        intent = self.intent_classifier.classify(query)
        working_state["intent"] = intent

        # 2. Retrieve context
        stage_filter = working_state.get("stage_dropped") or working_state.get("stage")
        metadata_filter = {"stage": stage_filter} if stage_filter else None
        results = self.retriever.retrieve(
            query,
            top_k=5,
            metadata_filter=metadata_filter,
            strategy="adaptive",
            apply_rerank=True,
            rerank_top_k=10,
        )

        # 3. Confidence
        confidence = self.confidence_scorer.score(results)
        low_confidence = self.confidence_scorer.is_low_confidence(confidence)

        # 4. Reasoning
        reasoning = self.drop_off_reasoner.analyze(working_state, results)

        # 5. Nudges
        nudges = self.nudge_generator.generate_all(
            reasoning.get("primary_reason", ""),
            working_state,
            language=user_state.get("language", "english"),
        )

        # 6. Synthesize
        response = self.response_synthesizer.synthesize(
            reasoning, nudges, results, confidence, low_confidence=low_confidence, intent=intent
        )

        # 7. Guardrails
        response = self.guardrail_validator.validate(response)

        # 8. Citation sanity check
        response["citation_check"] = validate_citations(results)
        return response

    def analyze_reason(self, user_state: Dict, query: Optional[str] = None) -> Dict:
        """
        Run retrieval + reasoning only, for the predict_reason endpoint.
        """
        working_state = dict(user_state) if user_state else {}
        retrieval_query = query or self._fallback_query(working_state)
        stage_filter = working_state.get("stage_dropped") or working_state.get("stage")
        metadata_filter = {"stage": stage_filter} if stage_filter else None

        results = self.retriever.retrieve(
            retrieval_query,
            top_k=5,
            metadata_filter=metadata_filter,
            strategy="adaptive",
            apply_rerank=True,
            rerank_top_k=10,
        )

        confidence = self.confidence_scorer.score(results)
        low_confidence = self.confidence_scorer.is_low_confidence(confidence)

        reasoning = self.drop_off_reasoner.analyze(working_state, results)
        response = self.response_synthesizer.synthesize(
            reasoning,
            nudges={},
            context=results,
            confidence=confidence,
            low_confidence=low_confidence,
            intent=None,
        )
        response = self.guardrail_validator.validate(response)
        response["citation_check"] = validate_citations(results)
        response["reasoning"] = reasoning
        return response

    def generate_nudges(self, user_state: Dict, language: str = "english", query: Optional[str] = None) -> Dict:
        """
        Run retrieval + reasoning + nudge generation, for the nudge_user endpoint.
        """
        working_state = dict(user_state) if user_state else {}
        retrieval_query = query or self._fallback_query(working_state)
        stage_filter = working_state.get("stage_dropped") or working_state.get("stage")
        metadata_filter = {"stage": stage_filter} if stage_filter else None

        results = self.retriever.retrieve(
            retrieval_query,
            top_k=5,
            metadata_filter=metadata_filter,
            strategy="adaptive",
            apply_rerank=True,
            rerank_top_k=10,
        )

        confidence = self.confidence_scorer.score(results)
        low_confidence = self.confidence_scorer.is_low_confidence(confidence)

        reasoning = self.drop_off_reasoner.analyze(working_state, results)
        nudges = self.nudge_generator.generate_all(
            reasoning.get("primary_reason", ""),
            working_state,
            language=user_state.get("language", language),
        )

        response = self.response_synthesizer.synthesize(
            reasoning,
            nudges,
            results,
            confidence,
            low_confidence=low_confidence,
            intent=None,
        )
        response = self.guardrail_validator.validate(response)
        response["citation_check"] = validate_citations(results)
        response["reasoning"] = reasoning
        response["nudge_messages"] = nudges
        return response

