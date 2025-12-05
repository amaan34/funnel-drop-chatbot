from typing import Dict, List


class ResponseSynthesizer:
    """
    Combines reasoning, nudges, and retrieved context into a structured response.
    """

    def synthesize(
        self,
        reasoning: Dict,
        nudges: Dict,
        context: List[Dict],
        confidence: float,
        low_confidence: bool = False,
        intent: Dict | None = None,
    ) -> Dict:
        citations = [
            {"chunk_id": c.get("chunk_id"), "section": c.get("metadata", {}).get("section")}
            for c in context[:5]
            if c.get("chunk_id")
        ]

        steps_to_fix = self._extract_steps(context)

        message_prefix = (
            "I'm not fully sure, but here's what I found:\n"
            if low_confidence
            else "Here's what I found:\n"
        )

        conversational_message = (
            f"{message_prefix}"
            f"Primary reason: {reasoning.get('primary_reason')}\n"
            f"Suggested next step: {steps_to_fix[0] if steps_to_fix else 'Please retry the last step with better lighting/network.'}"
        )

        return {
            "predicted_drop_reason": reasoning.get("primary_reason"),
            "explanation": reasoning.get("reasoning_chain"),
            "nudge_messages": nudges,
            "steps_to_fix": steps_to_fix,
            "confidence_score": confidence,
            "citations": citations,
            "conversational_message": conversational_message,
            "intent": intent or {},
            "low_confidence": low_confidence,
        }

    def _extract_steps(self, context: List[Dict]) -> List[str]:
        steps: List[str] = []
        for item in context:
            content = item.get("content", "")
            if "step" in content.lower():
                steps.append(content.split("\n")[0])
        return steps[:5]

